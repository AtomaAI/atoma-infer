//! Allocation-phase device state: weights, KV caches and byte buffers, allocated and filled on
//! the setup stream strictly before the first capture.
//!
//! The raw setup-stream handle stays inside this module and the kernels it calls; the runner's
//! orchestration never touches a raw stream, and the capture stream's handle appears only in the
//! descriptor seam.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use cudarc::driver::{result, CudaSlice, CudaStream, DevicePtr};

use crate::dims::{ModelDims, BF16_BYTES};
use crate::gpu::kernels::StepKernels;
use crate::layout::kv_cache_bytes_each;

const WEIGHT_SCALE: f32 = 0.05;
/// bf16(1.0), the RMSNorm gain everywhere so normed activations stay unit-scale.
const BF16_ONE: u16 = 0x3F80;

/// Weight addresses without the per-layer KV caches (which [`kv`] provides).
pub struct WeightPtrs {
    pub embedding: u64,
    pub final_norm: u64,
    pub lm_head: u64,
    pub layers: Vec<LayerWeights>,
}

/// One layer's weight addresses.
pub struct LayerWeights {
    pub w_qkv: u64,
    pub w_o: u64,
    pub w_gate: u64,
    pub w_up: u64,
    pub w_down: u64,
    pub rms1: u64,
    pub rms2: u64,
}

/// The KV pool: the owning slices and each layer's (K, V) cache addresses.
pub type KvPool = (Vec<CudaSlice<u8>>, Vec<(u64, u64)>);

/// Allocates and fills every weight tensor, returning the owning slices and the address table.
pub fn model(
    kernels: &StepKernels,
    stream: &Arc<CudaStream>,
    dims: &ModelDims,
    seed: u64,
) -> Result<(Vec<CudaSlice<u8>>, WeightPtrs)> {
    let raw = stream.cu_stream();
    let mut slices = Vec::new();
    let mut next_seed = seed;
    let mut fill = |elements: usize| -> Result<u64> {
        let slice = bytes(stream, elements * BF16_BYTES)?;
        let ptr = addr(&slice, stream);
        next_seed = next_seed.wrapping_add(1);
        unsafe { kernels.fill_random_bf16(raw, ptr, elements, next_seed, WEIGHT_SCALE) }?;
        slices.push(slice);
        Ok(ptr)
    };

    let embedding = fill(dims.vocab * dims.hidden)?;
    let lm_head = fill(dims.vocab * dims.hidden)?;
    let mut layers = Vec::new();
    for _ in 0..dims.num_layers {
        layers.push(LayerWeights {
            w_qkv: fill(dims.qkv_out() * dims.hidden)?,
            w_o: fill(dims.hidden * dims.hidden)?,
            w_gate: fill(dims.ffn * dims.hidden)?,
            w_up: fill(dims.ffn * dims.hidden)?,
            w_down: fill(dims.hidden * dims.ffn)?,
            rms1: 0,
            rms2: 0,
        });
    }

    // RMSNorm gains are 1.0 so every normed activation stays unit-scale (bf16 overflows near
    // 3e4; the day-0 engine mock died exactly there).
    let ones = vec![BF16_ONE; dims.hidden];
    let mut gain = || -> Result<u64> {
        let slice = bytes(stream, dims.hidden * BF16_BYTES)?;
        let ptr = addr(&slice, stream);
        unsafe { result::memcpy_htod_async(ptr, &ones, raw) }
            .map_err(|e| anyhow!("uploading RMSNorm gains: {:?}", e.0))?;
        slices.push(slice);
        Ok(ptr)
    };
    for layer in &mut layers {
        layer.rms1 = gain()?;
        layer.rms2 = gain()?;
    }
    let final_norm = gain()?;

    Ok((
        slices,
        WeightPtrs {
            embedding,
            final_norm,
            lm_head,
            layers,
        },
    ))
}

/// Allocates the per-layer paged K and V caches, pre-filled with unit-scale randoms so
/// "historical" positions hold deterministic data both runs share.
pub fn kv(
    kernels: &StepKernels,
    stream: &Arc<CudaStream>,
    dims: &ModelDims,
    total_blocks: usize,
    page_block: usize,
    seed: u64,
) -> Result<KvPool> {
    let raw = stream.cu_stream();
    let cache_bytes = kv_cache_bytes_each(dims, total_blocks, page_block);
    let elements = cache_bytes / BF16_BYTES;
    let mut slices = Vec::new();
    let mut ptrs = Vec::new();
    for layer in 0..dims.num_layers {
        let mut cache = |salt: u64| -> Result<u64> {
            let slice = bytes(stream, cache_bytes)?;
            let ptr = addr(&slice, stream);
            // 0x4B56 is "KV": keeps cache fills disjoint from the weight-fill seed sequence.
            unsafe {
                kernels.fill_random_bf16(raw, ptr, elements, seed ^ 0x4B56 ^ salt, WEIGHT_SCALE)
            }?;
            slices.push(slice);
            Ok(ptr)
        };
        let k = cache(2 * layer as u64)?;
        let v = cache(2 * layer as u64 + 1)?;
        ptrs.push((k, v));
    }
    Ok((slices, ptrs))
}

/// Allocates `count` zeroed bytes on the setup stream.
pub fn bytes(stream: &Arc<CudaStream>, count: usize) -> Result<CudaSlice<u8>> {
    stream
        .alloc_zeros::<u8>(count.max(1))
        .map_err(|e| anyhow!("allocating {count} bytes: {:?}", e.0))
}

/// The device address of a slice. With event tracking disabled at context creation, the guard
/// is a no-op and the address is stable for the slice's lifetime.
pub fn addr(slice: &CudaSlice<u8>, stream: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}

/// Synchronous D2H read of `buf.len()` bytes from `src`.
pub fn read_back(src: u64, buf: &mut [u8]) -> Result<()> {
    unsafe { result::memcpy_dtoh_sync(buf, src) }.map_err(|e| anyhow!("D2H read: {:?}", e.0))
}

/// Waits for everything enqueued on the setup stream — allocation-phase fills and uploads only.
pub fn synchronize(stream: &Arc<CudaStream>) -> Result<()> {
    unsafe { result::stream::synchronize(stream.cu_stream()) }
        .map_err(|e| anyhow!("setup-stream synchronize: {:?}", e.0))
}

/// The device's free memory in bytes, for the graph-overhead and soak-delta measurements.
pub fn free_memory() -> Result<i64> {
    let (free, _total) = result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;
    Ok(free as i64)
}
