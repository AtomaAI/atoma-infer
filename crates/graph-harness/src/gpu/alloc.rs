//! Allocation-phase device state: weights, the KV pool, per-cell statics and byte buffers,
//! allocated and filled on the setup stream strictly before the first capture.
//!
//! The raw setup-stream handle stays inside this module and the fill kernels it calls; the
//! runner's orchestration never touches a raw stream, and the capture stream's handle appears
//! only in the descriptor seam.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use atoma_runtime::session::BakedBuffers;
use cudarc::driver::{result, CudaSlice, CudaStream, DevicePtr};

use crate::dims::{ModelDims, BF16_BYTES};
use crate::gpu::kernels::StepKernels;
use crate::gpu::step::{InputPtrs, KvCachePtrs, LayerWeights, StaticPtrs, WeightPtrs};
use crate::layout::{kv_cache_bytes_each, StaticSizes};

const WEIGHT_SCALE: f32 = 0.05;
/// bf16(1.0), the RMSNorm gain everywhere so normed activations stay unit-scale.
const BF16_ONE: u16 = 0x3F80;
/// Salted into every KV-cache fill seed ("KV") so cache fills stay disjoint from the weight-fill
/// seed sequence.
const KV_SEED_SALT: u64 = 0x4B56;

/// Allocates and fills device state on the setup stream.
pub struct Allocator<'a> {
    kernels: &'a StepKernels,
    stream: &'a Arc<CudaStream>,
}

impl<'a> Allocator<'a> {
    pub fn new(kernels: &'a StepKernels, stream: &'a Arc<CudaStream>) -> Self {
        Self { kernels, stream }
    }

    /// Allocates and fills every weight tensor from `seed`.
    pub fn weights(&self, dims: &ModelDims, seed: u64) -> Result<Weights> {
        let mut slices = Vec::new();
        let mut seed = seed;
        let mut random = |slices: &mut Vec<CudaSlice<u8>>, elements: usize| -> Result<u64> {
            seed = seed.wrapping_add(1);
            let seed = seed;
            self.filled(slices, elements * BF16_BYTES, |ptr| {
                self.fill_random(ptr, elements, seed)
            })
        };
        // RMSNorm gains are 1.0 so every normed activation stays unit-scale (bf16 overflows near
        // 3e4; the day-0 engine mock died exactly there).
        let ones = vec![BF16_ONE; dims.hidden];
        let gain = |slices: &mut Vec<CudaSlice<u8>>| -> Result<u64> {
            self.filled(slices, dims.hidden * BF16_BYTES, |ptr| {
                self.upload(ptr, &ones)
            })
        };

        let embedding = random(&mut slices, dims.vocab * dims.hidden)?;
        let lm_head = random(&mut slices, dims.vocab * dims.hidden)?;
        let mut layers = Vec::with_capacity(dims.num_layers);
        for _ in 0..dims.num_layers {
            layers.push(LayerWeights {
                w_qkv: random(&mut slices, dims.qkv_out() * dims.hidden)?,
                w_o: random(&mut slices, dims.hidden * dims.hidden)?,
                w_gate: random(&mut slices, dims.ffn * dims.hidden)?,
                w_up: random(&mut slices, dims.ffn * dims.hidden)?,
                w_down: random(&mut slices, dims.hidden * dims.ffn)?,
                rms1: gain(&mut slices)?,
                rms2: gain(&mut slices)?,
            });
        }
        let final_norm = gain(&mut slices)?;

        Ok(Weights {
            slices,
            ptrs: WeightPtrs {
                embedding,
                final_norm,
                lm_head,
                layers,
            },
        })
    }

    /// Allocates the per-layer paged K and V caches, pre-filled with unit-scale randoms so
    /// "historical" positions hold deterministic data both runs share.
    pub fn kv_pool(
        &self,
        dims: &ModelDims,
        total_blocks: usize,
        page_block: usize,
        seed: u64,
    ) -> Result<KvPool> {
        let cache_bytes = kv_cache_bytes_each(dims, total_blocks, page_block);
        let elements = cache_bytes / BF16_BYTES;
        let mut slices = Vec::new();
        let mut layers = Vec::with_capacity(dims.num_layers);
        for layer in 0..dims.num_layers as u64 {
            let mut cache = |salt: u64| -> Result<u64> {
                self.filled(&mut slices, cache_bytes, |ptr| {
                    self.fill_random(ptr, elements, seed ^ KV_SEED_SALT ^ salt)
                })
            };
            layers.push(KvCachePtrs {
                k_cache: cache(2 * layer)?,
                v_cache: cache(2 * layer + 1)?,
            });
        }
        Ok(KvPool { slices, layers })
    }

    /// Allocates one cell's static buffers, zeroed.
    pub fn cell_statics(&self, sizes: &StaticSizes) -> Result<CellStatics> {
        Ok(CellStatics {
            inputs: self.input_buffers(sizes)?,
            staging: self.input_buffers(sizes)?,
            logits: self.bytes(sizes.logits)?,
            argmax: self.bytes(sizes.argmax)?,
            softmax_lse: self.bytes(sizes.softmax_lse)?,
            lse_accum: self.bytes(sizes.lse_accum.max(4))?,
            o_accum: self.bytes(sizes.o_accum.max(4))?,
        })
    }

    /// Allocates `count` zeroed bytes (at least one) on the setup stream.
    pub fn bytes(&self, count: usize) -> Result<CudaSlice<u8>> {
        self.stream
            .alloc_zeros::<u8>(count.max(1))
            .map_err(|e| anyhow!("allocating {count} bytes: {:?}", e.0))
    }

    /// Allocates `count` zeroed f32 values (at least one) on the setup stream.
    #[cfg(feature = "nccl")]
    pub fn f32s(&self, count: usize) -> Result<CudaSlice<f32>> {
        self.stream
            .alloc_zeros::<f32>(count.max(1))
            .map_err(|e| anyhow!("allocating {count} f32 values: {:?}", e.0))
    }

    /// Waits for everything enqueued on the setup stream — allocation-phase fills and uploads
    /// only.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("setup-stream synchronize: {:?}", e.0))
    }

    fn input_buffers(&self, sizes: &StaticSizes) -> Result<InputBuffers> {
        Ok(InputBuffers {
            token_ids: self.bytes(sizes.token_ids)?,
            seqlens_k: self.bytes(sizes.seqlens_k)?,
            block_table: self.bytes(sizes.block_table)?,
            slot_mapping: self.bytes(sizes.slot_mapping)?,
        })
    }

    /// Allocates `bytes`, runs `fill` on the new buffer's address, and keeps the buffer alive in
    /// `slices`. Returns the address.
    fn filled(
        &self,
        slices: &mut Vec<CudaSlice<u8>>,
        bytes: usize,
        fill: impl FnOnce(u64) -> Result<()>,
    ) -> Result<u64> {
        let slice = self.bytes(bytes)?;
        let ptr = addr(&slice);
        fill(ptr)?;
        slices.push(slice);
        Ok(ptr)
    }

    /// Fills `elements` bf16 values at `ptr` with deterministic uniforms from `seed`.
    fn fill_random(&self, ptr: u64, elements: usize, seed: u64) -> Result<()> {
        // SAFETY: `ptr` addresses `elements` bf16 values just allocated on this stream's device.
        unsafe {
            self.kernels.fill_random_bf16(
                self.stream.cu_stream(),
                ptr,
                elements,
                seed,
                WEIGHT_SCALE,
            )
        }
    }

    /// Uploads `src` to `ptr` on the setup stream.
    fn upload<T>(&self, ptr: u64, src: &[T]) -> Result<()> {
        // SAFETY: `ptr` addresses a buffer just allocated for `src` on this stream's device, and
        // pageable H2D is host-synchronous, so `src` outlives the copy.
        unsafe { result::memcpy_htod_async(ptr, src, self.stream.cu_stream()) }
            .map_err(|e| anyhow!("uploading RMSNorm gains: {:?}", e.0))
    }
}

/// The device address of a slice. With event tracking disabled at context creation, the guard
/// is a no-op and the address is stable for the slice's lifetime.
pub fn addr<T>(slice: &CudaSlice<T>) -> u64 {
    let (ptr, _guard) = slice.device_ptr(slice.stream());
    ptr
}

/// The weight tensors: the owning buffers and their address table.
pub struct Weights {
    slices: Vec<CudaSlice<u8>>,
    pub ptrs: WeightPtrs,
}

impl Weights {
    /// The live address of every weight buffer.
    pub fn addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.slices.iter().map(addr)
    }
}

/// The KV pool: the owning buffers and each layer's cache addresses.
pub struct KvPool {
    slices: Vec<CudaSlice<u8>>,
    pub layers: Vec<KvCachePtrs>,
}

impl KvPool {
    /// The live address of every cache buffer.
    pub fn addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.slices.iter().map(addr)
    }
}

/// The four per-step input buffers, in either of their homes.
pub struct InputBuffers {
    token_ids: CudaSlice<u8>,
    seqlens_k: CudaSlice<u8>,
    block_table: CudaSlice<u8>,
    slot_mapping: CudaSlice<u8>,
}

impl InputBuffers {
    fn addresses(&self) -> InputPtrs {
        InputPtrs {
            token_ids: addr(&self.token_ids),
            seqlens_k: addr(&self.seqlens_k),
            block_table: addr(&self.block_table),
            slot_mapping: addr(&self.slot_mapping),
        }
    }

    fn into_vec(self) -> Vec<CudaSlice<u8>> {
        let Self {
            token_ids,
            seqlens_k,
            block_table,
            slot_mapping,
        } = self;
        vec![token_ids, seqlens_k, block_table, slot_mapping]
    }
}

/// One cell's static device buffers: the graph's inputs and their staging mirrors, the two
/// outputs, and the attention workspaces.
pub struct CellStatics {
    inputs: InputBuffers,
    staging: InputBuffers,
    logits: CudaSlice<u8>,
    argmax: CudaSlice<u8>,
    softmax_lse: CudaSlice<u8>,
    lse_accum: CudaSlice<u8>,
    o_accum: CudaSlice<u8>,
}

impl CellStatics {
    /// The address table the step reads.
    pub fn addresses(&self) -> StaticPtrs {
        StaticPtrs {
            inputs: self.inputs.addresses(),
            staging: self.staging.addresses(),
            logits: addr(&self.logits),
            argmax: addr(&self.argmax),
            softmax_lse: addr(&self.softmax_lse),
            lse_accum: addr(&self.lse_accum),
            o_accum: addr(&self.o_accum),
        }
    }

    /// Sorts the buffers into the roles a recording bakes them in: staging is the host-written
    /// input, logits and argmax are read back, and everything else is graph-internal.
    pub fn into_baked(self) -> BakedBuffers {
        let Self {
            inputs,
            staging,
            logits,
            argmax,
            softmax_lse,
            lse_accum,
            o_accum,
        } = self;
        let mut workspaces = inputs.into_vec();
        workspaces.extend([softmax_lse, lse_accum, o_accum]);
        BakedBuffers {
            inputs: staging.into_vec(),
            outputs: vec![logits, argmax],
            workspaces,
        }
    }
}
