use crate::ffi;
use crate::ops::{SwapBlockCpuToGpuOp, SwapBlockGpuToCpuOp, SwapBlockOp};
use candle_core::CpuStorage;
use candle_core::{
    backend::BackendStorage,
    cuda::{
        cudarc::driver::{DevicePtrMut, DeviceRepr},
        CudaStorageSlice,
    },
    cuda_backend::CudaDType,
    CudaStorage, DType, Device, IndexOp, InplaceOp2, InplaceOp3, Layout, Result, Tensor,
};
use half::{bf16, f16};
use std::collections::HashMap;

/// Swaps blocks from `src` to `dst` tensors, using the block_mapping.
/// Both `src` and `dst` tensors must have the same dtype, and either be on
/// the same cuda device, or one in either cpu and the other in a cuda device.
/// Moreover, both `src` and `dst` have shape `[num_blocks, block_size, num_kv_heads, head_size]`,
/// where `num_blocks` is the total number of blocks available for the current device.
pub fn swap_blocks(
    src: &Tensor,
    dst: &mut Tensor,
    block_mapping: &HashMap<u32, u32>,
) -> Result<()> {
    let t_size_in_bytes = src.dtype().size_in_bytes();
    // NOTE: the rhs of * should be equivalent to src.i(0)?.elem_count()
    // but in this way, we do not need to clone the underlying `Tensor`
    let block_size_in_bytes =
        src.dtype().size_in_bytes() * src.dims()[1..].iter().product::<usize>();
    let src_device = src.device();
    let dst_device = dst.device();
    match (src_device, dst_device) {
        (Device::Cuda(src_device), Device::Cuda(dst_device)) => {
            if crate::utils::device_ordinal(src_device) != crate::utils::device_ordinal(dst_device)
            {
                candle_core::bail!(
                    "swap_blocks: Both src and dst tensors should be on the same device to swap"
                )
            }

            for (src_block, dst_block) in block_mapping.iter() {
                let swap_op = SwapBlockOp {
                    block_size_in_bytes,
                    src_offset: (*src_block as usize) * block_size_in_bytes,
                    dst_offset: (*dst_block as usize) * block_size_in_bytes,
                };
                dst.inplace_op2(src, &swap_op)?;
            }
        }
        (Device::Cpu, Device::Cuda(_)) => {
            let (src, _src_l) = src.storage_and_layout();
            let src_slice = match &*src {
                candle_core::Storage::Cpu(CpuStorage::BF16(ref src_c)) => {
                    crate::ops::utils::cast_slice(src_c.as_slice())
                }
                candle_core::Storage::Cpu(CpuStorage::F16(ref src_c)) => {
                    crate::ops::utils::cast_slice(src_c.as_slice())
                }
                _ => {
                    candle_core::bail!(
                        "swap_blocks: Invalid combination of src and dst tensors storage to swap"
                    )
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as usize) * block_size_in_bytes;
                let dst_offset = (*dst_block as usize) * block_size_in_bytes;
                let swap_block_cpu_to_gpu_op = SwapBlockCpuToGpuOp {
                    src_slice: &src_slice[src_offset..src_offset + block_size_in_bytes],
                    block_size_in_bytes,
                    src_offset,
                    dst_offset,
                };
                dst.inplace_op1(&swap_block_cpu_to_gpu_op)?;
            }
        }
        (Device::Cuda(src_device), Device::Cpu) => {
            let (src, src_l) = src.storage_and_layout();
            let src_slice = match &*src {
                candle_core::Storage::Cuda(src_c) => match &src_c.slice {
                    CudaStorageSlice::BF16(src_c) => unsafe {
                        src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda(
                                "swap_blocks: unable to transmute src_c".to_string().into(),
                            )
                        })?
                    },
                    CudaStorageSlice::F16(src_c) => unsafe {
                        src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda(
                                "swap_blocks: unable to transmute src_c".to_string().into(),
                            )
                        })?
                    },
                    _ => {
                        candle_core::bail!(
                            "swap_blocks:Invalid dtype for cuda src tensor, expected f16/bf16, got {:?}",
                            src_c.dtype()
                        )
                    }
                },
                _ => {
                    candle_core::bail!(
                        "swap_blocks: Invalid combination of src and dst tensors storage to swap"
                    )
                }
            };

            // NOTE: We need to do the conversion here, as we cast the slice to u8,
            // but the layout is still in the original dtype.
            let src_slice = src_slice.slice(src_l.start_offset() * t_size_in_bytes..);
            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as usize) * block_size_in_bytes;
                let dst_offset = (*dst_block as usize) * block_size_in_bytes;
                let swap_block_gpu_to_cpu_op = SwapBlockGpuToCpuOp {
                    src_slice: src_slice.slice(src_offset..src_offset + block_size_in_bytes),
                    cuda_device: src_device,
                    block_size_in_bytes,
                    dst_offset,
                };
                dst.inplace_op1(&swap_block_gpu_to_cpu_op)?;
            }
        }
        _ => {
            candle_core::bail!("swap_blocks: Either src and dst are on the same cuda device, or src and dst are on cpu and cuda devices, alternately")
        }
    }

    Ok(())
}

/// Copies the mapped blocks within a single cache tensor.
///
/// The destination is acquired through Candle's in-place API, so its mutable pointer guard records
/// this kernel write on the cache tensor's CUDA stream when the launch has been enqueued.
struct CopyBlocksOp {
    num_pairs: i64,
    numel_per_block: i64,
}

impl InplaceOp2 for CopyBlocksOp {
    fn name(&self) -> &'static str {
        "copy_blocks"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("copy_blocks requires CUDA tensors")
    }

    fn cuda_fwd(
        &self,
        cache: &mut CudaStorage,
        cache_layout: &Layout,
        block_mapping: &CudaStorage,
        block_mapping_layout: &Layout,
    ) -> Result<()> {
        match cache.dtype() {
            DType::F16 => launch_copy_blocks::<f16>(
                self,
                cache,
                cache_layout,
                block_mapping,
                block_mapping_layout,
            ),
            DType::BF16 => launch_copy_blocks::<bf16>(
                self,
                cache,
                cache_layout,
                block_mapping,
                block_mapping_layout,
            ),
            dtype => candle_core::bail!("copy_blocks only supports f16/bf16 caches, got {dtype:?}"),
        }
    }
}

fn launch_copy_blocks<T: CudaDType + DeviceRepr>(
    op: &CopyBlocksOp,
    cache: &mut CudaStorage,
    cache_layout: &Layout,
    block_mapping: &CudaStorage,
    block_mapping_layout: &Layout,
) -> Result<()> {
    let stream = cache.device().cuda_stream();
    let mut cache = cache
        .as_cuda_slice_mut::<T>()?
        .slice_mut(cache_layout.start_offset()..);
    let (cache_ptr, _cache_write_guard) = cache.device_ptr_mut(&stream);
    let block_mapping = block_mapping
        .as_cuda_slice::<i64>()?
        .slice(block_mapping_layout.start_offset()..);
    let (block_mapping_ptr, _block_mapping_guard) = block_mapping.view_ptr(&stream);

    unsafe {
        ffi::copy_blocks_cache(
            cache_ptr as *mut core::ffi::c_void,
            block_mapping_ptr as *const core::ffi::c_void,
            op.num_pairs,
            op.numel_per_block,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}

/// Launches the `copy_blocks_kernel` on the given `key_caches` and `value_caches`,
/// following the `block_mapping`, to copy the blocks on both `key_cache` and `value_cache`.
///
/// For each block pair `[src_block_index, dst_block_index]`, the source block is copied within
/// every key and value cache. `block_mapping` must be an i64 CUDA tensor with shape
/// `[num_pairs, 2]`.
pub fn copy_blocks(
    key_caches: &[&mut Tensor],
    value_caches: &[&mut Tensor],
    block_mapping: Tensor,
) -> Result<()> {
    if key_caches.len() != value_caches.len() {
        candle_core::bail!("key_caches and value_caches must have the same length")
    }
    if key_caches.is_empty() {
        return Ok(());
    }
    if block_mapping.rank() != 2 || block_mapping.dims()[1] != 2 {
        candle_core::bail!("block_mapping must have shape [num_pairs, 2]")
    }
    if block_mapping.dtype() != DType::I64 {
        candle_core::bail!("block_mapping must have dtype i64")
    }

    let cuda_device = match key_caches[0].device() {
        Device::Cuda(device) => device,
        _ => candle_core::bail!("key_caches and value_caches must be CUDA tensors"),
    };
    let dtype = key_caches[0].dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) {
        candle_core::bail!("copy_blocks only supports f16/bf16 caches, got {dtype:?}")
    }
    match block_mapping.device() {
        Device::Cuda(device)
            if crate::utils::device_ordinal(cuda_device)
                == crate::utils::device_ordinal(device) => {}
        _ => candle_core::bail!(
            "key_caches, value_caches and block_mapping must be on the same CUDA device"
        ),
    }

    let numel_per_block = key_caches[0]
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .map_err(|_| candle_core::Error::Msg("cache block is too large".into()))?;
    let op = CopyBlocksOp {
        num_pairs: block_mapping.dims()[0] as i64,
        numel_per_block,
    };

    for cache in key_caches.iter().chain(value_caches.iter()) {
        match cache.device() {
            Device::Cuda(device)
                if cache.dtype() == dtype
                    && crate::utils::device_ordinal(cuda_device)
                        == crate::utils::device_ordinal(device) => {}
            _ => candle_core::bail!(
                "key_caches and value_caches must have the same dtype and CUDA device"
            ),
        }
        cache.inplace_op2(&block_mapping, &op)?;
    }
    Ok(())
}

/// Writes a source tensor into a single paged KV cache according to a slot mapping.
///
/// The cache is the mutable operand of Candle's in-place operation, which retains a write guard
/// until this CUDA launch is submitted.
struct ReshapeAndCacheFlashOp {
    block_stride: i64,
    num_tokens: i64,
    num_heads: i64,
    head_size: i64,
    block_size: i64,
    source_stride: i64,
    dtype: u32,
}

impl InplaceOp3 for ReshapeAndCacheFlashOp {
    fn name(&self) -> &'static str {
        "reshape_and_cache_flash"
    }

    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<()> {
        candle_core::bail!("reshape_and_cache_flash requires CUDA tensors")
    }

    fn cuda_fwd(
        &self,
        cache: &mut CudaStorage,
        cache_layout: &Layout,
        source: &CudaStorage,
        source_layout: &Layout,
        slot_mapping: &CudaStorage,
        slot_mapping_layout: &Layout,
    ) -> Result<()> {
        match cache.dtype() {
            DType::F16 => launch_reshape_and_cache_flash::<f16>(
                self,
                cache,
                cache_layout,
                source,
                source_layout,
                slot_mapping,
                slot_mapping_layout,
            ),
            DType::BF16 => launch_reshape_and_cache_flash::<bf16>(
                self,
                cache,
                cache_layout,
                source,
                source_layout,
                slot_mapping,
                slot_mapping_layout,
            ),
            dtype => candle_core::bail!(
                "reshape_and_cache_flash only supports f16/bf16 caches, got {dtype:?}"
            ),
        }
    }
}

fn launch_reshape_and_cache_flash<T: CudaDType + DeviceRepr>(
    op: &ReshapeAndCacheFlashOp,
    cache: &mut CudaStorage,
    cache_layout: &Layout,
    source: &CudaStorage,
    source_layout: &Layout,
    slot_mapping: &CudaStorage,
    slot_mapping_layout: &Layout,
) -> Result<()> {
    let stream = cache.device().cuda_stream();
    let mut cache = cache
        .as_cuda_slice_mut::<T>()?
        .slice_mut(cache_layout.start_offset()..);
    let (cache_ptr, _cache_write_guard) = cache.device_ptr_mut(&stream);
    let source = source
        .as_cuda_slice::<T>()?
        .slice(source_layout.start_offset()..);
    let (source_ptr, _source_guard) = source.view_ptr(&stream);
    let slot_mapping = slot_mapping
        .as_cuda_slice::<i64>()?
        .slice(slot_mapping_layout.start_offset()..);
    let (slot_mapping_ptr, _slot_mapping_guard) = slot_mapping.view_ptr(&stream);

    unsafe {
        ffi::reshape_and_cache_flash_cache(
            source_ptr as *const core::ffi::c_void,
            cache_ptr as *mut core::ffi::c_void,
            slot_mapping_ptr as *const i64,
            op.block_stride,
            op.num_tokens,
            op.num_heads,
            op.head_size,
            op.block_size,
            op.source_stride,
            op.dtype,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}

/// Launches the `reshape_and_cache_kernel_flash` on the given key and value caches, respecting a
/// slot mapping.
pub fn reshape_and_cache_flash(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if key.dtype() != value.dtype()
        || key.dtype() != key_cache.dtype()
        || key.dtype() != value_cache.dtype()
    {
        candle_core::bail!("key, value, key_cache and value_cache must have the same dtype")
    }
    let dtype = match key.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        dtype => candle_core::bail!(
            "reshape_and_cache_flash only supports f16/bf16 tensors, got {dtype:?}"
        ),
    };
    if slot_mapping.dtype() != DType::I64 {
        candle_core::bail!("slot_mapping must have dtype i64")
    }

    let cuda_device = match key.device() {
        Device::Cuda(device) => device,
        _ => candle_core::bail!("key must be a CUDA tensor"),
    };
    for tensor in [value, key_cache, value_cache, slot_mapping] {
        match tensor.device() {
            Device::Cuda(device)
                if crate::utils::device_ordinal(cuda_device)
                    == crate::utils::device_ordinal(device) => {}
            _ => candle_core::bail!(
                "key, value, key_cache, value_cache and slot_mapping must be on the same CUDA device"
            ),
        }
    }

    if key.rank() != 3 || value.rank() != 3 {
        candle_core::bail!("key and value tensors must have rank 3")
    }
    if key_cache.rank() != 4 || value_cache.rank() != 4 {
        candle_core::bail!("key_cache and value_cache tensors must have rank 4")
    }
    let &[num_tokens, num_heads, head_size] = key.dims() else {
        unreachable!("key rank was checked above")
    };
    let &[num_blocks, block_size, cache_heads, cache_head_size] = key_cache.dims() else {
        unreachable!("key_cache rank was checked above")
    };
    if value.dims() != [num_tokens, num_heads, head_size]
        || cache_heads != num_heads
        || cache_head_size != head_size
        || value_cache.dims() != [num_blocks, block_size, num_heads, head_size]
    {
        candle_core::bail!("key, value, key_cache and value_cache have incompatible shapes")
    }
    if slot_mapping.dims() != [num_tokens] {
        candle_core::bail!("slot_mapping must have shape [{num_tokens}]")
    }
    let block_stride = key_cache.stride()[0];
    if block_stride != value_cache.stride()[0] {
        candle_core::bail!("key_cache and value_cache must have the same block stride")
    }

    let key_op = ReshapeAndCacheFlashOp {
        block_stride: block_stride as i64,
        num_tokens: num_tokens as i64,
        num_heads: num_heads as i64,
        head_size: head_size as i64,
        block_size: block_size as i64,
        source_stride: key.stride()[0] as i64,
        dtype,
    };
    key_cache.inplace_op3(key, slot_mapping, &key_op)?;
    let value_op = ReshapeAndCacheFlashOp {
        source_stride: value.stride()[0] as i64,
        ..key_op
    };
    value_cache.inplace_op3(value, slot_mapping, &value_op)
}
