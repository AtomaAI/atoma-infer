#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>

namespace vllm {

// Each launch owns one cache tensor, so Candle can acquire a mutable pointer guard for that
// tensor and record the write after the launch is enqueued.
template <typename scalar_t>
__global__ void copy_blocks_kernel(scalar_t* cache,
                                   const int64_t* __restrict__ block_mapping,
                                   const int64_t numel_per_block) {
    const int pair_idx = blockIdx.y;
    int64_t src_block_number = block_mapping[2 * pair_idx];
    int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

    const int64_t src_block_offset = src_block_number * numel_per_block;
    const int64_t dst_block_offset = dst_block_number * numel_per_block;
    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        int64_t src_offset = src_block_offset + i;
        int64_t dst_offset = dst_block_offset + i;
        cache[dst_offset] = cache[src_offset];
    }
}

}  // namespace vllm

// f16 and bf16 have the same 16-bit representation width, so this kernel copies their bits.
extern "C" {
cudaError_t copy_blocks_cache(
    void* cache,
    const void* block_mapping,
    int64_t num_pairs,
    int64_t numel_per_block,
    cudaStream_t stream) {
    dim3 grid(1, num_pairs);
    dim3 block(std::min(int64_t(1024), int64_t(numel_per_block)));

    vllm::copy_blocks_kernel<int16_t><<<grid, block, 0, stream>>>(
        (int16_t*)cache,
        (const int64_t*)block_mapping,
        numel_per_block);
    return cudaGetLastError();
}
}

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_flash_single_kernel(
    const scalar_t* __restrict__ source,
    scalar_t* __restrict__ cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t block_stride,
    const int64_t source_stride,
    const int64_t num_heads,
    const int64_t head_size,
    const int64_t block_size) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n = num_heads * head_size;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t src_idx = token_idx * source_stride + i;
        const int head_idx = i / head_size;
        const int head_offset = i % head_size;
        const int64_t tgt_idx = block_idx * block_stride +
                                block_offset * num_heads * head_size +
                                head_idx * head_size + head_offset;
        cache[tgt_idx] = source[src_idx];
    }
}

#define CALL_RESHAPE_AND_CACHE_FLASH_SINGLE(T)                           \
    vllm::reshape_and_cache_flash_single_kernel<T><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const T*>(source),                              \
        reinterpret_cast<T*>(cache),                                     \
        slot_mapping,                                                     \
        block_stride,                                                     \
        source_stride,                                                    \
        num_heads,                                                        \
        head_size,                                                        \
        block_size);
}  // namespace vllm

extern "C" cudaError_t reshape_and_cache_flash_cache(
    const void* source,
    void* cache,
    const int64_t* slot_mapping,
    int64_t block_stride,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t source_stride,
    uint32_t dtype,
    cudaStream_t stream
) {
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, int64_t(512)));

    if (dtype == 0) {
        CALL_RESHAPE_AND_CACHE_FLASH_SINGLE(uint16_t);
    } else if (dtype == 1) {
        CALL_RESHAPE_AND_CACHE_FLASH_SINGLE(__nv_bfloat16);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
