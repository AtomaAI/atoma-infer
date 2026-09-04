// The decode step's own kernels: the per-row and elementwise work between its GEMMs and its
// attention call. Every launcher takes the caller's stream and returns the launch status, and
// every kernel reduces in f32 in a fixed order, so an eager launch and a replayed one are
// bit-identical. Activations are bf16; cos and sin tables and every accumulator are f32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16_rn(x); }

unsigned int blocks_covering(int64_t elements) {
    return static_cast<unsigned int>((elements + kThreads - 1) / kThreads);
}

// One block per token; the threads stride over the embedding row.
__global__ void embedding_gather_kernel(const __nv_bfloat16* __restrict__ table,
                                        const uint32_t* __restrict__ token_ids,
                                        __nv_bfloat16* __restrict__ out, int64_t hidden) {
    const int64_t token = blockIdx.x;
    const __nv_bfloat16* row = table + static_cast<int64_t>(token_ids[token]) * hidden;
    __nv_bfloat16* out_row = out + token * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        out_row[i] = row[i];
    }
}

// One block per token row: the sum of squares is reduced across the block in a fixed tree, then
// the row is scaled by the inverse root mean square and the gain.
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ gain,
                               __nv_bfloat16* __restrict__ out, int64_t hidden, float eps) {
    __shared__ float partial[kThreads];
    const __nv_bfloat16* row = x + static_cast<int64_t>(blockIdx.x) * hidden;
    __nv_bfloat16* out_row = out + static_cast<int64_t>(blockIdx.x) * hidden;

    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = to_f32(row[i]);
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        out_row[i] = to_bf16(to_f32(row[i]) * inv * to_f32(gain[i]));
    }
}

// Half-rotation over the first `rot_heads` heads of each fused qkv row, the q heads and then the
// k heads: pair p of a head, (x[p], x[p + half]), becomes (x[p] cos - x[p + half] sin,
// x[p + half] cos + x[p] sin), with cos and sin read from the tables at the token's position.
// One thread per (token, head, pair).
__global__ void rope_kernel(__nv_bfloat16* qkv, const int32_t* __restrict__ positions,
                            const float* __restrict__ cos_table,
                            const float* __restrict__ sin_table, int64_t n_tokens, int rot_heads,
                            int head_dim, int64_t row_width) {
    const int half = head_dim / 2;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tokens * rot_heads * half) {
        return;
    }
    const int pair = static_cast<int>(i % half);
    const int head = static_cast<int>((i / half) % rot_heads);
    const int64_t token = i / (static_cast<int64_t>(half) * rot_heads);
    __nv_bfloat16* base = qkv + token * row_width + static_cast<int64_t>(head) * head_dim;
    const int64_t table = static_cast<int64_t>(positions[token]) * half + pair;
    const float c = cos_table[table];
    const float s = sin_table[table];
    const float lo = to_f32(base[pair]);
    const float hi = to_f32(base[pair + half]);
    base[pair] = to_bf16(lo * c - hi * s);
    base[pair + half] = to_bf16(hi * c + lo * s);
}

__global__ void silu_mul_kernel(const __nv_bfloat16* __restrict__ gate,
                                const __nv_bfloat16* __restrict__ up,
                                __nv_bfloat16* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = to_f32(gate[i]);
    out[i] = to_bf16(g / (1.0f + expf(-g)) * to_f32(up[i]));
}

__global__ void add_kernel(const __nv_bfloat16* __restrict__ lhs,
                           const __nv_bfloat16* __restrict__ rhs,
                           __nv_bfloat16* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = to_bf16(to_f32(lhs[i]) + to_f32(rhs[i]));
}

}  // namespace

extern "C" cudaError_t decode_embedding_gather_bf16(const void* table, const void* token_ids,
                                                    void* out, int64_t hidden, int64_t n_tokens,
                                                    cudaStream_t stream) {
    if (n_tokens == 0 || hidden == 0) {
        return cudaSuccess;
    }
    embedding_gather_kernel<<<static_cast<unsigned int>(n_tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table), static_cast<const uint32_t*>(token_ids),
        static_cast<__nv_bfloat16*>(out), hidden);
    return cudaGetLastError();
}

extern "C" cudaError_t decode_rmsnorm_bf16(const void* x, const void* gain, void* out,
                                           int64_t hidden, int64_t n_tokens, float eps,
                                           cudaStream_t stream) {
    if (n_tokens == 0 || hidden == 0) {
        return cudaSuccess;
    }
    rmsnorm_kernel<<<static_cast<unsigned int>(n_tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(gain),
        static_cast<__nv_bfloat16*>(out), hidden, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t decode_rope_bf16(void* qkv, const void* positions, const void* cos_table,
                                        const void* sin_table, int64_t n_tokens,
                                        int64_t rot_heads, int64_t head_dim, int64_t row_width,
                                        cudaStream_t stream) {
    const int64_t threads = n_tokens * rot_heads * (head_dim / 2);
    if (threads == 0) {
        return cudaSuccess;
    }
    rope_kernel<<<blocks_covering(threads), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(qkv), static_cast<const int32_t*>(positions),
        static_cast<const float*>(cos_table), static_cast<const float*>(sin_table), n_tokens,
        static_cast<int>(rot_heads), static_cast<int>(head_dim), row_width);
    return cudaGetLastError();
}

extern "C" cudaError_t decode_silu_mul_bf16(const void* gate, const void* up, void* out,
                                            int64_t len, cudaStream_t stream) {
    if (len == 0) {
        return cudaSuccess;
    }
    silu_mul_kernel<<<blocks_covering(len), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_bfloat16*>(out), len);
    return cudaGetLastError();
}

extern "C" cudaError_t decode_add_bf16(const void* lhs, const void* rhs, void* out, int64_t len,
                                       cudaStream_t stream) {
    if (len == 0) {
        return cudaSuccess;
    }
    add_kernel<<<blocks_covering(len), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(lhs), static_cast<const __nv_bfloat16*>(rhs),
        static_cast<__nv_bfloat16*>(out), len);
    return cudaGetLastError();
}
