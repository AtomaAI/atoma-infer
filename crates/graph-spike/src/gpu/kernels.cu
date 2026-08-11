// Throwaway naive kernels for the #143 graph spike: correctness only, no performance intent.
// bf16 travels as raw u16 with explicit conversions so NVRTC needs no CUDA headers; every kernel
// reduces in f32 with a fixed order, so eager and replayed launches are bit-identical.

typedef unsigned short bf16_t;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float bf16_to_f32(bf16_t x) {
    return __uint_as_float(((u32)x) << 16);
}

// Round-to-nearest-even, matching hardware bf16 conversion for the finite values the spike
// generates (NaN payloads are not preserved; the spike never produces NaN on purpose).
__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    u32 u = __float_as_uint(f);
    u32 bias = 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)((u + bias) >> 16);
}

__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Fills a buffer with deterministic uniforms in [-scale, scale]; used once at setup for weights
// and the KV pool, keeping every activation unit-scale (bf16 overflows to inf near 3e4).
extern "C" __global__ void fill_random_bf16(bf16_t* out, u64 n, u64 seed, float scale) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        u64 r = splitmix64(seed ^ i);
        float unit = (float)(r >> 40) * (1.0f / 16777216.0f);
        out[i] = f32_to_bf16((2.0f * unit - 1.0f) * scale);
    }
}

extern "C" __global__ void embedding_gather(const bf16_t* table, const u32* token_ids,
                                            bf16_t* out, int hidden, int n_tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * hidden) return;
    int token = i / hidden;
    int col = i - token * hidden;
    out[i] = table[(u64)token_ids[token] * hidden + col];
}

// One block of 256 threads per token row.
extern "C" __global__ void rmsnorm_bf16(const bf16_t* x, const bf16_t* gamma, bf16_t* out,
                                        int hidden, float eps) {
    __shared__ float red[256];
    const bf16_t* row = x + (u64)blockIdx.x * hidden;
    bf16_t* out_row = out + (u64)blockIdx.x * hidden;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = bf16_to_f32(row[i]);
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) red[threadIdx.x] += red[threadIdx.x + offset];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hidden + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out_row[i] = f32_to_bf16(bf16_to_f32(row[i]) * inv * bf16_to_f32(gamma[i]));
    }
}

// NeoX-style half-rotation over the q and k segments of the fused qkv row; the k heads sit
// directly after the q heads, so one contiguous head index covers both. The token's position is
// seqlens_k[token] - 1. One thread per (token, head, dim pair).
extern "C" __global__ void rope_qk(bf16_t* qkv, const int* seqlens_k, int n_tokens,
                                   int num_q_heads, int num_kv_heads, int head_dim, float theta) {
    int half = head_dim / 2;
    int rot_heads = num_q_heads + num_kv_heads;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * rot_heads * half) return;
    int pair = i % half;
    int head = (i / half) % rot_heads;
    int token = i / (half * rot_heads);

    int row_width = (num_q_heads + 2 * num_kv_heads) * head_dim;
    bf16_t* base = qkv + (u64)token * row_width + head * head_dim;
    int pos = seqlens_k[token] - 1;
    float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
    float angle = (float)pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    float lo = bf16_to_f32(base[pair]);
    float hi = bf16_to_f32(base[pair + half]);
    base[pair] = f32_to_bf16(lo * c - hi * s);
    base[pair + half] = f32_to_bf16(lo * s + hi * c);
}

extern "C" __global__ void silu_mul(const bf16_t* gate, const bf16_t* up, bf16_t* out, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = bf16_to_f32(gate[i]);
    float u = bf16_to_f32(up[i]);
    out[i] = f32_to_bf16((g / (1.0f + expf(-g))) * u);
}

extern "C" __global__ void add_bf16(const bf16_t* a, const bf16_t* b, bf16_t* out, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

// One block of 256 threads per sequence; ties break toward the lowest index, which the
// lexicographic (value desc, index asc) reduction keeps associative and therefore deterministic.
extern "C" __global__ void argmax_bf16(const bf16_t* logits, int* out, int vocab) {
    __shared__ float best_val[256];
    __shared__ int best_idx[256];
    const bf16_t* row = logits + (u64)blockIdx.x * vocab;

    float val = -3.402823e38f;
    int idx = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = bf16_to_f32(row[i]);
        if (v > val || (v == val && i < idx)) {
            val = v;
            idx = i;
        }
    }
    best_val[threadIdx.x] = val;
    best_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other_val = best_val[threadIdx.x + offset];
            int other_idx = best_idx[threadIdx.x + offset];
            if (other_val > best_val[threadIdx.x] ||
                (other_val == best_val[threadIdx.x] && other_idx < best_idx[threadIdx.x])) {
                best_val[threadIdx.x] = other_val;
                best_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = best_idx[0];
}

extern "C" __global__ void bf16_to_f32_arr(const bf16_t* in, float* out, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf16_to_f32(in[i]);
}

extern "C" __global__ void f32_to_bf16_arr(const float* in, bf16_t* out, u64 n) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16(in[i]);
}
