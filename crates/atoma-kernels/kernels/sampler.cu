// The sampler's kernels: the gather that takes a row's token from what the device last sampled
// for its request slot, and the sample that draws each row's next token from its logits under
// the slot's record and leaves it on the device. Every launcher takes the caller's stream and
// returns the launch status. One block serves one row, and every reduction inside it runs in a
// fixed order, so a row's token depends on its logits, its record and nothing else: not on the
// batch it sits in, not on the slot it occupies, and not on which launch computed it.
//
// The sample kernel is the host reference (atoma-engine's sampling::reference) step for step:
// logits are ordered by a key monotone in their value with a not-a-number last; a greedy record
// takes the largest and draws nothing; a drawn record admits the top_k largest with every tie of
// the k-th, weights each admitted token by exp((logit - max) / temperature) truncated to 32-bit
// fixed point, keeps the heaviest tokens whose mass reaches top_p of the total with every tie of
// the last, and picks a token in index order by its share of what is left under one 64-bit
// Philox uniform. The selections are radix walks over histograms where the reference sorts.

#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
// The block reductions finish in warp zero, which reads one entry per warp with one lane each.
static_assert(kWarps == 32, "one block of kThreads threads must hold exactly one warp of warps");
constexpr int kBins = 256;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
// The weight of the largest logit: one, in 32-bit fixed point.
constexpr double kUnitWeight = 4294967296.0;

// One request slot's record, as atoma-engine's sampling::record lays it out.
struct SlotRecord {
    float temperature;
    float top_p;
    uint32_t top_k;
    uint32_t draws;
    uint64_t seed;
};
static_assert(sizeof(SlotRecord) == 24, "the record is 24 bytes, as the host declares it");

// Philox4x32-10, as Random123 defines it and the host computes it.
__device__ __forceinline__ void philox_round(uint32_t c[4], const uint32_t k[2]) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c[0]);
    const uint32_t lo0 = 0xD2511F53u * c[0];
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c[2]);
    const uint32_t lo1 = 0xCD9E8D57u * c[2];
    const uint32_t next0 = hi1 ^ c[1] ^ k[0];
    const uint32_t next2 = hi0 ^ c[3] ^ k[1];
    c[0] = next0;
    c[1] = lo1;
    c[2] = next2;
    c[3] = lo0;
}

// The 64-bit uniform for draw number `draw` under `seed`: the block's first two words for the
// counter (draw, 0, 0, 0) under the seed's two words, the first low.
__device__ uint64_t philox_draw(uint64_t seed, uint32_t draw) {
    uint32_t c[4] = {draw, 0u, 0u, 0u};
    uint32_t k[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    for (int round = 0; round < 10; ++round) {
        philox_round(c, k);
        if (round + 1 < 10) {
            k[0] += 0x9E3779B9u;
            k[1] += 0xBB67AE85u;
        }
    }
    return static_cast<uint64_t>(c[0]) | (static_cast<uint64_t>(c[1]) << 32);
}

// A logit's order among its row: monotone in the value, with a not-a-number below every number.
__device__ __forceinline__ uint32_t order_key(float logit) {
    if (isnan(logit)) {
        logit = -INFINITY;
    }
    const uint32_t bits = __float_as_uint(logit);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// An admitted token's weight in 32-bit fixed point; a not-a-number weighs nothing.
__device__ __forceinline__ unsigned long long weight_q(float logit, float max, float temperature) {
    if (isnan(logit)) {
        return 0ull;
    }
    const float w = expf((logit - max) / temperature);
    return static_cast<unsigned long long>(static_cast<double>(w) * kUnitWeight);
}

// The first largest key across the block: keys descending, then indices ascending.
__device__ void block_argmax(uint32_t& key, long long& index, uint32_t* keys, long long* indices) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint32_t other_key = __shfl_down_sync(kFullMask, key, offset);
        const long long other_index = __shfl_down_sync(kFullMask, index, offset);
        if (other_key > key || (other_key == key && other_index < index)) {
            key = other_key;
            index = other_index;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        keys[warp] = key;
        indices[warp] = index;
    }
    __syncthreads();
    if (warp == 0) {
        key = keys[lane];
        index = indices[lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            const uint32_t other_key = __shfl_down_sync(kFullMask, key, offset);
            const long long other_index = __shfl_down_sync(kFullMask, index, offset);
            if (other_key > key || (other_key == key && other_index < index)) {
                key = other_key;
                index = other_index;
            }
        }
        if (lane == 0) {
            keys[0] = key;
            indices[0] = index;
        }
    }
    __syncthreads();
    key = keys[0];
    index = indices[0];
    __syncthreads();
}

// The sum of every thread's value, in a fixed order.
__device__ unsigned long long block_sum(unsigned long long value, unsigned long long* totals) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        totals[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = totals[lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(kFullMask, value, offset);
        }
        if (lane == 0) {
            totals[0] = value;
        }
    }
    __syncthreads();
    value = totals[0];
    __syncthreads();
    return value;
}

// Every thread's exclusive prefix over the block's values in thread order, and the total.
__device__ unsigned long long block_exclusive_scan(unsigned long long value,
                                                   unsigned long long* totals,
                                                   unsigned long long& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned long long inclusive = value;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned long long other = __shfl_up_sync(kFullMask, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        totals[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        unsigned long long warp_inclusive = totals[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned long long other = __shfl_up_sync(kFullMask, warp_inclusive, offset);
            if (lane >= offset) {
                warp_inclusive += other;
            }
        }
        totals[lane] = warp_inclusive - totals[lane];
        if (lane == 31) {
            totals[kWarps] = warp_inclusive;
        }
    }
    __syncthreads();
    const unsigned long long exclusive = totals[warp] + inclusive - value;
    total = totals[kWarps];
    __syncthreads();
    return exclusive;
}

// The digit of the bin the walk from the top crosses `remaining` in; `remaining` is what is left
// to reach once the bins above are taken off. Thread zero walks and keeps `remaining` for the
// next digit; the other threads' copies are never read again, and every thread reads the digit
// from shared memory.
__device__ uint32_t crossing_bin(const unsigned long long* hist, unsigned long long& remaining,
                                 uint32_t* chosen) {
    if (threadIdx.x == 0) {
        uint32_t digit = 0;
        for (int bin = kBins - 1; bin >= 0; --bin) {
            if (remaining <= hist[bin]) {
                digit = static_cast<uint32_t>(bin);
                break;
            }
            remaining -= hist[bin];
        }
        *chosen = digit;
    }
    __syncthreads();
    const uint32_t digit = *chosen;
    __syncthreads();
    return digit;
}

// The k-th largest key of the row: after four digit walks the prefix is that key exactly. Every
// token whose key is at least it is admitted, which is the k largest and every tie of the k-th.
__device__ uint32_t kth_largest_key(const float* row, int64_t vocab, uint32_t k,
                                    unsigned long long* hist, uint32_t* chosen) {
    uint32_t prefix = 0;
    unsigned long long remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int bin = threadIdx.x; bin < kBins; bin += kThreads) {
            hist[bin] = 0;
        }
        __syncthreads();
        const uint32_t prefix_mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int64_t i = threadIdx.x; i < vocab; i += kThreads) {
            const uint32_t key = order_key(row[i]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 0xFFu], 1ull);
            }
        }
        __syncthreads();
        prefix |= crossing_bin(hist, remaining, chosen) << shift;
    }
    return prefix;
}

// The weight of the token the walk down the admitted weights crosses `target` at: after five
// digit walks over the 33-bit weights the prefix is that weight exactly. Every admitted token
// weighing at least it is kept, which is the heaviest reaching the target and every tie of the
// last.
__device__ unsigned long long crossing_weight(const float* row, int64_t vocab, uint32_t admit_key,
                                              float max, float temperature,
                                              unsigned long long target,
                                              unsigned long long* hist, uint32_t* chosen) {
    unsigned long long prefix = 0;
    unsigned long long remaining = target;
    for (int shift = 32; shift >= 0; shift -= 8) {
        for (int bin = threadIdx.x; bin < kBins; bin += kThreads) {
            hist[bin] = 0;
        }
        __syncthreads();
        const unsigned long long prefix_mask = shift == 32 ? 0ull : (~0ull << (shift + 8));
        for (int64_t i = threadIdx.x; i < vocab; i += kThreads) {
            const float logit = row[i];
            if (order_key(logit) < admit_key) {
                continue;
            }
            const unsigned long long q = weight_q(logit, max, temperature);
            if ((q & prefix_mask) == prefix) {
                atomicAdd(&hist[(q >> shift) & 0xFFull], q);
            }
        }
        __syncthreads();
        prefix |= static_cast<unsigned long long>(crossing_bin(hist, remaining, chosen)) << shift;
    }
    return prefix;
}

// The tokens thread `thread` owns for the pick: a contiguous run, so the block's threads in
// order cover the row in index order.
__device__ __forceinline__ void owned_range(int64_t vocab, int64_t& begin, int64_t& end) {
    const int64_t thread = threadIdx.x;
    begin = (thread * vocab) / kThreads;
    end = ((thread + 1) * vocab) / kThreads;
}

__global__ void __launch_bounds__(kThreads)
    sample_kernel(const float* __restrict__ logits, const int32_t* __restrict__ row_slots,
                  SlotRecord* records, uint32_t* sampled, uint32_t* __restrict__ out,
                  int64_t vocab) {
    __shared__ unsigned long long hist[kBins];
    __shared__ unsigned long long totals[kWarps + 1];
    __shared__ uint32_t keys[kWarps];
    __shared__ long long indices[kWarps];
    __shared__ uint32_t chosen;

    const int64_t row = blockIdx.x;
    const float* logits_row = logits + row * vocab;
    const int32_t slot = row_slots[row];
    const SlotRecord record = records[slot];

    // The first largest logit. A thread's run is in index order, so on a tie the earlier index
    // wins by staying; the sentinel index sits past the row and loses every tie.
    uint32_t best_key = order_key(-INFINITY);
    long long best_index = vocab;
    for (int64_t i = threadIdx.x; i < vocab; i += kThreads) {
        const uint32_t key = order_key(logits_row[i]);
        if (key > best_key || (key == best_key && i < best_index)) {
            best_key = key;
            best_index = i;
        }
    }
    block_argmax(best_key, best_index, keys, indices);
    const float max = logits_row[best_index];
    const uint32_t best = static_cast<uint32_t>(best_index);

    // A greedy record takes it; so does a row with no finite logit to draw from.
    if (record.temperature == 0.0f || !isfinite(max)) {
        if (threadIdx.x == 0) {
            out[row] = best;
            sampled[slot] = best;
        }
        return;
    }

    // The top_k largest, ties included.
    const uint32_t vocab_u32 = vocab > 0xFFFFFFFFll ? 0xFFFFFFFFu : static_cast<uint32_t>(vocab);
    const bool filter_k = record.top_k != 0 && record.top_k < vocab_u32;
    const uint32_t admit_key =
        filter_k ? kth_largest_key(logits_row, vocab, record.top_k, hist, &chosen) : 0u;

    // The mass of what is admitted, and the heaviest of it reaching top_p.
    unsigned long long local = 0;
    for (int64_t i = threadIdx.x; i < vocab; i += kThreads) {
        const float logit = logits_row[i];
        if (order_key(logit) >= admit_key) {
            local += weight_q(logit, max, record.temperature);
        }
    }
    const unsigned long long mass = block_sum(local, totals);
    unsigned long long keep_at_least = 1;
    if (record.top_p < 1.0f) {
        const double target = ceil(static_cast<double>(record.top_p) * static_cast<double>(mass));
        unsigned long long target_mass = static_cast<unsigned long long>(target);
        if (target_mass < 1) {
            target_mass = 1;
        }
        keep_at_least = crossing_weight(logits_row, vocab, admit_key, max, record.temperature,
                                        target_mass, hist, &chosen);
    }

    // The pick: each thread's run of tokens, its mass, and where the uniform falls.
    int64_t begin;
    int64_t end;
    owned_range(vocab, begin, end);
    local = 0;
    for (int64_t i = begin; i < end; ++i) {
        const float logit = logits_row[i];
        if (order_key(logit) >= admit_key) {
            const unsigned long long q = weight_q(logit, max, record.temperature);
            if (q >= keep_at_least) {
                local += q;
            }
        }
    }
    unsigned long long total;
    const unsigned long long exclusive = block_exclusive_scan(local, totals, total);
    const unsigned long long point = philox_draw(record.seed, record.draws) % total;
    if (exclusive <= point && point < exclusive + local) {
        unsigned long long cumulative = exclusive;
        for (int64_t i = begin; i < end; ++i) {
            const float logit = logits_row[i];
            if (order_key(logit) < admit_key) {
                continue;
            }
            const unsigned long long q = weight_q(logit, max, record.temperature);
            if (q < keep_at_least) {
                continue;
            }
            cumulative += q;
            if (cumulative > point) {
                const uint32_t token = static_cast<uint32_t>(i);
                out[row] = token;
                sampled[slot] = token;
                records[slot].draws = record.draws + 1;
                break;
            }
        }
    }
}

__global__ void gather_kernel(uint32_t* token_ids, const int32_t* __restrict__ gather_slots,
                              const uint32_t* __restrict__ sampled, int64_t n_rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int32_t slot = gather_slots[row];
    if (slot < 0) {
        return;
    }
    token_ids[row] = sampled[slot];
}

}  // namespace

extern "C" cudaError_t sampler_sample_f32(const void* logits, const void* row_slots, void* records,
                                          void* sampled, void* out, int64_t vocab, int64_t n_rows,
                                          cudaStream_t stream) {
    if (n_rows == 0 || vocab == 0) {
        return cudaSuccess;
    }
    sample_kernel<<<static_cast<unsigned int>(n_rows), kThreads, 0, stream>>>(
        static_cast<const float*>(logits), static_cast<const int32_t*>(row_slots),
        static_cast<SlotRecord*>(records), static_cast<uint32_t*>(sampled),
        static_cast<uint32_t*>(out), vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t sampler_gather_u32(void* token_ids, const void* gather_slots,
                                          const void* sampled, int64_t n_rows,
                                          cudaStream_t stream) {
    if (n_rows == 0) {
        return cudaSuccess;
    }
    constexpr int kGatherThreads = 256;
    const unsigned int blocks = static_cast<unsigned int>((n_rows + kGatherThreads - 1) / kGatherThreads);
    gather_kernel<<<blocks, kGatherThreads, 0, stream>>>(
        static_cast<uint32_t*>(token_ids), static_cast<const int32_t*>(gather_slots),
        static_cast<const uint32_t*>(sampled), n_rows);
    return cudaGetLastError();
}
