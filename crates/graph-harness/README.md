# graph-harness

A disposable experiment answering one question before the real CUDA-graph implementation is
built: **can this workspace's exact kernel set be captured into a CUDA graph and replayed
against new inputs, cheaply and bit-exactly?** (#143; the design it de-risks was decided in
#168.)

Nothing in this crate ships. It exists to produce a findings note and a handful of
measurements, and it is deleted once #143 closes.

## What it does

It builds a fake-but-representative Llama-8B-shaped decode step — the same op classes the real
model launches, with random weights:

| Op | Implementation |
|---|---|
| qkv / o / gate / up / down / lm-head projections | cuBLAS (`cublasGemmEx`, bf16 in, f32 compute) |
| RMSNorm, RoPE, SwiGLU, residual adds, embedding gather, argmax | naive NVRTC-compiled kernels (`src/gpu/kernels.cu`, correctness only) |
| Paged decode attention | the vendored FlashAttention-2 `run_mha`, called through the same FFI the engine uses |
| KV-cache write | `reshape_and_cache_flash_cache`, same FFI |
| TP collective (optional) | ws=1 NCCL all-reduce per layer, inside the captured step |

For every cell of the capture matrix (batch sizes 1/8/32/64 × decode with and without the
in-graph all-reduce, largest bucket first), the harness:

1. runs the step eagerly once (warmup: cuBLAS workspaces and first-call setup land before
   capture),
2. records the whole step into a CUDA graph via `atoma-runtime` and instantiates it,
3. replays the graph against changing inputs — new token ids, advancing sequence lengths,
   block tables that change every step — and compares logits and argmax **byte-for-byte**
   against an eager run of the same ops on the same inputs (≥32 steps), asserting every baked
   device address unchanged at every replay,
4. soaks: ~1000 consecutive replays with a `cuMemGetInfo` delta that must be zero after the
   first warm replay,
5. measures what cannot be computed on paper: the driver-side memory one instantiated graph
   costs (the calibration constant for bucket-ladder memory budgeting), capture time, and host
   time per step for eager vs replay.

A failure is a deliverable, not an abort: it is classified (kernel capture-illegal / cudarc
behavior under capture / hidden FFI sync / NCCL) and recorded, the capture is drained, and the
run continues with the next cell.

## Building and running

Everything except the FlashAttention-2 FFI seam (`src/fa2.rs`) type-checks and unit-tests on a
machine with no GPU, no CUDA toolkit, and no driver:

```sh
cargo test -p graph-harness --features nccl
cargo run -p graph-harness -- plan --nccl        # prints the matrix and memory budget anywhere
```

The real run needs a CUDA rig (the `cuda` feature links the vendored FA2 kernels; `nccl` adds
the all-reduce cells):

```sh
cargo build -p graph-harness --features cuda,nccl --release
target/release/graph-harness run --nccl --out .scratch/graph-harness/run1
```

Outputs land in `--out`: `findings.md` (capture matrix table, timings, memory), a raw
`measurements.json`, and one Graphviz `.dot` topology dump per captured cell
(`cuGraphDebugDotPrint`).

Useful knobs: `--layers` (default 4), `--buckets`, `--steps` (bit-identity comparisons, default
32), `--soak` (default 1000), `--max-seqlen`, `--page-block`, `--seed`. Every run is
deterministic from the seed.

## Module map

| Module | Responsibility |
|---|---|
| `dims` | Llama-8B-shaped dimensions and derived sizes |
| `matrix` | The capture matrix: buckets × step contents, largest bucket first |
| `variation` | Deterministic per-step inputs: token ids, seq lens, block tables, slots |
| `layout` | Arena roles, static buffer sizes, device memory budget |
| `splits` | The FA2 split-KV heuristic, mirrored so accumulators allocate before capture |
| `compare` | Bit-identity comparison and bf16 divergence reporting |
| `report` | Findings note: matrix table, timings, memory measurements |
| `gpu` | Device runner: NVRTC kernels, cuBLAS, the step function, capture and replay |
| `fa2` | Paged decode attention + KV write over the FA2 FFI (`cuda` feature only) |
