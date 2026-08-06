# Rung-0 baseline — protocol and procedure

**Status: not yet measured.** The harness that produces the table landed with PR-4
([#164](https://github.com/AtomaAI/atoma-infer/issues/164)); the numbers land when it is run on the
rung-0 H100 stint. The table itself lives in
[rung0-baseline-table.md](rung0-baseline-table.md), which the harness overwrites — this file is
hand-written and is not touched by any run.

The table is the rung-0 exit evidence for [the ladder](../plan/README.md#3-the-ladder):
*"atoma-vs-vLLM baseline table exists. No performance bar — the number is the deliverable."* There
is no bar to clear. The point is to have a measured starting line, taken under a protocol the later
rungs can be measured against.

## Protocol

Per [plan §8](../plan/README.md#8-benchmark--correctness-protocol):

| | |
|---|---|
| **Baseline** | vLLM `v0.26.0` — latest stable at rung-0 kickoff (released 2026-07-27), pinned as `vllm/vllm-openai:v0.26.0`. The harness refuses a moving tag |
| **Baseline config** | vLLM's documented recommended configuration: server defaults plus `--model`. Any deviation goes in `server_args` and is published with the results |
| **Headline metric** | Goodput at a fixed SLO — completed requests/s meeting TTFT ≤ 2000 ms and mean ITL ≤ 50 ms |
| **Latencies** | TTFT, ITL and end-to-end recorded as hdrhistogram distributions; p50/p90/p99 published, never means alone |
| **Load** | Open-loop Poisson arrivals at a target rate, not a fixed-concurrency loop; one seed fixes both the prompts and the arrival times |
| **Workloads** | ShareGPT-derived conversational trace, and a long-context workload at 8k input tokens. One artifact and one table per workload — they are not averaged together |
| **Runs** | ≥3 per engine per workload; the median run by goodput is reported, and every run is listed |
| **Reporting** | Absolute numbers alongside ratios; both engines' full configurations emitted with the results |
| **KV-leak probe** | `atoma-infer`'s free-block gauge is sampled across each of its runs. A pool whose ceiling falls, or that ends a run below where sampling started, fails the run and no table is written. vLLM publishes no such gauge, so its runs are recorded as unwatched |

## How it is produced

A comparison is three commands, because both engines want the whole GPU and cannot run at once.

```shell
cp crates/bench/bench.example.toml bench.toml   # fill in the host's paths and hardware string

# 1. atoma-infer serving:
cargo run --release -p atoma-bench -- run \
    --config bench.toml --results results/atoma-sharegpt.json

# 2. stop atoma-infer, then start and drive the pinned baseline:
cargo run --release -p atoma-bench -- baseline \
    --config bench.toml --results results/vllm-sharegpt.json

# 3. combine, check against the protocol, and render:
cargo run --release -p atoma-bench -- compare \
    --config bench.toml \
    --engine results/atoma-sharegpt.json \
    --baseline results/vllm-sharegpt.json \
    --results docs/benchmarks/rung0-baseline-sharegpt.json \
    --table docs/benchmarks/rung0-baseline-table.md
```

Then repeat all three with the long-context workload configured, writing to its own artifact and
table paths — a second run against the same paths would overwrite the first workload's numbers.

`compare` writes the table only if the runs meet the protocol: too few runs, or a run whose KV pool
did not hold, leaves the artifact behind for inspection and no table. `render` re-renders a table
from an artifact without re-running anything.

Requirements on the host: the rung's GPU, Docker with the NVIDIA runtime, the ShareGPT dump, and a
Hugging Face token (`HF_TOKEN`, forwarded to the container by name) for the model's weights.
