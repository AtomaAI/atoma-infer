# Rung-0 baseline — protocol and procedure

**Status: measured on 2026-09-03**, on one H100 PCIe 80GB, both engines serving
`NousResearch/Meta-Llama-3.1-8B-Instruct`. One table per workload, both written by the harness:

| Workload | Table | Artifact |
|---|---|---|
| ShareGPT-derived, 500 requests at 8 req/s | [rung0-baseline-table.md](rung0-baseline-table.md) | [rung0-baseline-sharegpt.json](rung0-baseline-sharegpt.json) |
| Long-context, 8k input tokens, 150 requests at 0.5 req/s | [rung0-baseline-table-long-context.md](rung0-baseline-table-long-context.md) | [rung0-baseline-long-context.json](rung0-baseline-long-context.json) |

This file is hand-written and is not touched by any run.

Both engines were given the same KV budget and the same batch limits, so the tables compare engines
rather than memory splits: vLLM's defaults chose 463,040 KV tokens, 128 sequences per step and an
8192-token step budget on this host, and `atoma-infer` was configured to match. `atoma-infer` served
every run eagerly — replaying captured graphs is not yet wired into its serving path — so these are
the numbers for an engine that runs each step from scratch.

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
| **KV-leak probe** | `atoma-infer`'s `atoma_engine_available_blocks` gauge — the blocks the pool can still hand out, cached ones included — is sampled across each of its runs, starting from a baseline read before any load is offered. A pool whose ceiling falls, or that ends a run below that baseline, fails the run: `run` exits non-zero and `compare` writes no table. The companion `atoma_engine_free_blocks` gauge is the free list alone and is not what the probe judges: a healthy engine drains it to zero as retired requests park their blocks for prefix reuse. vLLM publishes no comparable gauge, so its runs are recorded as unwatched |

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

`run` on its own is the standing KV-leak regression guard. It writes its artifact and then exits
non-zero if the probe did not pass a run it watched, so a reintroduced leak fails the job rather
than appearing in a log line nobody reads. An engine configured without a `metrics_url` publishes
no gauge to sample; its runs are measured but not guarded, and `compare` refuses to publish a table
from them.

Requirements on the host: the rung's GPU, Docker with the NVIDIA runtime, the ShareGPT dump, and
the model's weights. Gated weights need a Hugging Face token (`HF_TOKEN`, forwarded to the
container by name); the rung-0 runs used an ungated mirror and needed none.

Set `engine.version` rather than leaving it to be read from git when the host's checkout is not the
tree that built the binary — a host synced by `rsync` or unpacked from an archive records whatever
commit its `.git` happens to hold.

## Choosing the offered rate

The protocol fixes the arrival process, not the rate. Both rates above were chosen by measuring
where the engines stop keeping up, because goodput at a fixed SLO only discriminates below that
point:

- **ShareGPT at 8 req/s.** `atoma-infer` serves 6.87 req/s of it and vLLM 7.71, so both are at
  their knee and the metric separates them.
- **Long-context at 0.5 req/s.** At 8k input tokens the two engines sustain 1.01 and 1.51 req/s.
  A first pass at 2 req/s put both far past that: every run completed, but first-token latency ran
  to 52 s for `atoma-infer` and 12 s for vLLM, and goodput fell to 0.013 against 0.005 — two
  numbers near zero, comparing nothing. 0.5 req/s sits under both knees.
