# atoma-infer

Rust-native LLM inference engine: CUDA kernels bound via cudarc, an OpenAI-compatible API, and a
scheduler/KV core designed for frontier MoE serving.

## Language

### Planning

**Rung**:
One level of the capability ladder (un-brick → engine core → performance core → MoE → MLA →
launch completion). Work lands on its rung; each rung exits on a measured gate.

**Launch gate**:
The measured bar for launch: goodput within 10% of the better of vLLM/SGLang on DeepSeek-class
serving, plus at least two outright headline wins.

**Spike**:
A short, time-boxed build that answers a design question with evidence. Distinct from a
prototype ticket only in that its outcome feeds a spec rather than deciding between substrates.

### Tensors and substrates

**Tensor path**:
The decode hot path built on `atoma-runtime::Tensor` — our tensor type over cudarc-owned device
buffers. All hot-path math lives here.
_Avoid_: "candle path" for anything performance-critical

**Cold path**:
The non-latency-critical work candle still owns: weight loading and host-side glue. Retires
family-by-family as model definitions move to the tensor path.

### CUDA graphs

**FullDecodeOnly step**:
A decode step for a batch in which every sequence generates exactly one token (or 1+k under
speculation) — the uniform shape that a full-graph capture covers.

**Uniform decode**:
The dispatch predicate for FullDecodeOnly: every request in the batch has the same query length
(1, or 1+k with speculation).

**Bucket ladder**:
The fixed list of batch sizes at which decode graphs are captured (e.g. 1/8/32/64).

**Pad-to-bucket**:
Padding a live batch up to the nearest captured bucket size and replaying that graph, rather
than falling back to eager execution.

**Replay**:
Launching a captured graph for a new step: persistent input buffers are rewritten in place,
then the graph executes with no per-kernel launch overhead.

**Replay soak**:
A sustained run of consecutive replays (~1000) asserting no graph invalidation and no memory
growth.

**Segmented capture**:
Capturing the static kernel segments between eager break points (attention/MoE boundaries) as
separate graphs, so mixed and prefill batches also benefit. The native equivalent of vLLM's
piecewise mode, without torch.compile.

**Capture-legal**:
Property of a kernel or op sequence that can run inside CUDA stream capture: no synchronous
allocs, no stream syncs, no legacy-stream use.

### Kernel policy

**Bindings-first**:
The rule that external kernels are consumed as prebuilt artifacts or libraries with Rust host
logic — never vendored as sources. See ADR 0001.

**Artifact pin**:
The pinned prebuilt kernel artifact (cubin/library version) a release binds against; host-side
launch logic is re-verified when the pin moves.

**Plan/run split**:
The graph-compatible attention contract: `plan` computes scheduling metadata on the host outside
capture; `run` executes inside capture reading only persistent buffers.

**In-house kernel**:
A `.cu` file we author and own in-repo (paged-KV cache ops, fused elementwise). The deliberate
exception to bindings-first: no upstream to track.
