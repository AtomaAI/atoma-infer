# Bindings-first kernel policy

External CUDA kernels are never vendored as sources. They are consumed as prebuilt artifacts or
libraries — FlashInfer cubins loaded via cudarc for attention, cuBLASLt for GEMM, NCCL for
comms — with the host-side plan/run logic implemented in Rust. Small in-house kernels
(paged-KV cache ops, fused elementwise) stay in-repo as `.cu`: they have no upstream team to
diverge from.

The repo previously vendored flash-attention 2 forward kernels (66 translation units) plus the
cutlass submodule to build them. Maintaining patched copies of external kernels in parallel to
their upstream teams does not scale to the kernel surface this engine needs (FA3-class SM90,
MLA, SM100). The vendored FA2 kernels are frozen; the change that lands FlashInfer-bound SM90
attention deletes them and the cutlass submodule in the same PR, gated on golden-test parity.

Accepted cost: reimplementing each artifact's host-side launch/plan logic in Rust, re-verified
per artifact pin. The alternative — vendoring FA3 sources + cutlass 3.x + patches — was the
same maintenance burden this policy exists to end.

Decision record: AtomaAI/atoma-infer#169 (2026-08-05).
