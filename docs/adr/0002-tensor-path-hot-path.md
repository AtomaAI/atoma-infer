# The decode hot path runs on atoma-runtime tensors, not candle

The decode hot path (model forward, attention, sampling inputs) is built on
`atoma-runtime::Tensor` — our own tensor type over cudarc-owned device buffers. candle is
cold-path only: weight loading and host-side glue, retiring family-by-family as model
definitions move to the tensor path.

Why not capture CUDA graphs over candle: production graph support here means decode-step graphs
per batch-size bucket *and* segmented capture (eager break points at attention/MoE boundaries).
Segmented capture needs deterministic allocator behavior between captured segments; over an
allocator we don't own, that is the hardest version of the problem — built on a substrate that
was already scheduled for retirement. Owning the buffers makes stable addresses, capture-pool
management, and replay patching straightforward instead of adversarial.

Decision record: AtomaAI/atoma-infer#168 (2026-08-05), amending the earlier
prototype-first plan (capture-over-candle, AtomaAI/atoma-infer#143).
