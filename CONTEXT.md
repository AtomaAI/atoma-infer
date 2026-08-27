# atoma-infer

Shared language for the CUDA-graph execution work: capturing decode steps as replayable CUDA
graphs over engine-owned memory. This file fixes the vocabulary used in tracked identifiers.

## Language

**Capture**:
Recording a step's device work into a CUDA graph instead of executing it.
_Avoid_: trace, record (as nouns)

**Replay**:
Launching a previously captured graph executable for one step.
_Avoid_: playback, re-run

**Allocation**:
The first of the executor's three phases — Allocation, Capture, Replay — each named for the
operation it permits. Allocation fixes every device address and binds every stream and
communicator. Nothing is captured in it.
_Avoid_: setup, init, startup

**Bucket**:
One captured batch size. A live batch is padded up to the nearest bucket.
_Avoid_: shape, size class

**Bucket ladder**:
The ordered list of buckets the engine captures. Always written qualified — "ladder" alone is
ambiguous with the rung ladder.
_Avoid_: capture sizes, batch-size list

**Live batch**:
The batch the scheduler reports for the current step, before padding.
_Avoid_: batch shape

**Graph key**:
The value that selects one captured graph for a padded batch.
_Avoid_: signature, shape id, cache key

**Rung**:
A numbered project milestone, used in `rungN-MM` commit-message prefixes. Not a bucket.
_Avoid_: ladder (unqualified), phase

**Arena**:
The engine-owned device allocation from which every captured step's activations are addressed.
_Avoid_: pool — the arena is neither a CUDA memory pool nor the KV block pool; its slots are
fixed, role-addressed extents, not interchangeable blocks

**Role**:
One tensor the step produces, declared with a per-token width and a lifetime.
_Avoid_: tensor name, buffer kind

**Lifetime**:
The half-open range of a layer's op order in which a role's slot holds live data.
_Avoid_: liveness, span

**Slot**:
The arena extent reserved for one tensor role in one layer.
_Avoid_: buffer (for arena extents), region

**Activation**:
An intermediate tensor produced and consumed within a single step.
_Avoid_: scratch, temporary

**Segment**:
One captured graph within a forward pass that is split around eager operations.
_Avoid_: piece, partial graph

**Break point**:
The place in a forward pass where one segment ends and the next begins.
_Avoid_: split, boundary, cut

**Bridge buffer**:
A fixed-address buffer through which an eager operation passes data between two segments.
_Avoid_: intermediate buffer, staging buffer

**Lease**:
The value that holds one pool block for exactly one owner. While it exists the block cannot be
evicted or handed out again; surrendering it is the block's only way back to the pool.
_Avoid_: ref count, handle

**Chain hash**:
The digest of one block-sized token run chained through its parent run's digest, so equal hashes
mean equal prefixes. The tier-agnostic identity of cached KV.
_Avoid_: content hash, prefix hash

**Dummy**:
A padding request filling one bucket slot. Each dummy owns its own KV block, leased from the
pool at startup for the process lifetime.
_Avoid_: filler, fake request

**Layer group**:
A set of layers with a common cache kind and geometry. A model's cache is declared group by
group, and a group can share another group's cache instead of writing its own.
_Avoid_: cache group, attention group

**Prefix index**:
The radix tree over chains of block hashes that answers longest-prefix match. It answers in
block hashes, never slot ids; which slot holds a hash's bytes is the pool's residence lookup.
_Avoid_: radix cache, prefix tree

**Tier**:
Where cached KV bytes live — device or host memory. A tier is where bytes live, never a
preemption mechanism.
_Avoid_: swap, cache level

**Residence**:
Which tier and slot currently hold a block hash's bytes. Identity is the chain hash and never
carries residence; residence is always a separate lookup.
_Avoid_: placement (for cached KV bytes; arena slot placement is unrelated and stays legal)

**Spike**:
A time-boxed experiment that answers a design question with measurements on real hardware.
_Avoid_: prototype, proof of concept

**Stint**:
A scheduled block of GPU-rig time during which spikes and verifications run.
_Avoid_: session, rental
