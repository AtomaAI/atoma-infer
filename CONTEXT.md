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
_Avoid_: pool — a pool is a CUDA memory pool, which the arena deliberately is not

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

**Spike**:
A time-boxed experiment that answers a design question with measurements on real hardware.
_Avoid_: prototype, proof of concept

**Stint**:
A scheduled block of GPU-rig time during which spikes and verifications run.
_Avoid_: session, rental
