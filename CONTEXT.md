# atoma-infer

Shared language for the engine core and the CUDA-graph execution work: one thread scheduling
requests over engine-owned KV, and decode steps captured as replayable graphs. This file fixes
the vocabulary used in tracked identifiers.

## Language

**Capture**:
Recording a step's device work into a CUDA graph instead of executing it.
_Avoid_: trace, record (as nouns)

**Replay**:
Launching a previously captured graph executable for one step.
_Avoid_: playback, re-run

**Session phase**:
One of the executor session's three — Allocation, Capture, Replay — each named for the
operation it permits. Always written qualified: "phase" alone is ambiguous with request phase.
_Avoid_: phase (unqualified), stage, mode

**Allocation**:
The first session phase. Allocation fixes every device address and binds every stream and
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
The shape of one step's scheduled set — its token and request counts — before padding.
_Avoid_: batch shape

**Dispatch**:
Choosing, for a live batch, the captured graph to replay or the eager fallback. Engine-thread
work; the executor never re-derives it.
_Avoid_: admission (for this), graph lookup, routing

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
A padding request filling one bucket slot. Each dummy occupies a request slot and owns its own
KV block, both held from startup for the process lifetime; it never finishes and never enters
admission.
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

**Request phase**:
Where a request is in its lifecycle: Waiting, Running, Preempted, Finished, or Padding for a
dummy. Always written qualified; illegal transitions between phases are unrepresentable.
_Avoid_: phase (unqualified), status, state, stage

**Prefilling**:
A running request whose computed count is below its prompt length. A derived property of the
Running phase, never a phase of its own; a chunked prefill is a prefilling request that computes
fewer tokens this step than remain.
_Avoid_: prefill stage, prompt phase, is_prompt

**Decoding**:
A running request whose computed count has reached its prompt length. A derived property of the
Running phase, never a phase of its own.
_Avoid_: decode stage, generation phase

**Computed**:
The count of a request's tokens whose KV is resident. Resets to zero on preemption; whatever
the prefix index still holds is rediscovered when the request next enters Running.
_Avoid_: num computed tokens, processed, cached length

**Total**:
A request's prompt tokens plus every token it has generated so far.
_Avoid_: length, sequence length (for a request)

**Query length**:
The tokens an entry computes in one step.
_Avoid_: chunk size, token chunk size, num new tokens

**Context length**:
The tokens an entry already holds in KV before the step — its computed count when the step
command is built — in the sense attention kernels use. Never the model's maximum.
_Avoid_: context window, context size, num computed (in a step command)

**Sequence length**:
An entry's context length plus its query length: the tokens its KV holds after the step, in the
sense attention kernels use.
_Avoid_: seq len (in prose), total (in a step command)

**Max model length**:
The longest sequence the model serves. The only meaning "context window" carries here.
_Avoid_: context length (for the maximum), max context

**Intake**:
Taking a submitted request off ingress into Waiting: minting its request id, giving it a slab
slot and queueing it — or finishing it on the spot when its prompt can never run. The one
transition into Waiting, and never a word for admission.
_Avoid_: enqueue, accept, submit (for this transition), admission (for this)

**Admission**:
Moving a request from Waiting or Preempted into Running under the admission policy, which
examines a bounded window of candidates per pass and offers preempted requests first, last-in
first-out. The one transition into Running; nothing else is called admission.
_Avoid_: scheduling (for this transition), pickup, intake (which is the transition into
Waiting), resume

**Preemption**:
Releasing a running request's KV and returning it to run again from whatever the prefix index
still holds. Never a swap: nothing is written out to bring back.
_Avoid_: swap, evict (the prefix index's word for cached blocks), recompute (as the noun)

**Request**:
The client unit: one prompt, one set of sampling parameters, one egress sink. What admission
admits and preemption displaces, as a whole.
_Avoid_: sequence group, job, query (for a request)

**Sequence**:
One token stream inside a request, with its own block table, computed count, total and finish.
A request is born with one sequence and forks at its first sample when it asks for more.
_Avoid_: beam, candidate, hypothesis, stream

**Block table**:
The ordered block ids a sequence's KV occupies. Host-native: a step command is built from it
with no device read.
_Avoid_: block list, slot mapping (for the table)

**Step**:
One engine iteration: a scheduling pass, the step command it yields, and the step result that
comes back. Numbered by step id.
_Avoid_: iteration, tick, cycle

**Scheduled**:
The output of one scheduling pass: the entries this step runs, the slots it preempted, and the
block deltas the executor must see. Indices and counts, never copied request state.
_Avoid_: schedule, scheduler output, scheduler step, plan

**Entry**:
One row of a Scheduled: a sequence, its query length, and whether it samples. An entry samples
only when its query reaches the sequence's total; a non-final prefill chunk does not.
_Avoid_: scheduled tokens, scheduled sequence, batch item

**Uniform decode**:
A Scheduled whose every entry has query length one. The condition full-graph replay requires.
_Avoid_: decode-only, pure decode, all-decode

**Token budget**:
The per-step cap on query tokens summed over entries, plus a request cap equal to the largest
bucket. Spent by running requests first; the remainder is offered to admission.
_Avoid_: scheduling budget, batch budget, max batched tokens (as the name of the budget)

**Window**:
The count of admission candidates one pass examines. Scheduler-wide: every admission policy
sees the same window and differs only in how it orders it.
_Avoid_: lookahead, scan depth, sliding window (a KV geometry term)

**Engine thread**:
The one thread that owns all engine state — the request slab, block pool, prefix index,
scheduler and dispatch — and every transition on it. No lock sits on its step path.
_Avoid_: scheduler thread, engine core, driver

**Executor thread**:
The pinned per-rank thread that owns the device and runs the session. It acts on a step command
and re-derives nothing in it.
_Avoid_: worker, model thread, runner

**Ingress**:
The bounded channel that carries requests into the engine thread. A refused send is overload.
_Avoid_: request queue, inbound, input channel

**Overload**:
The condition an ingress refusal signals: the engine cannot take another request right now. The
API's 429.
_Avoid_: backpressure (as the condition), rejection, throttling

**Control**:
The bounded channel the engine thread drains before ingress on every pass. Carries drain,
shutdown and state queries — never cancels and never requests.
_Avoid_: command channel, admin channel

**Egress**:
The per-request channel that carries a request's output to its client. Its receiver dropping is
the one and only cancel; a failed send returns nothing to ignore.
_Avoid_: response channel, output stream, sink (for the channel)

**Ring**:
One of the two single-producer single-consumer rings between the engine thread and the
executor thread: step commands one way, step results the other. Rings are not channels.
_Avoid_: channel (for these), queue, pipe

**Step command**:
Everything the executor acts on for one step: entries with context and sequence lengths, block
tables, the padding dummies inserted, and the dispatch decision. Built with zero device reads.
_Avoid_: execute model request, model input, batch (for the command)

**Step result**:
What the executor returns for one step: each sampling entry's token and whatever the engine
needs to advance request state.
_Avoid_: model output, step output, sampler output

**Drain**:
The control message that stops admission and lets running requests finish. The engine is
drained when no step is in flight and nothing runs; that is the point at which control is
honoured.
_Avoid_: quiesce, pause, stop-the-world

**Heartbeat**:
The pass counter and timestamp the engine thread publishes every pass, so liveness is read from
the thread that could wedge rather than from the API in front of it.
_Avoid_: health check (for the signal), liveness probe, ping

**Spike**:
A time-boxed experiment that answers a design question with measurements on real hardware.
_Avoid_: prototype, proof of concept

**Stint**:
A scheduled block of GPU-rig time during which spikes and verifications run.
_Avoid_: session, rental
