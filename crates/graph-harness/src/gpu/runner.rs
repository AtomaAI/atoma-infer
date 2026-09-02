//! The run driver: one capture session carrying every capture-matrix cell through its three
//! session phases — Allocation, Capture, Replay — then the weight reload carrying the first cell
//! through them again, with the comparison and soak measurements along the way.
//!
//! Ownership follows the session's rules: everything is allocated strictly before the session
//! moves into Capture, each recording hands the session the buffers its graph baked, and the
//! shared device state (weights, the KV pool, the arena) outlives the session's graph set. A cell
//! failure is classified and recorded, the recording is discarded by the session, and the run
//! continues — failures are spec findings, not aborts.
//!
//! No raw stream handle appears in this module: the capture stream reaches kernels only inside
//! the descriptor seam ([`crate::gpu::descriptor`]) and cuBLAS only through the bind seam in
//! Allocation, and the setup stream's raw handle stays inside the allocation helpers
//! ([`crate::gpu::alloc`]).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use atoma_runtime::arena::{BucketIdx, CaptureArena};
#[cfg(feature = "nccl")]
use atoma_runtime::communicator::Communicator;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::graph_entry::GraphEntry;
use atoma_runtime::session::{Allocation, Capture, GraphIdx, Replay};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaSlice;
#[cfg(feature = "nccl")]
use cudarc::nccl::Id;

use crate::compare::first_bf16_divergence;
use crate::dims::ModelDims;
use crate::gpu::alloc::{self, Allocator, CellStatics, KvPool, Weights};
use crate::gpu::blas::StepBlas;
#[cfg(feature = "nccl")]
use crate::gpu::descriptor::AllReduce;
use crate::gpu::descriptor::StepDescriptor;
use crate::gpu::kernels::StepKernels;
use crate::gpu::observe;
use crate::gpu::step::{StaticPtrs, StepContext, StepPtrs};
use crate::layout::{build_arena, StaticSizes};
#[cfg(feature = "nccl")]
use crate::matrix::StepContents;
use crate::matrix::{capture_matrix, CaptureCell};
use crate::report::{render_markdown, CellReport, DivergenceReport, Stats};
use crate::splits;
use crate::variation::{PlanConfig, StepInputs, VariationPlan};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

/// Run parameters, straight from the CLI.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub device_ordinal: usize,
    pub layers: usize,
    pub buckets: Vec<usize>,
    pub identity_steps: usize,
    pub soak_replays: usize,
    pub page_block: usize,
    pub max_seqlen: usize,
    pub start_seqlen: usize,
    pub seed: u64,
    pub include_all_reduce: bool,
    pub out_dir: PathBuf,
}

/// The run's derived shape: model dims, the cell matrix, the bucket ladder and the arena.
struct RunPlan {
    dims: ModelDims,
    cells: Vec<CaptureCell>,
    ladder: Vec<usize>,
    arena: CaptureArena,
    max_blocks_per_seq: usize,
    total_blocks: usize,
}

impl RunPlan {
    fn new(cfg: &RunConfig) -> Self {
        let dims = ModelDims::llama_8b_shaped(cfg.layers);
        let cells = capture_matrix(&cfg.buckets, cfg.include_all_reduce);
        let mut ladder = cfg.buckets.clone();
        ladder.sort_unstable_by(|a, b| b.cmp(a));
        ladder.dedup();
        let arena = build_arena(&dims, &ladder);
        let max_blocks_per_seq = cfg.max_seqlen / cfg.page_block;
        let total_blocks = ladder.first().copied().unwrap_or(1) * max_blocks_per_seq;
        Self {
            dims,
            cells,
            ladder,
            arena,
            max_blocks_per_seq,
            total_blocks,
        }
    }

    /// The bucket ladder position of `cell`'s batch size.
    fn bucket(&self, cell: &CaptureCell) -> Result<BucketIdx> {
        self.ladder
            .iter()
            .position(|&b| b == cell.batch_size)
            .map(BucketIdx)
            .ok_or_else(|| {
                anyhow!(
                    "cell {} is not on the bucket ladder {:?}",
                    cell.label(),
                    self.ladder
                )
            })
    }
}

/// What every phase of the run reads and none of them changes: the configuration, the plan, the
/// compiled kernels, the bound cuBLAS handle and the device's SM count.
struct RunContext<'a> {
    cfg: &'a RunConfig,
    plan: &'a RunPlan,
    kernels: &'a StepKernels,
    blas: &'a StepBlas,
    sm_count: usize,
}

/// The device state every cell's graph bakes and the session's graph set must not outlive: the
/// weights, the KV pool and the arena.
struct SharedState {
    weights: Weights,
    kv: KvPool,
    arena: CudaSlice<u8>,
}

impl SharedState {
    fn allocate(allocator: &Allocator<'_>, run: &RunContext<'_>, seed: u64) -> Result<Self> {
        let plan = run.plan;
        let weights = allocator
            .weights(&plan.dims, seed)
            .context("allocating and filling weights")?;
        let kv = allocator
            .kv_pool(&plan.dims, plan.total_blocks, run.cfg.page_block, seed)
            .context("allocating and filling the KV pool")?;
        let arena = allocator.bytes(plan.arena.total_size())?;
        Ok(Self { weights, kv, arena })
    }

    /// The live address of every shared buffer a graph bakes.
    fn addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.weights
            .addresses()
            .chain(self.kv.addresses())
            .chain(std::iter::once(alloc::addr(&self.arena)))
    }
}

/// One cell across the session phases: its plan, its address table, and what each phase found.
struct PreparedCell {
    cell: CaptureCell,
    bucket: BucketIdx,
    num_splits: u32,
    sizes: StaticSizes,
    ptrs: StepPtrs,
    plan: VariationPlan,
    /// The f32 buffer this cell's collective reduces in; `None` for cells without one.
    #[cfg(feature = "nccl")]
    mirror: Option<CudaSlice<f32>>,
    /// The recorded graph, once Capture succeeds.
    graph: Option<GraphIdx>,
    report: CellReport,
}

impl PreparedCell {
    /// The upload descriptor for one step's `inputs`.
    fn upload<'a>(&'a self, inputs: &'a StepInputs) -> StepDescriptor<'a> {
        StepDescriptor::Upload {
            staging: &self.ptrs.statics.staging,
            inputs,
        }
    }

    /// The decode descriptor at this cell's exact shape, with the cell's all-reduce installed
    /// when `comm` is the communicator its collective runs on.
    #[cfg(feature = "nccl")]
    fn decode<'a>(
        &'a mut self,
        ctx: &'a StepContext<'a>,
        comm: Option<&'a Communicator>,
    ) -> StepDescriptor<'a> {
        let all_reduce = comm
            .zip(self.mirror.as_mut())
            .map(|(comm, mirror)| AllReduce { comm, mirror });
        StepDescriptor::Decode {
            ctx,
            ptrs: &self.ptrs,
            sizes: &self.sizes,
            all_reduce,
        }
    }

    /// The decode descriptor at this cell's exact shape.
    #[cfg(not(feature = "nccl"))]
    fn decode<'a>(&'a self, ctx: &'a StepContext<'a>) -> StepDescriptor<'a> {
        StepDescriptor::Decode {
            ctx,
            ptrs: &self.ptrs,
            sizes: &self.sizes,
        }
    }

    /// Every device address this cell's graph baked: the entry's own buffers, the shared state
    /// and, for an all-reduce cell, its mirror.
    fn baked_addresses(&self, shared: &SharedState, entry: &GraphEntry) -> Vec<u64> {
        #[cfg(feature = "nccl")]
        let mirror = self.mirror.as_ref().map(alloc::addr);
        #[cfg(not(feature = "nccl"))]
        let mirror = None;
        entry
            .inputs()
            .iter()
            .chain(entry.outputs())
            .chain(entry.workspaces())
            .map(alloc::addr)
            .chain(shared.addresses())
            .chain(mirror)
            .collect()
    }
}

/// Per-cell device buffers allocated before the session moves into Capture; the recording takes
/// them over.
struct CellBuffers {
    statics: CellStatics,
    #[cfg(feature = "nccl")]
    comm: Option<Communicator>,
}

/// Host copies of one step's outputs.
struct StepOutputs {
    logits: Vec<u8>,
    argmax: Vec<u8>,
}

impl StepOutputs {
    fn zeroed(sizes: &StaticSizes) -> Self {
        Self {
            logits: vec![0u8; sizes.logits],
            argmax: vec![0u8; sizes.argmax],
        }
    }

    /// Reads the outputs back; call after the stream that wrote them has been synchronized.
    fn read(&mut self, statics: &StaticPtrs) -> Result<()> {
        observe::read_back(statics.logits, &mut self.logits)?;
        observe::read_back(statics.argmax, &mut self.argmax)
    }
}

/// Identity-loop timings in microseconds.
#[derive(Default)]
struct Timings {
    replay_enqueue: Vec<f64>,
    replay_step: Vec<f64>,
    eager_enqueue: Vec<f64>,
    eager_step: Vec<f64>,
}

impl Timings {
    fn record(self, report: &mut CellReport) {
        report.replay_enqueue = Stats::from_micros(self.replay_enqueue);
        report.replay_step = Stats::from_micros(self.replay_step);
        report.eager_enqueue = Stats::from_micros(self.eager_enqueue);
        report.eager_step = Stats::from_micros(self.eager_step);
    }
}

fn micros_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e6
}

/// Rejects configurations the build or the kernels cannot honour, and creates the output dir.
fn validate(cfg: &RunConfig) -> Result<()> {
    if cfg.include_all_reduce && !cfg!(feature = "nccl") {
        bail!("all-reduce cells need the nccl feature: rebuild with --features cuda,nccl");
    }
    if !cfg.max_seqlen.is_multiple_of(cfg.page_block) {
        bail!(
            "max_seqlen {} must be a multiple of page_block {}",
            cfg.max_seqlen,
            cfg.page_block
        );
    }
    fs::create_dir_all(&cfg.out_dir)
        .with_context(|| format!("creating out dir {}", cfg.out_dir.display()))
}

/// Runs the full capture matrix and the weight reload, and writes `findings.md`,
/// `measurements.json` and per-cell graph topology dumps into `cfg.out_dir`.
pub fn run(cfg: RunConfig) -> Result<Vec<CellReport>> {
    validate(&cfg)?;
    let plan = RunPlan::new(&cfg);

    // Allocation. The context comes first — it disables cudarc's event tracking before anything
    // is allocated — then the session, then every handle, buffer and communicator.
    let ctx = RuntimeContext::new(cfg.device_ordinal)?;
    if ctx.cuda().is_event_tracking() {
        bail!(
            "cudarc event tracking is still enabled after RuntimeContext::new; a wait on a \
             pre-capture event would invalidate every capture"
        );
    }
    let allocation = Allocation::new(&ctx)?;
    let sm_count = ctx
        .cuda()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| anyhow!("querying SM count: {:?}", e.0))? as usize;
    let setup_stream = ctx.cuda().default_stream();
    let kernels = StepKernels::compile_and_load(&ctx).context("compiling step kernels")?;
    let blas =
        StepBlas::new(allocation.stream()).context("creating and binding the cuBLAS handle")?;
    let run = RunContext {
        cfg: &cfg,
        plan: &plan,
        kernels: &kernels,
        blas: &blas,
        sm_count,
    };
    let allocator = Allocator::new(&kernels, &setup_stream);

    // Shared device state; declared before the session's graph set exists so it outlives it.
    let shared = SharedState::allocate(&allocator, &run, cfg.seed)?;
    let (mut prepared, buffers) = prepare_cells(&run, &allocator, &allocation, &shared)?;
    allocator.synchronize()?;

    // Capture: per cell, warmup at the exact shape, then the recording. No address moves and no
    // handle binds past this point.
    let mut capture = allocation.into_capture();
    capture_cells(&run, &mut capture, &mut prepared, buffers);

    // Replay: the identity loop and the soak for every cell that recorded.
    let replay = capture.into_replay();
    replay_cells(&run, &replay, &shared, &mut prepared);

    // Weight reload: the one transition back to Allocation. The graph set dies with the Replay
    // phase, so the old weights can go before the new ones are allocated.
    let allocation = replay.reload_weights();
    drop(shared);
    let reloaded = reload_cycle(&run, &allocator, allocation)?;

    let mut reports: Vec<CellReport> = prepared.into_iter().map(|cell| cell.report).collect();
    reports.push(reloaded);
    write_findings(&run, &reports)?;
    Ok(reports)
}

/// Fixes every matrix cell's plan, static buffers and address table, and creates the
/// communicators of the cells that reduce — allocation only, no capture.
fn prepare_cells(
    run: &RunContext<'_>,
    allocator: &Allocator<'_>,
    allocation: &Allocation,
    shared: &SharedState,
) -> Result<(Vec<PreparedCell>, Vec<CellBuffers>)> {
    let mut prepared = Vec::new();
    let mut buffers = Vec::new();
    for (index, cell) in run.plan.cells.iter().enumerate() {
        let (cell_prepared, statics) = prepare_cell(run, allocator, shared, *cell, index)?;
        prepared.push(cell_prepared);
        buffers.push(cell_buffers(allocation, cell, statics)?);
    }
    Ok((prepared, buffers))
}

/// Fixes one cell's plan, static buffers and address table. `cell_index` salts the cell's input
/// variation so no two cells replay the same token sequence.
fn prepare_cell(
    run: &RunContext<'_>,
    allocator: &Allocator<'_>,
    shared: &SharedState,
    cell: CaptureCell,
    cell_index: usize,
) -> Result<(PreparedCell, CellStatics)> {
    let cfg = run.cfg;
    let plan = run.plan;
    let dims = &plan.dims;
    let bucket = plan.bucket(&cell)?;
    let num_splits = splits::num_splits(
        cell.batch_size,
        dims.num_q_heads,
        dims.head_dim,
        cfg.max_seqlen,
        run.sm_count,
    );
    let sizes = StaticSizes::for_bucket(dims, cell.batch_size, plan.max_blocks_per_seq, num_splits);
    let variation = VariationPlan::new(PlanConfig {
        batch_size: cell.batch_size,
        page_block: cfg.page_block,
        max_blocks_per_seq: plan.max_blocks_per_seq,
        total_blocks: plan.total_blocks,
        start_seqlen: cfg.start_seqlen,
        planned_steps: 1 + cfg.identity_steps + cfg.soak_replays,
        vocab: dims.vocab,
        seed: cfg.seed ^ ((cell_index as u64) << 32),
    })?;
    let statics = allocator.cell_statics(&sizes)?;
    #[cfg(feature = "nccl")]
    let mirror = match cell.contents {
        StepContents::DecodeAllReduce => Some(allocator.f32s(cell.batch_size * dims.hidden)?),
        StepContents::Decode => None,
    };
    let ptrs = StepPtrs {
        weights: shared.weights.ptrs.clone(),
        kv: shared.kv.layers.clone(),
        arena_base: alloc::addr(&shared.arena),
        statics: statics.addresses(),
    };
    let prepared = PreparedCell {
        cell,
        bucket,
        num_splits,
        sizes,
        ptrs,
        plan: variation,
        #[cfg(feature = "nccl")]
        mirror,
        graph: None,
        report: CellReport::new(cell.label()),
    };
    Ok((prepared, statics))
}

/// The buffers one recording takes over, plus the communicator an all-reduce cell's collective
/// runs on — created here, in Allocation, because NCCL init allocates.
#[cfg(feature = "nccl")]
fn cell_buffers(
    allocation: &Allocation,
    cell: &CaptureCell,
    statics: CellStatics,
) -> Result<CellBuffers> {
    let comm = match cell.contents {
        StepContents::Decode => None,
        StepContents::DecodeAllReduce => {
            let id = Id::new().map_err(|e| anyhow!("ncclGetUniqueId: {:?}", e.0))?;
            Some(allocation.stream().nccl_comm(0, 1, id)?)
        }
    };
    Ok(CellBuffers { statics, comm })
}

/// The buffers one recording takes over. Without the nccl feature no cell has a communicator
/// (`validate` rejects all-reduce cells), so the buffers are the statics alone.
#[cfg(not(feature = "nccl"))]
fn cell_buffers(
    _allocation: &Allocation,
    _cell: &CaptureCell,
    statics: CellStatics,
) -> Result<CellBuffers> {
    Ok(CellBuffers { statics })
}

/// The Capture-phase pass: every cell warmed and recorded, failures noted per report.
fn capture_cells(
    run: &RunContext<'_>,
    capture: &mut Capture,
    prepared: &mut [PreparedCell],
    buffers: Vec<CellBuffers>,
) {
    for (cell, cell_buffers) in prepared.iter_mut().zip(buffers) {
        if let Err(err) = capture_cell(run, capture, cell, cell_buffers) {
            cell.report.failure = Some(format!("{err:#}"));
        }
    }
}

/// One warmup pass at the cell's exact shape, then the recording. The session's entry takes
/// ownership of every baked buffer, and the communicator is attached before the next recording.
fn capture_cell(
    run: &RunContext<'_>,
    capture: &mut Capture,
    prepared: &mut PreparedCell,
    buffers: CellBuffers,
) -> Result<()> {
    let CellBuffers {
        statics,
        #[cfg(feature = "nccl")]
        comm,
    } = buffers;
    let step_ctx = step_context(run, prepared);
    let warmup = prepared.plan.next_step();
    capture
        .warm_up(&mut prepared.upload(&warmup))
        .context("warmup upload")?;
    #[cfg(feature = "nccl")]
    let mut decode = prepared.decode(&step_ctx, comm.as_ref());
    #[cfg(not(feature = "nccl"))]
    let mut decode = prepared.decode(&step_ctx);
    capture.warm_up(&mut decode).context("warmup step")?;

    let free_before = observe::free_memory()?;
    let capture_started = Instant::now();
    let idx = capture
        .record(&mut decode, statics.into_baked())
        .context("recording the step")?;
    let capture_ms = micros_since(capture_started) / 1e3;
    #[cfg(feature = "nccl")]
    if let Some(comm) = comm {
        capture.attach_comm(idx, comm);
    }
    let free_after = observe::free_memory()?;

    let graph = capture.entry(idx).graph();
    let dot_path = run
        .cfg
        .out_dir
        .join(format!("{}.dot", prepared.report.label));
    graph.write_debug_dot(&dot_path, 0)?;
    let report = &mut prepared.report;
    report.capture_ms = Some(capture_ms);
    report.graph_dedicated_bytes = Some(free_before - free_after);
    report.graph_node_count = Some(graph.node_count()?);
    prepared.graph = Some(idx);
    Ok(())
}

/// The Replay-phase pass: the identity loop and the soak for every cell that recorded.
fn replay_cells(
    run: &RunContext<'_>,
    replay: &Replay,
    shared: &SharedState,
    prepared: &mut [PreparedCell],
) {
    for cell in prepared.iter_mut() {
        let Some(idx) = cell.graph else { continue };
        let replayed = identity_loop(run, replay, shared, cell, idx)
            .and_then(|()| soak(run, replay, shared, cell, idx));
        if let Err(err) = replayed {
            cell.report.failure = Some(format!("{err:#}"));
        }
    }
}

/// The identity loop: each step replays the graph and runs the same step eagerly, and the two
/// must agree byte for byte in the logits and the argmax. Baked addresses are re-checked after
/// every replay.
fn identity_loop(
    run: &RunContext<'_>,
    replay: &Replay,
    shared: &SharedState,
    prepared: &mut PreparedCell,
    idx: GraphIdx,
) -> Result<()> {
    let step_ctx = step_context(run, prepared);
    let baked = prepared.baked_addresses(shared, replay.entry(idx));
    let mut timings = Timings::default();
    let mut replayed = StepOutputs::zeroed(&prepared.sizes);
    let mut eager = StepOutputs::zeroed(&prepared.sizes);

    for step_index in 0..run.cfg.identity_steps {
        let inputs = prepared.plan.next_step();
        replay.run(&mut prepared.upload(&inputs))?;

        let start = Instant::now();
        replay.replay(idx)?;
        timings.replay_enqueue.push(micros_since(start));
        replay.synchronize()?;
        timings.replay_step.push(micros_since(start));
        if prepared.baked_addresses(shared, replay.entry(idx)) != baked {
            bail!("baked device pointers moved between capture and replay {step_index}");
        }
        replayed.read(&prepared.ptrs.statics)?;

        let start = Instant::now();
        #[cfg(feature = "nccl")]
        let mut decode = prepared.decode(&step_ctx, replay.entry(idx).comm());
        #[cfg(not(feature = "nccl"))]
        let mut decode = prepared.decode(&step_ctx);
        replay
            .run(&mut decode)
            .with_context(|| format!("eager step {step_index}"))?;
        timings.eager_enqueue.push(micros_since(start));
        replay.synchronize()?;
        timings.eager_step.push(micros_since(start));
        eager.read(&prepared.ptrs.statics)?;

        if let Some(divergence) = first_bf16_divergence(&replayed.logits, &eager.logits) {
            prepared.report.divergence = Some(DivergenceReport {
                step: step_index,
                divergence,
            });
            bail!("bit-identity failed in the logits at step {step_index}");
        }
        if replayed.argmax != eager.argmax {
            bail!("bit-identity failed in the argmax outputs at step {step_index}");
        }
        prepared.report.identity_steps += 1;
    }
    timings.record(&mut prepared.report);
    Ok(())
}

/// The replay-only soak: every baked address must hold after each replay, and free memory must
/// be flat after the first warm replay — a nonzero delta fails the cell.
fn soak(
    run: &RunContext<'_>,
    replay: &Replay,
    shared: &SharedState,
    prepared: &mut PreparedCell,
    idx: GraphIdx,
) -> Result<()> {
    let baked = prepared.baked_addresses(shared, replay.entry(idx));
    let mut free_after_warm = None;
    for soak_index in 0..run.cfg.soak_replays {
        let inputs = prepared.plan.next_step();
        replay.run(&mut prepared.upload(&inputs))?;
        replay
            .replay(idx)
            .with_context(|| format!("soak replay {soak_index}"))?;
        if soak_index == 0 || soak_index % 64 == 63 {
            replay.synchronize()?;
        }
        if prepared.baked_addresses(shared, replay.entry(idx)) != baked {
            bail!("baked device pointers moved during soak replay {soak_index}");
        }
        if soak_index == 0 {
            free_after_warm = Some(observe::free_memory()?);
        }
        prepared.report.soak_replays += 1;
    }
    replay.synchronize()?;
    let Some(free_after_warm) = free_after_warm else {
        return Ok(());
    };
    let delta = free_after_warm - observe::free_memory()?;
    prepared.report.soak_mem_delta_bytes = Some(delta);
    if delta != 0 {
        bail!(
            "free memory moved by {delta} bytes across {} soak replays",
            prepared.report.soak_replays
        );
    }
    Ok(())
}

/// The weight reload taken through all three session phases on the first matrix cell: new
/// weights and caches land at new addresses, and the cell is prepared, captured and replayed
/// against eager again. Its report is labelled as the reload.
fn reload_cycle(
    run: &RunContext<'_>,
    allocator: &Allocator<'_>,
    allocation: Allocation,
) -> Result<CellReport> {
    let Some(&cell) = run.plan.cells.first() else {
        bail!("the capture matrix is empty; nothing to reload");
    };
    let shared = SharedState::allocate(allocator, run, run.cfg.seed.wrapping_add(1))?;
    let (mut prepared, statics) =
        prepare_cell(run, allocator, &shared, cell, run.plan.cells.len())?;
    prepared.report = CellReport::new(format!("{}+reload", cell.label()));
    let buffers = cell_buffers(&allocation, &cell, statics)?;
    allocator.synchronize()?;

    let mut capture = allocation.into_capture();
    capture_cells(
        run,
        &mut capture,
        std::slice::from_mut(&mut prepared),
        vec![buffers],
    );
    let replay = capture.into_replay();
    replay_cells(run, &replay, &shared, std::slice::from_mut(&mut prepared));

    // The graph set dies before the state whose addresses it baked.
    drop(replay);
    Ok(prepared.report)
}

/// One cell's step configuration; copies the cell's scalars, so it borrows only from `run`.
fn step_context<'a>(run: &RunContext<'a>, prepared: &PreparedCell) -> StepContext<'a> {
    StepContext {
        kernels: run.kernels,
        blas: run.blas,
        dims: &run.plan.dims,
        arena: &run.plan.arena,
        bucket: prepared.bucket,
        batch_size: prepared.cell.batch_size,
        page_block: run.cfg.page_block,
        max_blocks_per_seq: run.plan.max_blocks_per_seq,
        num_splits: prepared.num_splits,
        rope_theta: ROPE_THETA,
        rms_eps: RMS_EPS,
    }
}

/// Renders findings.md and measurements.json into the run's output directory.
fn write_findings(run: &RunContext<'_>, reports: &[CellReport]) -> Result<()> {
    let cfg = run.cfg;
    let header = vec![
        format!(
            "device ordinal {}, {} SMs, event tracking disabled at context creation (checked)",
            cfg.device_ordinal, run.sm_count
        ),
        format!(
            "{} layers, buckets {:?}, {} identity steps, {} soak replays",
            cfg.layers, run.plan.ladder, cfg.identity_steps, cfg.soak_replays
        ),
        format!(
            "page_block {}, max_seqlen {}, start_seqlen {}, seed {}",
            cfg.page_block, cfg.max_seqlen, cfg.start_seqlen, cfg.seed
        ),
    ];
    let markdown = render_markdown(&header, reports);
    fs::write(cfg.out_dir.join("findings.md"), markdown)?;
    let json = serde_json::to_string_pretty(reports)?;
    fs::write(cfg.out_dir.join("measurements.json"), json)?;
    Ok(())
}
