//! The run driver: one capture session carrying every capture-matrix cell through its three
//! session phases — Allocation, Capture, Replay — and the comparison and soak measurements.
//!
//! Ownership follows the session's rules: everything is allocated strictly before the session
//! seals into Capture, each recording hands the session the buffers its graph baked, and the
//! shared device state (weights, KV caches, arena, the all-reduce mirror) outlives the session's
//! graph set. A cell failure is classified and recorded, the recording is discarded by the
//! session, and the run continues — failures are spec findings, not aborts.
//!
//! No raw stream handle appears in this module: the capture stream reaches kernels only inside
//! the descriptor seam ([`crate::gpu::descriptor`]), and the setup stream's raw handle stays
//! inside the allocation helpers ([`crate::gpu::alloc`]).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use atoma_runtime::arena::{BucketIdx, CaptureArena};
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::graph_entry::GraphEntry;
use atoma_runtime::session::{Allocation, BakedBuffers, Capture, GraphIdx, Replay};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm, Id};

use crate::compare::first_bf16_divergence;
use crate::dims::ModelDims;
use crate::gpu::alloc::{self, WeightPtrs};
use crate::gpu::blas::StepBlas;
use crate::gpu::descriptor::{AllReduce, StepWork};
use crate::gpu::kernels::StepKernels;
use crate::gpu::step::{LayerPtrs, StagingPtrs, StepContext, StepPtrs};
use crate::layout::{build_arena, StaticSizes};
#[cfg(feature = "nccl")]
use crate::matrix::StepContents;
use crate::matrix::{capture_matrix, CaptureCell};
use crate::report::{render_markdown, CellReport, DivergenceReport, Stats};
use crate::splits;
use crate::variation::{PlanConfig, VariationPlan};

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

/// Immutable per-run dependencies every phase reads.
struct Deps<'a> {
    cfg: &'a RunConfig,
    dims: &'a ModelDims,
    arena: &'a CaptureArena,
    kernels: &'a StepKernels,
    blas: &'a StepBlas,
    setup_stream: &'a Arc<CudaStream>,
    max_blocks_per_seq: usize,
}

/// Everything one cell needs across the session phases.
struct PreparedCell {
    cell: CaptureCell,
    bucket: BucketIdx,
    num_splits: u32,
    sizes: StaticSizes,
    ptrs: StepPtrs,
    staging: StagingPtrs,
    plan: VariationPlan,
}

/// Per-cell device buffers, allocated before the session seals; the recording takes them over.
struct CellBuffers {
    statics: Vec<CudaSlice<u8>>,
    staging: Vec<CudaSlice<u8>>,
    #[cfg(feature = "nccl")]
    comm: Option<Comm>,
}

/// The all-reduce mirror: the f32 buffer every cell's collective reduces in, and its snapshotted
/// address.
struct Mirror<'a> {
    buffer: &'a mut CudaSlice<f32>,
    ptr: u64,
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

/// Runs the full capture matrix and writes `findings.md`, `measurements.json`, and per-cell
/// graph topology dumps into `cfg.out_dir`.
pub fn run(cfg: RunConfig) -> Result<Vec<CellReport>> {
    validate(&cfg)?;
    let plan = RunPlan::new(&cfg);

    // Allocation phase. The context comes first — it disables cudarc's event tracking before
    // anything is allocated — then the session, then every buffer, handle and communicator.
    let ctx = RuntimeContext::new(cfg.device_ordinal)?;
    let allocation = Allocation::new(&ctx)?;
    let sm_count = ctx
        .cuda()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| anyhow!("querying SM count: {:?}", e.0))? as usize;
    let setup_stream = ctx.cuda().default_stream();
    let kernels = StepKernels::compile_and_load(&ctx).context("compiling step kernels")?;
    let blas = StepBlas::new().context("creating the cuBLAS handle")?;

    // Shared device state; declared before the session's graph set exists so it outlives it.
    let (model_slices, model) = alloc::model(&kernels, &setup_stream, &plan.dims, cfg.seed)
        .context("allocating and filling weights")?;
    let (kv_slices, kv_ptrs) = alloc::kv(
        &kernels,
        &setup_stream,
        &plan.dims,
        plan.total_blocks,
        cfg.page_block,
        cfg.seed,
    )
    .context("allocating and filling the KV pool")?;
    let arena_buf = alloc::bytes(&setup_stream, plan.arena.total_size().max(1))?;
    let arena_base = alloc::addr(&arena_buf, &setup_stream);
    let max_bucket = plan.ladder.first().copied().unwrap_or(1);
    let mut allreduce_buf: CudaSlice<f32> = setup_stream
        .alloc_zeros(max_bucket * plan.dims.hidden)
        .map_err(|e| anyhow!("allocating the all-reduce mirror: {:?}", e.0))?;
    let allreduce_ptr = {
        let (ptr, _guard) = allreduce_buf.device_ptr(&setup_stream);
        ptr
    };
    let mut mirror = Mirror {
        buffer: &mut allreduce_buf,
        ptr: allreduce_ptr,
    };

    let deps = Deps {
        cfg: &cfg,
        dims: &plan.dims,
        arena: &plan.arena,
        kernels: &kernels,
        blas: &blas,
        setup_stream: &setup_stream,
        max_blocks_per_seq: plan.max_blocks_per_seq,
    };
    let mut prepared = Vec::new();
    let mut buffers = Vec::new();
    for (index, cell) in plan.cells.iter().enumerate() {
        let (cell_prepared, cell_buffers) = prepare_cell(
            &deps,
            &allocation,
            *cell,
            index,
            &plan.ladder,
            &model,
            &kv_ptrs,
            arena_base,
            plan.total_blocks,
            sm_count,
        )?;
        prepared.push(cell_prepared);
        buffers.push(cell_buffers);
    }
    alloc::synchronize(&setup_stream)?;

    // Capture phase: per cell, warmup at the exact shape, then the recording. No address moves
    // and no handle binds past the seal.
    let mut capture = allocation.seal();
    let (mut reports, recorded) =
        capture_cells(&deps, &mut capture, &mut prepared, buffers, &mut mirror);

    // Replay phase: the identity loop and the soak for every cell that recorded.
    let replay = capture.seal();
    exercise_cells(
        &deps,
        &replay,
        &mut prepared,
        &recorded,
        &mut mirror,
        &mut reports,
    );

    write_findings(&cfg, sm_count, &plan.ladder, &reports)?;

    // The graph set dies before the shared device state whose addresses it baked.
    drop(replay);
    drop(model_slices);
    drop(kv_slices);
    Ok(reports)
}

/// Builds one cell's plan, buffers, and address table (allocation only — no capture).
#[allow(clippy::too_many_arguments)] // one-call fan-in of the run's shared state; a struct would only rename it
fn prepare_cell(
    deps: &Deps<'_>,
    allocation: &Allocation,
    cell: CaptureCell,
    cell_index: usize,
    ladder: &[usize],
    model: &WeightPtrs,
    kv_ptrs: &[(u64, u64)],
    arena_base: u64,
    total_blocks: usize,
    sm_count: usize,
) -> Result<(PreparedCell, CellBuffers)> {
    let cfg = deps.cfg;
    let dims = deps.dims;
    let bucket_pos = ladder
        .iter()
        .position(|&b| b == cell.batch_size)
        .expect("matrix cells come from the ladder");
    let num_splits = splits::num_splits(
        cell.batch_size,
        dims.num_q_heads,
        dims.head_dim,
        cfg.max_seqlen,
        sm_count,
    );
    let sizes = StaticSizes::for_bucket(dims, cell.batch_size, deps.max_blocks_per_seq, num_splits);
    let plan = VariationPlan::new(PlanConfig {
        batch_size: cell.batch_size,
        page_block: cfg.page_block,
        max_blocks_per_seq: deps.max_blocks_per_seq,
        total_blocks,
        start_seqlen: cfg.start_seqlen,
        planned_steps: 1 + cfg.identity_steps + cfg.soak_replays,
        vocab: dims.vocab,
        seed: cfg.seed ^ ((cell_index as u64) << 32),
    })?;

    let stream = deps.setup_stream;
    let statics: Vec<CudaSlice<u8>> = [
        sizes.token_ids,
        sizes.seqlens_k,
        sizes.block_table,
        sizes.slot_mapping,
        sizes.logits,
        sizes.argmax,
        sizes.softmax_lse,
        sizes.lse_accum.max(4),
        sizes.o_accum.max(4),
    ]
    .iter()
    .map(|&bytes| alloc::bytes(stream, bytes))
    .collect::<Result<_>>()?;
    let staging: Vec<CudaSlice<u8>> = [
        sizes.token_ids,
        sizes.seqlens_k,
        sizes.block_table,
        sizes.slot_mapping,
    ]
    .iter()
    .map(|&bytes| alloc::bytes(stream, bytes))
    .collect::<Result<_>>()?;

    let s = |i: usize| alloc::addr(&statics[i], stream);
    let layers = model
        .layers
        .iter()
        .zip(kv_ptrs)
        .map(|(weights, &(k_cache, v_cache))| LayerPtrs {
            w_qkv: weights.w_qkv,
            w_o: weights.w_o,
            w_gate: weights.w_gate,
            w_up: weights.w_up,
            w_down: weights.w_down,
            rms1: weights.rms1,
            rms2: weights.rms2,
            k_cache,
            v_cache,
        })
        .collect();
    let ptrs = StepPtrs {
        embedding: model.embedding,
        final_norm: model.final_norm,
        lm_head: model.lm_head,
        layers,
        arena_base,
        token_ids: s(0),
        seqlens_k: s(1),
        block_table: s(2),
        slot_mapping: s(3),
        logits: s(4),
        argmax: s(5),
        softmax_lse: s(6),
        lse_accum: s(7),
        o_accum: s(8),
    };
    let staging_ptrs = StagingPtrs {
        token_ids: alloc::addr(&staging[0], stream),
        seqlens_k: alloc::addr(&staging[1], stream),
        block_table: alloc::addr(&staging[2], stream),
        slot_mapping: alloc::addr(&staging[3], stream),
    };

    #[cfg(feature = "nccl")]
    let comm = if cell.contents == StepContents::DecodeAllReduce {
        // NCCL init allocates, so it happens here — in Allocation, strictly before the seal.
        let id = Id::new().map_err(|e| anyhow!("ncclGetUniqueId: {:?}", e.0))?;
        Some(allocation.stream().nccl_comm(0, 1, id)?)
    } else {
        None
    };
    #[cfg(not(feature = "nccl"))]
    let _ = (cell.contents, allocation);

    Ok((
        PreparedCell {
            cell,
            bucket: BucketIdx(bucket_pos),
            num_splits,
            sizes,
            ptrs,
            staging: staging_ptrs,
            plan,
        },
        CellBuffers {
            statics,
            staging,
            #[cfg(feature = "nccl")]
            comm,
        },
    ))
}

/// One warmup pass at the cell's exact shape, then the recording. The session's entry takes
/// ownership of every baked buffer, and the communicator is attached before the next recording.
fn capture_cell(
    deps: &Deps<'_>,
    capture: &mut Capture,
    prepared: &mut PreparedCell,
    buffers: CellBuffers,
    mirror: &mut Mirror<'_>,
    report: &mut CellReport,
) -> Result<GraphIdx> {
    let step_ctx = step_context(deps, prepared);
    let warmup = prepared.plan.next_step();
    capture
        .warm_up(&mut StepWork::Upload {
            staging: &prepared.staging,
            inputs: &warmup,
        })
        .context("warmup upload")?;
    {
        #[cfg(feature = "nccl")]
        let all_reduce = make_all_reduce(buffers.comm.as_ref(), mirror);
        #[cfg(not(feature = "nccl"))]
        let all_reduce = make_all_reduce(mirror);
        capture
            .warm_up(&mut StepWork::Decode {
                ctx: &step_ctx,
                ptrs: &prepared.ptrs,
                staging: &prepared.staging,
                sizes: &prepared.sizes,
                all_reduce,
            })
            .context("warmup step")?;
    }

    let free_before = alloc::free_memory()?;
    let capture_started = Instant::now();
    #[cfg(feature = "nccl")]
    let CellBuffers {
        statics,
        staging,
        comm,
    } = buffers;
    #[cfg(not(feature = "nccl"))]
    let CellBuffers { statics, staging } = buffers;
    // Statics order: the four copy-in destinations, then logits and argmax, then accumulators.
    let mut statics = statics.into_iter();
    let mut workspaces: Vec<CudaSlice<u8>> = statics.by_ref().take(4).collect();
    let outputs: Vec<CudaSlice<u8>> = statics.by_ref().take(2).collect();
    workspaces.extend(statics);
    let baked = BakedBuffers {
        inputs: staging,
        outputs,
        workspaces,
    };

    let idx = {
        #[cfg(feature = "nccl")]
        let all_reduce = make_all_reduce(comm.as_ref(), mirror);
        #[cfg(not(feature = "nccl"))]
        let all_reduce = make_all_reduce(mirror);
        capture
            .record(
                &mut StepWork::Decode {
                    ctx: &step_ctx,
                    ptrs: &prepared.ptrs,
                    staging: &prepared.staging,
                    sizes: &prepared.sizes,
                    all_reduce,
                },
                baked,
            )
            .context("recording the step")?
    };
    report.capture_ms = Some(capture_started.elapsed().as_secs_f64() * 1e3);
    #[cfg(feature = "nccl")]
    if let Some(comm) = comm {
        capture.attach_comm(idx, comm);
    }

    let free_after = alloc::free_memory()?;
    report.graph_dedicated_bytes = Some(free_before - free_after);

    let graph = capture.entry(idx).graph();
    let dot_path = deps
        .cfg
        .out_dir
        .join(format!("{}.dot", prepared.cell.label()));
    graph.write_debug_dot(&dot_path, 0)?;
    report.graph_node_count = Some(graph.node_count()?);
    Ok(idx)
}

/// The identity loop and the soak, recording timings, divergences, pointer stability, and the
/// memory deltas.
fn exercise(
    deps: &Deps<'_>,
    replay: &Replay,
    prepared: &mut PreparedCell,
    idx: GraphIdx,
    mirror: &mut Mirror<'_>,
    report: &mut CellReport,
) -> Result<()> {
    let step_ctx = step_context(deps, prepared);
    let baked = baked_ptrs(replay.entry(idx), deps.setup_stream);
    let mut replay_enqueue = Vec::new();
    let mut replay_step = Vec::new();
    let mut eager_enqueue = Vec::new();
    let mut eager_step = Vec::new();
    let mut replay_logits = vec![0u8; prepared.sizes.logits];
    let mut replay_argmax = vec![0u8; prepared.sizes.argmax];
    let mut eager_logits = vec![0u8; prepared.sizes.logits];
    let mut eager_argmax = vec![0u8; prepared.sizes.argmax];

    for step_index in 0..deps.cfg.identity_steps {
        let inputs = prepared.plan.next_step();
        replay.run(&mut StepWork::Upload {
            staging: &prepared.staging,
            inputs: &inputs,
        })?;

        let start = Instant::now();
        replay.replay(idx)?;
        replay_enqueue.push(start.elapsed().as_secs_f64() * 1e6);
        replay.synchronize()?;
        replay_step.push(start.elapsed().as_secs_f64() * 1e6);

        if baked_ptrs(replay.entry(idx), deps.setup_stream) != baked {
            bail!("baked device pointers moved between capture and replay {step_index}");
        }
        alloc::read_back(prepared.ptrs.logits, &mut replay_logits)?;
        alloc::read_back(prepared.ptrs.argmax, &mut replay_argmax)?;

        let start = Instant::now();
        {
            #[cfg(feature = "nccl")]
            let all_reduce = make_all_reduce(replay.entry(idx).comm(), mirror);
            #[cfg(not(feature = "nccl"))]
            let all_reduce = make_all_reduce(mirror);
            replay
                .run(&mut StepWork::Decode {
                    ctx: &step_ctx,
                    ptrs: &prepared.ptrs,
                    staging: &prepared.staging,
                    sizes: &prepared.sizes,
                    all_reduce,
                })
                .with_context(|| format!("eager step {step_index}"))?;
        }
        eager_enqueue.push(start.elapsed().as_secs_f64() * 1e6);
        replay.synchronize()?;
        eager_step.push(start.elapsed().as_secs_f64() * 1e6);
        alloc::read_back(prepared.ptrs.logits, &mut eager_logits)?;
        alloc::read_back(prepared.ptrs.argmax, &mut eager_argmax)?;

        if let Some(divergence) = first_bf16_divergence(&replay_logits, &eager_logits) {
            report.divergence = Some(DivergenceReport {
                step: step_index,
                divergence,
            });
            bail!("bit-identity failed in the logits at step {step_index}");
        }
        if replay_argmax != eager_argmax {
            bail!("bit-identity failed in the argmax outputs at step {step_index}");
        }
        report.identity_steps += 1;
    }
    report.replay_enqueue = Stats::from_micros(replay_enqueue);
    report.replay_step = Stats::from_micros(replay_step);
    report.eager_enqueue = Stats::from_micros(eager_enqueue);
    report.eager_step = Stats::from_micros(eager_step);

    soak(deps, replay, prepared, idx, report)
}

/// The replay-only soak: every baked address must hold after each replay, and free memory must
/// be flat after the first warm replay — a nonzero delta fails the cell.
fn soak(
    deps: &Deps<'_>,
    replay: &Replay,
    prepared: &mut PreparedCell,
    idx: GraphIdx,
    report: &mut CellReport,
) -> Result<()> {
    let baked = baked_ptrs(replay.entry(idx), deps.setup_stream);
    let mut free_after_warm = None;
    for soak_index in 0..deps.cfg.soak_replays {
        let inputs = prepared.plan.next_step();
        replay.run(&mut StepWork::Upload {
            staging: &prepared.staging,
            inputs: &inputs,
        })?;
        replay
            .replay(idx)
            .with_context(|| format!("soak replay {soak_index}"))?;
        if soak_index == 0 || soak_index % 64 == 63 {
            replay.synchronize()?;
        }
        if baked_ptrs(replay.entry(idx), deps.setup_stream) != baked {
            bail!("baked device pointers moved during soak replay {soak_index}");
        }
        if soak_index == 0 {
            free_after_warm = Some(alloc::free_memory()?);
        }
        report.soak_replays += 1;
    }
    replay.synchronize()?;
    let Some(free_after_warm) = free_after_warm else {
        return Ok(());
    };
    let delta = free_after_warm - alloc::free_memory()?;
    report.soak_mem_delta_bytes = Some(delta);
    if delta != 0 {
        bail!(
            "free memory moved by {delta} bytes across {} soak replays",
            report.soak_replays
        );
    }
    Ok(())
}

/// One cell's step configuration; copies the cell's scalars, so it borrows only from `deps`.
fn step_context<'a>(deps: &Deps<'a>, prepared: &PreparedCell) -> StepContext<'a> {
    StepContext {
        kernels: deps.kernels,
        blas: deps.blas,
        dims: deps.dims,
        arena: deps.arena,
        bucket: prepared.bucket,
        batch_size: prepared.cell.batch_size,
        page_block: deps.cfg.page_block,
        max_blocks_per_seq: deps.max_blocks_per_seq,
        num_splits: prepared.num_splits,
        rope_theta: ROPE_THETA,
        rms_eps: RMS_EPS,
    }
}

/// The all-reduce hook parts for a cell with a communicator, wherever the communicator lives —
/// the cell's pre-record buffers or its recorded entry.
#[cfg(feature = "nccl")]
fn make_all_reduce<'a>(
    comm: Option<&'a Comm>,
    mirror: &'a mut Mirror<'_>,
) -> Option<AllReduce<'a>> {
    let comm = comm?;
    Some(AllReduce {
        comm,
        buffer: &mut *mirror.buffer,
        buffer_ptr: mirror.ptr,
    })
}

/// Without the nccl feature no cell has a communicator, so no step gets a hook.
#[cfg(not(feature = "nccl"))]
fn make_all_reduce<'a>(mirror: &'a mut Mirror<'_>) -> Option<AllReduce<'a>> {
    let _ = (&mirror.buffer, mirror.ptr);
    None
}

/// The Capture-phase pass: every cell warmed and recorded, failures noted per report.
fn capture_cells(
    deps: &Deps<'_>,
    capture: &mut Capture,
    prepared: &mut [PreparedCell],
    buffers: Vec<CellBuffers>,
    mirror: &mut Mirror<'_>,
) -> (Vec<CellReport>, Vec<Option<GraphIdx>>) {
    let mut reports = Vec::new();
    let mut recorded = Vec::new();
    for (cell_prepared, cell_buffers) in prepared.iter_mut().zip(buffers) {
        let mut report = CellReport::new(cell_prepared.cell.label());
        match capture_cell(
            deps,
            capture,
            cell_prepared,
            cell_buffers,
            mirror,
            &mut report,
        ) {
            Ok(idx) => recorded.push(Some(idx)),
            Err(err) => {
                report.failure = Some(format!("{err:#}"));
                recorded.push(None);
            }
        }
        reports.push(report);
    }
    (reports, recorded)
}

/// The Replay-phase pass: the identity loop and the soak for every cell that recorded.
fn exercise_cells(
    deps: &Deps<'_>,
    replay: &Replay,
    prepared: &mut [PreparedCell],
    recorded: &[Option<GraphIdx>],
    mirror: &mut Mirror<'_>,
    reports: &mut [CellReport],
) {
    for ((cell_prepared, idx), report) in prepared.iter_mut().zip(recorded).zip(reports.iter_mut())
    {
        let Some(idx) = idx else { continue };
        if let Err(err) = exercise(deps, replay, cell_prepared, *idx, mirror, report) {
            report.failure = Some(format!("{err:#}"));
        }
    }
}

/// Renders findings.md and measurements.json into the run's output directory.
fn write_findings(
    cfg: &RunConfig,
    sm_count: usize,
    ladder: &[usize],
    reports: &[CellReport],
) -> Result<()> {
    let header = vec![
        format!("device ordinal {}, {} SMs", cfg.device_ordinal, sm_count),
        format!(
            "{} layers, buckets {:?}, {} identity steps, {} soak replays",
            cfg.layers, ladder, cfg.identity_steps, cfg.soak_replays
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

/// Every device address baked into the cell's graph, for the per-replay stability assert.
fn baked_ptrs(entry: &GraphEntry, stream: &Arc<CudaStream>) -> Vec<u64> {
    entry
        .inputs()
        .iter()
        .chain(entry.outputs())
        .chain(entry.workspaces())
        .map(|slice| alloc::addr(slice, stream))
        .collect()
}
