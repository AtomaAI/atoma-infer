//! The spike driver: setup, warmup, capture, replay-vs-eager comparison, soak, and measurement
//! for every capture-matrix cell.
//!
//! Ownership follows the harness rules: everything is allocated strictly before the first
//! capture; each cell's `GraphEntry` owns the buffers baked only into its graph; weights, KV
//! caches, the arena, staging, and the all-reduce mirror are shared across cells and outlive
//! every entry (locals drop in reverse declaration order in [`run`]). A cell failure is
//! classified and recorded, the capture is drained, and the run continues — failures are spec
//! findings, not aborts.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::{fs, mem};

use anyhow::{anyhow, bail, Context, Result};
use atoma_runtime::arena::{BucketIdx, CaptureArena};
use atoma_runtime::capture::{self, CaptureState};
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::graph_entry::GraphEntry;
use atoma_runtime::stream::CaptureStream;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{result, sys, CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm, Id, ReduceOp};

use crate::compare::first_bf16_divergence;
use crate::dims::{ModelDims, BF16_BYTES};
use crate::gpu::blas::SpikeBlas;
use crate::gpu::kernels::SpikeKernels;
use crate::gpu::step::{self, LayerPtrs, StagingPtrs, StepContext, StepPtrs};
use crate::layout::{build_arena, kv_cache_bytes_each, StaticSizes};
#[cfg(feature = "nccl")]
use crate::matrix::StepContents;
use crate::matrix::{capture_matrix, CaptureCell};
use crate::report::{render_markdown, CellReport, DivergenceReport, Stats};
use crate::splits;
use crate::variation::{PlanConfig, StepInputs, VariationPlan};

const WEIGHT_SCALE: f32 = 0.05;
const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;
/// bf16(1.0), the RMSNorm gain everywhere so normed activations stay unit-scale.
const BF16_ONE: u16 = 0x3F80;

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
    kernels: &'a SpikeKernels,
    blas: &'a SpikeBlas,
    capture: &'a CaptureStream,
    setup_stream: &'a Arc<CudaStream>,
    stream: sys::CUstream,
    max_blocks_per_seq: usize,
}

/// The pieces the per-layer all-reduce hook needs. Only all-reduce cells construct one, and only
/// the `nccl` build can run them.
pub struct AllReduce<'a> {
    #[cfg(feature = "nccl")]
    pub comm: &'a Comm,
    pub buffer: &'a mut CudaSlice<f32>,
    pub buffer_ptr: u64,
}

/// A cell whose graph instantiated: the entry owns the baked per-cell buffers, staging outlives
/// the graph (field order is drop order).
struct CellState {
    entry: GraphEntry,
    staging: Vec<CudaSlice<u8>>,
}

/// Everything one cell needs across its phases.
struct PreparedCell {
    cell: CaptureCell,
    bucket: BucketIdx,
    num_splits: u32,
    sizes: StaticSizes,
    ptrs: StepPtrs,
    staging: StagingPtrs,
    plan: VariationPlan,
}

/// Per-cell device buffers, allocated before the first capture.
struct CellBuffers {
    statics: Vec<CudaSlice<u8>>,
    staging: Vec<CudaSlice<u8>>,
    #[cfg(feature = "nccl")]
    comm: Option<Comm>,
}

/// Runs the full capture matrix and writes `findings.md`, `measurements.json`, and per-cell
/// graph topology dumps into `cfg.out_dir`.
pub fn run(cfg: RunConfig) -> Result<Vec<CellReport>> {
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
        .with_context(|| format!("creating out dir {}", cfg.out_dir.display()))?;

    let dims = ModelDims::llama_8b_shaped(cfg.layers);
    let cells = capture_matrix(&cfg.buckets, cfg.include_all_reduce);
    let ladder: Vec<usize> = {
        let mut buckets = cfg.buckets.clone();
        buckets.sort_unstable_by(|a, b| b.cmp(a));
        buckets.dedup();
        buckets
    };
    let arena = build_arena(&dims, &ladder);
    let max_blocks_per_seq = cfg.max_seqlen / cfg.page_block;
    let total_blocks = ladder.first().copied().unwrap_or(1) * max_blocks_per_seq;

    // Context first: it disables cudarc's event tracking before anything is allocated.
    let ctx = RuntimeContext::new(cfg.device_ordinal)?;
    let sm_count = ctx
        .cuda()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| anyhow!("querying SM count: {:?}", e.0))? as usize;
    let capture_stream = CaptureStream::new(&ctx)?;
    let stream = capture_stream.cu_stream();
    let setup_stream = ctx.cuda().default_stream();
    let kernels = SpikeKernels::compile_and_load(&ctx).context("compiling spike kernels")?;
    let blas = SpikeBlas::new(stream).context("creating the cuBLAS handle")?;

    // Shared device state; declared before `states` so every graph drops first.
    let (model_slices, model) = alloc_model(&kernels, &setup_stream, &dims, cfg.seed)
        .context("allocating and filling weights")?;
    let (kv_slices, kv_ptrs) = alloc_kv(
        &kernels,
        &setup_stream,
        &dims,
        total_blocks,
        cfg.page_block,
        cfg.seed,
    )
    .context("allocating and filling the KV pool")?;
    let arena_buf = alloc_bytes(&setup_stream, arena.total_size().max(1))?;
    let arena_base = addr(&arena_buf, &setup_stream);
    let max_bucket = ladder.first().copied().unwrap_or(1);
    let mut allreduce_buf: CudaSlice<f32> = setup_stream
        .alloc_zeros(max_bucket * dims.hidden)
        .map_err(|e| anyhow!("allocating the all-reduce mirror: {:?}", e.0))?;
    let allreduce_ptr = {
        let (ptr, _guard) = allreduce_buf.device_ptr(&setup_stream);
        ptr
    };

    let deps = Deps {
        cfg: &cfg,
        dims: &dims,
        arena: &arena,
        kernels: &kernels,
        blas: &blas,
        capture: &capture_stream,
        setup_stream: &setup_stream,
        stream,
        max_blocks_per_seq,
    };
    let mut prepared = Vec::new();
    let mut buffers = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        let (cell_prepared, cell_buffers) = prepare_cell(
            &deps,
            *cell,
            index,
            &ladder,
            &model,
            &kv_ptrs,
            arena_base,
            total_blocks,
            sm_count,
        )?;
        prepared.push(cell_prepared);
        buffers.push(cell_buffers);
    }
    unsafe { result::stream::synchronize(setup_stream.cu_stream()) }
        .map_err(|e| anyhow!("setup synchronize: {:?}", e.0))?;

    // Graph-owning states; declared after everything shared so they drop first.
    let mut states: Vec<CellState> = Vec::new();
    let mut reports = Vec::new();
    for (cell_prepared, cell_buffers) in prepared.into_iter().zip(buffers) {
        let mut report = CellReport::new(cell_prepared.cell.label());
        match run_cell(
            &deps,
            cell_prepared,
            cell_buffers,
            &mut allreduce_buf,
            allreduce_ptr,
            &mut report,
        ) {
            Ok(state) => states.push(state),
            Err(err) => {
                drain_capture(&capture_stream);
                report.failure = Some(format!("{err:#}"));
            }
        }
        reports.push(report);
    }

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
    let markdown = render_markdown(&header, &reports);
    fs::write(cfg.out_dir.join("findings.md"), markdown)?;
    let json = serde_json::to_string_pretty(&reports)?;
    fs::write(cfg.out_dir.join("measurements.json"), json)?;
    drop(model_slices);
    drop(kv_slices);
    Ok(reports)
}

/// Allocates and fills every weight tensor, returning the owning slices and the address table.
fn alloc_model(
    kernels: &SpikeKernels,
    stream: &Arc<CudaStream>,
    dims: &ModelDims,
    seed: u64,
) -> Result<(Vec<CudaSlice<u8>>, StepPtrsModel)> {
    let raw = stream.cu_stream();
    let mut slices = Vec::new();
    let mut next_seed = seed;
    let mut fill = |elements: usize| -> Result<u64> {
        let slice = alloc_bytes(stream, elements * BF16_BYTES)?;
        let ptr = addr(&slice, stream);
        next_seed = next_seed.wrapping_add(1);
        unsafe { kernels.fill_random_bf16(raw, ptr, elements, next_seed, WEIGHT_SCALE) }?;
        slices.push(slice);
        Ok(ptr)
    };

    let embedding = fill(dims.vocab * dims.hidden)?;
    let lm_head = fill(dims.vocab * dims.hidden)?;
    let mut layers = Vec::new();
    for _ in 0..dims.num_layers {
        layers.push(LayerWeights {
            w_qkv: fill(dims.qkv_out() * dims.hidden)?,
            w_o: fill(dims.hidden * dims.hidden)?,
            w_gate: fill(dims.ffn * dims.hidden)?,
            w_up: fill(dims.ffn * dims.hidden)?,
            w_down: fill(dims.hidden * dims.ffn)?,
            rms1: 0,
            rms2: 0,
        });
    }

    // RMSNorm gains are 1.0 so every normed activation stays unit-scale (bf16 overflows near
    // 3e4; the day-0 engine mock died exactly there).
    let ones = vec![BF16_ONE; dims.hidden];
    let mut gain = || -> Result<u64> {
        let slice = alloc_bytes(stream, dims.hidden * BF16_BYTES)?;
        let ptr = addr(&slice, stream);
        unsafe { result::memcpy_htod_async(ptr, &ones, raw) }
            .map_err(|e| anyhow!("uploading RMSNorm gains: {:?}", e.0))?;
        slices.push(slice);
        Ok(ptr)
    };
    for layer in &mut layers {
        layer.rms1 = gain()?;
        layer.rms2 = gain()?;
    }
    let final_norm = gain()?;

    Ok((
        slices,
        StepPtrsModel {
            embedding,
            final_norm,
            lm_head,
            layers,
        },
    ))
}

/// Weight addresses without the per-layer KV caches (which [`alloc_kv`] provides).
struct StepPtrsModel {
    embedding: u64,
    final_norm: u64,
    lm_head: u64,
    layers: Vec<LayerWeights>,
}

struct LayerWeights {
    w_qkv: u64,
    w_o: u64,
    w_gate: u64,
    w_up: u64,
    w_down: u64,
    rms1: u64,
    rms2: u64,
}

/// The KV pool: the owning slices and each layer's (K, V) cache addresses.
type KvPool = (Vec<CudaSlice<u8>>, Vec<(u64, u64)>);

/// Allocates the per-layer paged K and V caches, pre-filled with unit-scale randoms so
/// "historical" positions hold deterministic data both runs share.
fn alloc_kv(
    kernels: &SpikeKernels,
    stream: &Arc<CudaStream>,
    dims: &ModelDims,
    total_blocks: usize,
    page_block: usize,
    seed: u64,
) -> Result<KvPool> {
    let raw = stream.cu_stream();
    let bytes = kv_cache_bytes_each(dims, total_blocks, page_block);
    let elements = bytes / BF16_BYTES;
    let mut slices = Vec::new();
    let mut ptrs = Vec::new();
    for layer in 0..dims.num_layers {
        let mut cache = |salt: u64| -> Result<u64> {
            let slice = alloc_bytes(stream, bytes)?;
            let ptr = addr(&slice, stream);
            // 0x4B56 is "KV": keeps cache fills disjoint from the weight-fill seed sequence.
            unsafe {
                kernels.fill_random_bf16(raw, ptr, elements, seed ^ 0x4B56 ^ salt, WEIGHT_SCALE)
            }?;
            slices.push(slice);
            Ok(ptr)
        };
        let k = cache(2 * layer as u64)?;
        let v = cache(2 * layer as u64 + 1)?;
        ptrs.push((k, v));
    }
    Ok((slices, ptrs))
}

/// Builds one cell's plan, buffers, and address table (allocation only — no capture).
#[allow(clippy::too_many_arguments)]
fn prepare_cell(
    deps: &Deps<'_>,
    cell: CaptureCell,
    cell_index: usize,
    ladder: &[usize],
    model: &StepPtrsModel,
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
    .map(|&bytes| alloc_bytes(stream, bytes))
    .collect::<Result<_>>()?;
    let staging: Vec<CudaSlice<u8>> = [
        sizes.token_ids,
        sizes.seqlens_k,
        sizes.block_table,
        sizes.slot_mapping,
    ]
    .iter()
    .map(|&bytes| alloc_bytes(stream, bytes))
    .collect::<Result<_>>()?;

    let s = |i: usize| addr(&statics[i], stream);
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
        token_ids: addr(&staging[0], stream),
        seqlens_k: addr(&staging[1], stream),
        block_table: addr(&staging[2], stream),
        slot_mapping: addr(&staging[3], stream),
    };

    #[cfg(feature = "nccl")]
    let comm = if cell.contents == StepContents::DecodeAllReduce {
        // NCCL init allocates, so it happens here — strictly before the first capture.
        let id = Id::new().map_err(|e| anyhow!("ncclGetUniqueId: {:?}", e.0))?;
        Some(deps.capture.nccl_comm(0, 1, id)?)
    } else {
        None
    };
    #[cfg(not(feature = "nccl"))]
    let _ = cell.contents;

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

/// Warms up, captures, exercises, and soaks one cell. Pre-instantiate failures return `Err`;
/// once the graph exists, later failures are recorded in `report` and the state is still kept
/// alive and returned.
fn run_cell(
    deps: &Deps<'_>,
    mut prepared: PreparedCell,
    mut buffers: CellBuffers,
    allreduce_buf: &mut CudaSlice<f32>,
    allreduce_ptr: u64,
    report: &mut CellReport,
) -> Result<CellState> {
    let step_ctx = StepContext {
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
        stream: deps.stream,
    };

    // Warmup: the same step, eagerly, so every lazy allocation (cuBLAS workspace, FA2 first
    // call) lands before capture.
    let warmup = prepared.plan.next_step();
    upload_staging(deps.stream, &prepared.staging, &warmup)?;
    unsafe {
        step::copy_inputs(
            deps.stream,
            &prepared.ptrs,
            &prepared.staging,
            &prepared.sizes,
        )
    }?;
    {
        #[cfg(feature = "nccl")]
        let mut ar = make_all_reduce(&buffers, allreduce_buf, allreduce_ptr);
        #[cfg(not(feature = "nccl"))]
        let mut ar: Option<AllReduce<'_>> = None;
        step_with_hook(deps, &step_ctx, &prepared.ptrs, ar.as_mut()).context("warmup step")?;
    }
    sync_outside_capture(deps)?;

    let (free_before, _) =
        result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;

    // Capture and instantiate.
    let capture_started = Instant::now();
    deps.capture.begin_capture()?;
    {
        #[cfg(feature = "nccl")]
        let mut ar = make_all_reduce(&buffers, allreduce_buf, allreduce_ptr);
        #[cfg(not(feature = "nccl"))]
        let mut ar: Option<AllReduce<'_>> = None;
        unsafe {
            step::copy_inputs(
                deps.stream,
                &prepared.ptrs,
                &prepared.staging,
                &prepared.sizes,
            )
        }
        .context("recording copy-in")?;
        step_with_hook(deps, &step_ctx, &prepared.ptrs, ar.as_mut()).context("recording step")?;
    }
    let graph = capture::end_capture_instantiate(deps.capture)?;
    report.capture_ms = Some(capture_started.elapsed().as_secs_f64() * 1e3);

    graph.upload()?;
    sync_outside_capture(deps)?;
    let (free_after, _) = result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;
    report.graph_dedicated_bytes = Some(free_before as i64 - free_after as i64);

    let dot_path = deps
        .cfg
        .out_dir
        .join(format!("{}.dot", prepared.cell.label()));
    unsafe { capture::debug_dot_print(graph.cu_graph(), &dot_path, 0) }?;
    report.graph_node_count = Some(unsafe { capture::graph_nodes(graph.cu_graph()) }?.len());

    let statics = mem::take(&mut buffers.statics);
    let mut statics = statics.into_iter();
    let inputs: Vec<CudaSlice<u8>> = statics.by_ref().take(4).collect();
    let outputs: Vec<CudaSlice<u8>> = statics.by_ref().take(2).collect();
    let workspaces: Vec<CudaSlice<u8>> = statics.collect();
    let entry = GraphEntry::new(inputs, outputs, workspaces, graph);
    #[cfg(feature = "nccl")]
    let entry = match buffers.comm.take() {
        Some(comm) => entry.with_comm(comm),
        None => entry,
    };
    let mut state = CellState {
        entry,
        staging: mem::take(&mut buffers.staging),
    };

    if let Err(err) = exercise(
        deps,
        &step_ctx,
        &mut prepared,
        &mut state,
        allreduce_buf,
        allreduce_ptr,
        report,
    ) {
        report.failure = Some(format!("{err:#}"));
    }
    Ok(state)
}

/// The identity loop and the soak, recording timings, divergences, pointer stability, and the
/// memory deltas.
#[allow(clippy::too_many_arguments)]
fn exercise(
    deps: &Deps<'_>,
    step_ctx: &StepContext<'_>,
    prepared: &mut PreparedCell,
    state: &mut CellState,
    allreduce_buf: &mut CudaSlice<f32>,
    allreduce_ptr: u64,
    report: &mut CellReport,
) -> Result<()> {
    let baked = baked_ptrs(&mut state.entry, &state.staging, deps.setup_stream);
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
        upload_staging(deps.stream, &prepared.staging, &inputs)?;

        let start = Instant::now();
        state.entry.graph().replay()?;
        replay_enqueue.push(start.elapsed().as_secs_f64() * 1e6);
        sync_outside_capture(deps)?;
        replay_step.push(start.elapsed().as_secs_f64() * 1e6);

        let now = baked_ptrs(&mut state.entry, &state.staging, deps.setup_stream);
        if now != baked {
            bail!("baked device pointers moved between capture and replay {step_index}");
        }
        read_back(prepared.ptrs.logits, &mut replay_logits)?;
        read_back(prepared.ptrs.argmax, &mut replay_argmax)?;

        let start = Instant::now();
        unsafe {
            step::copy_inputs(
                deps.stream,
                &prepared.ptrs,
                &prepared.staging,
                &prepared.sizes,
            )
        }?;
        {
            #[cfg(feature = "nccl")]
            let mut ar = make_all_reduce_from_entry(&state.entry, allreduce_buf, allreduce_ptr);
            #[cfg(not(feature = "nccl"))]
            let mut ar: Option<AllReduce<'_>> = {
                let (_, _) = (&allreduce_buf, allreduce_ptr);
                None
            };
            step_with_hook(deps, step_ctx, &prepared.ptrs, ar.as_mut())
                .with_context(|| format!("eager step {step_index}"))?;
        }
        eager_enqueue.push(start.elapsed().as_secs_f64() * 1e6);
        sync_outside_capture(deps)?;
        eager_step.push(start.elapsed().as_secs_f64() * 1e6);
        read_back(prepared.ptrs.logits, &mut eager_logits)?;
        read_back(prepared.ptrs.argmax, &mut eager_argmax)?;

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

    // Soak: replay-only; memory must be flat after the first warm replay.
    let mut free_after_warm = None;
    for soak_index in 0..deps.cfg.soak_replays {
        let inputs = prepared.plan.next_step();
        upload_staging(deps.stream, &prepared.staging, &inputs)?;
        state
            .entry
            .graph()
            .replay()
            .with_context(|| format!("soak replay {soak_index}"))?;
        if soak_index == 0 || soak_index % 64 == 63 {
            sync_outside_capture(deps)?;
        }
        if soak_index == 0 {
            let (free, _) =
                result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;
            free_after_warm = Some(free as i64);
        }
        report.soak_replays += 1;
    }
    sync_outside_capture(deps)?;
    if let Some(free_after_warm) = free_after_warm {
        let (free_end, _) =
            result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;
        report.soak_mem_delta_bytes = Some(free_after_warm - free_end as i64);
    }
    Ok(())
}

/// Runs the step with the cell's all-reduce hook when it has one.
fn step_with_hook(
    deps: &Deps<'_>,
    ctx: &StepContext<'_>,
    ptrs: &StepPtrs,
    all_reduce: Option<&mut AllReduce<'_>>,
) -> Result<()> {
    match all_reduce {
        #[cfg(feature = "nccl")]
        Some(parts) => {
            let kernels = deps.kernels;
            let stream = ctx.stream;
            let buffer_ptr = parts.buffer_ptr;
            let comm = parts.comm;
            let buffer = &mut *parts.buffer;
            let mut hook = |o_proj: u64, elements: usize| -> Result<()> {
                unsafe { kernels.bf16_to_f32(stream, o_proj, buffer_ptr, elements) }?;
                let mut view = buffer.slice_mut(0..elements);
                comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
                    .map_err(|e| anyhow!("ncclAllReduce: {:?}", e.0))?;
                unsafe { kernels.f32_to_bf16(stream, buffer_ptr, o_proj, elements) }?;
                Ok(())
            };
            unsafe { step::run_step(ctx, ptrs, Some(&mut hook)) }
        }
        #[cfg(not(feature = "nccl"))]
        Some(_) => bail!("all-reduce cell in a build without the nccl feature"),
        None => {
            let _ = deps;
            unsafe { step::run_step(ctx, ptrs, None) }
        }
    }
}

#[cfg(feature = "nccl")]
fn make_all_reduce<'a>(
    buffers: &'a CellBuffers,
    allreduce_buf: &'a mut CudaSlice<f32>,
    allreduce_ptr: u64,
) -> Option<AllReduce<'a>> {
    buffers.comm.as_ref().map(|comm| AllReduce {
        comm,
        buffer: allreduce_buf,
        buffer_ptr: allreduce_ptr,
    })
}

#[cfg(feature = "nccl")]
fn make_all_reduce_from_entry<'a>(
    entry: &'a GraphEntry,
    allreduce_buf: &'a mut CudaSlice<f32>,
    allreduce_ptr: u64,
) -> Option<AllReduce<'a>> {
    entry.comm().map(|comm| AllReduce {
        comm,
        buffer: allreduce_buf,
        buffer_ptr: allreduce_ptr,
    })
}

/// Uploads one step's inputs into staging. Pageable H2D is host-synchronous, so the borrowed
/// host slices cannot outlive the copy; the graph itself only ever reads staging via captured
/// D2D nodes.
fn upload_staging(stream: sys::CUstream, staging: &StagingPtrs, inputs: &StepInputs) -> Result<()> {
    unsafe {
        result::memcpy_htod_async(staging.token_ids, &inputs.token_ids, stream)
            .and_then(|()| result::memcpy_htod_async(staging.seqlens_k, &inputs.seqlens_k, stream))
            .and_then(|()| {
                result::memcpy_htod_async(staging.block_table, &inputs.block_table, stream)
            })
            .and_then(|()| {
                result::memcpy_htod_async(staging.slot_mapping, &inputs.slot_mapping, stream)
            })
    }
    .map_err(|e| anyhow!("staging upload: {:?}", e.0))
}

/// Synchronizes the capture stream, refusing to run while a capture is active — the sync ban on
/// `CaptureStream`'s surface exists to protect recordings, and this helper enforces the same
/// rule for the harness's raw escape hatch.
fn sync_outside_capture(deps: &Deps<'_>) -> Result<()> {
    let state = deps.capture.state()?;
    if state != CaptureState::Idle {
        bail!("refusing to synchronize while the stream capture state is {state:?}");
    }
    unsafe { result::stream::synchronize(deps.stream) }
        .map_err(|e| anyhow!("stream synchronize: {:?}", e.0))
}

/// Best-effort drain after a failure so the next cell starts from an idle stream.
fn drain_capture(stream: &CaptureStream) {
    if matches!(
        stream.state(),
        Ok(CaptureState::Active | CaptureState::Invalidated)
    ) {
        let _ = capture::end_capture_discard(stream);
    }
}

/// Every device address baked into the cell's graph, for the per-replay stability assert.
fn baked_ptrs(
    entry: &mut GraphEntry,
    staging: &[CudaSlice<u8>],
    stream: &Arc<CudaStream>,
) -> Vec<u64> {
    let mut ptrs = Vec::new();
    for slice in entry.inputs_mut().iter() {
        ptrs.push(addr(slice, stream));
    }
    for slice in entry.outputs() {
        ptrs.push(addr(slice, stream));
    }
    for slice in entry.workspaces_mut().iter() {
        ptrs.push(addr(slice, stream));
    }
    for slice in staging {
        ptrs.push(addr(slice, stream));
    }
    ptrs
}

/// Synchronous D2H read of `buf.len()` bytes from `src`.
fn read_back(src: u64, buf: &mut [u8]) -> Result<()> {
    unsafe { result::memcpy_dtoh_sync(buf, src) }.map_err(|e| anyhow!("D2H read: {:?}", e.0))
}

fn alloc_bytes(stream: &Arc<CudaStream>, bytes: usize) -> Result<CudaSlice<u8>> {
    stream
        .alloc_zeros::<u8>(bytes.max(1))
        .map_err(|e| anyhow!("allocating {bytes} bytes: {:?}", e.0))
}

/// The device address of a slice. With event tracking disabled at context creation, the guard
/// is a no-op and the address is stable for the slice's lifetime.
fn addr(slice: &CudaSlice<u8>, stream: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}
