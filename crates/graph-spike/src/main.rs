//! CLI for the #143 graph spike: `plan` prints the matrix and memory budget on any machine;
//! `run` executes the capture matrix on a CUDA rig.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use graph_spike::dims::ModelDims;
use graph_spike::gpu::harness::{self, RunConfig};
use graph_spike::layout::{build_arena, kv_cache_bytes_each, StaticSizes};
use graph_spike::matrix::capture_matrix;
use graph_spike::splits;

#[derive(Parser)]
#[command(about = "CUDA-graph capture spike for the tensor path (#143)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the capture matrix, arena footprint, and device memory budget without a GPU.
    Plan(CommonArgs),
    /// Run the capture matrix on the rig, writing findings and measurements to --out.
    Run(RunArgs),
}

#[derive(Args)]
struct CommonArgs {
    /// Transformer layers in the shaped step (the runsheet says 2–4).
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Capture-matrix buckets (batch sizes).
    #[arg(long, value_delimiter = ',', default_values_t = [1usize, 8, 32, 64])]
    buckets: Vec<usize>,
    /// Paged-KV block size in tokens; FA2 requires a multiple of 16.
    #[arg(long, default_value_t = 16)]
    page_block: usize,
    /// Maximum tokens per sequence; fixes the block-table width the graphs bake.
    #[arg(long, default_value_t = 2048)]
    max_seqlen: usize,
    /// Tokens already in the cache when a cell starts (sequence i adds i % page_block).
    #[arg(long, default_value_t = 128)]
    start_seqlen: usize,
    /// Bit-identity comparison steps per cell.
    #[arg(long, default_value_t = 32)]
    steps: usize,
    /// Replay-only soak iterations per cell.
    #[arg(long, default_value_t = 1000)]
    soak: usize,
    /// Include the ws=1 NCCL all-reduce cells (needs the nccl feature to run).
    #[arg(long)]
    nccl: bool,
    /// SM count used by `plan` for the split heuristic (H100 PCIe has 114; `run` queries the
    /// device instead).
    #[arg(long, default_value_t = 114)]
    sms: usize,
    #[arg(long, default_value_t = 0x5eed_5eed)]
    seed: u64,
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Output directory for findings.md, measurements.json, and the .dot topology dumps.
    #[arg(long, default_value = ".scratch/graph-spike")]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Plan(args) => plan(&args),
        Command::Run(args) => run(args),
    }
}

#[allow(clippy::print_stdout, clippy::cast_precision_loss)]
fn plan(args: &CommonArgs) -> Result<()> {
    let dims = ModelDims::llama_8b_shaped(args.layers);
    let cells = capture_matrix(&args.buckets, args.nccl);
    let mut ladder = args.buckets.clone();
    ladder.sort_unstable_by(|a, b| b.cmp(a));
    ladder.dedup();
    let arena = build_arena(&dims, &ladder);
    let max_blocks_per_seq = args.max_seqlen / args.page_block;
    let total_blocks = ladder.first().copied().unwrap_or(1) * max_blocks_per_seq;
    let kv_bytes = 2 * dims.num_layers * kv_cache_bytes_each(&dims, total_blocks, args.page_block);

    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    println!("# graph-spike plan\n");
    println!(
        "model: {} layers, hidden {}, {}q/{}kv heads, head_dim {}, ffn {}, vocab {}",
        dims.num_layers,
        dims.hidden,
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.ffn,
        dims.vocab
    );
    println!("weights: {:.1} MiB", mib(dims.total_weight_bytes()));
    println!(
        "kv pool: {:.1} MiB ({} blocks x {} tokens, {} layers x K+V)",
        mib(kv_bytes),
        total_blocks,
        args.page_block,
        dims.num_layers
    );
    println!("arena: {:.1} MiB (largest bucket)", mib(arena.total_size()));
    println!("\ncells (execution order):");
    let mut statics_total = 0;
    for cell in &cells {
        let num_splits = splits::num_splits(
            cell.batch_size,
            dims.num_q_heads,
            dims.head_dim,
            args.max_seqlen,
            args.sms,
        );
        let sizes = StaticSizes::for_bucket(&dims, cell.batch_size, max_blocks_per_seq, num_splits);
        statics_total += sizes.total_bytes() + sizes.input_bytes();
        println!(
            "  {:>10}  num_splits {:>2}  statics+staging {:>8.2} MiB",
            cell.label(),
            num_splits,
            mib(sizes.total_bytes() + sizes.input_bytes()),
        );
    }
    println!(
        "\ntotal budget: {:.1} MiB (+ per-graph driver overhead, which the run measures — the \
         A4 calibration constant)",
        mib(dims.total_weight_bytes() + kv_bytes + arena.total_size() + statics_total)
    );
    println!(
        "steps per cell: 1 warmup + {} identity + {} soak (needs start_seqlen {} + {} <= {})",
        args.steps,
        args.soak,
        args.start_seqlen,
        args.steps + args.soak + args.page_block,
        args.max_seqlen
    );
    Ok(())
}

#[allow(clippy::print_stdout)]
fn run(args: RunArgs) -> Result<()> {
    let common = args.common;
    let reports = harness::run(RunConfig {
        device_ordinal: args.device,
        layers: common.layers,
        buckets: common.buckets,
        identity_steps: common.steps,
        soak_replays: common.soak,
        page_block: common.page_block,
        max_seqlen: common.max_seqlen,
        start_seqlen: common.start_seqlen,
        seed: common.seed,
        include_all_reduce: common.nccl,
        out_dir: args.out.clone(),
    })?;

    let failed = reports.iter().filter(|r| r.failure.is_some()).count();
    println!(
        "{} cells, {} failed; findings in {}",
        reports.len(),
        failed,
        args.out.join("findings.md").display()
    );
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
