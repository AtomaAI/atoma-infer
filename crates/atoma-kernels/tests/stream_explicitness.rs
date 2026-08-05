//! Guards the stream-explicitness invariant: every kernel launch runs on the caller's stream.
//!
//! These checks read the kernel sources rather than running kernels, so they hold the invariant in
//! CPU CI where no CUDA toolkit or GPU exists. Without them the invariant is only observable on a
//! GPU rig, where a regression would surface as a silent correctness bug rather than a test
//! failure.

use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Reads a repository file, stripping nothing: the checks below are deliberately textual.
fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Collects every `.rs` file under a `src/` directory of the workspace crates.
///
/// Restricted to `src/` so this file's own diagnostic strings are not mistaken for offending code.
fn workspace_rust_sources() -> Vec<PathBuf> {
    let crates_dir = crate_root()
        .parent()
        .expect("atoma-kernels must live under crates/")
        .to_path_buf();
    let mut sources = Vec::new();
    let mut stack = vec![crates_dir];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("failed to read dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("failed to read dir entry").path();
            // Skip the vendored cutlass submodule and any build artefacts.
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if name != "cutlass" && name != "target" {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs")
                && path.components().any(|c| c.as_os_str() == "src")
            {
                sources.push(path);
            }
        }
    }
    sources
}

/// Collects every `.cu` entry-point source in the kernels directory.
///
/// The per-head-dimension `flash_fwd_*` files only instantiate templates; the launch-configuring
/// entry points are the ones that can name a stream.
fn kernel_entry_point_sources() -> Vec<PathBuf> {
    let kernels_dir = crate_root().join("kernels");
    let entries = std::fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", kernels_dir.display()));
    entries
        .map(|entry| entry.expect("failed to read dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            !name.starts_with("flash_fwd_")
        })
        .collect()
}

#[test]
fn no_kernel_source_hardcodes_the_default_stream() {
    // `cudaStream_t stream = 0` and its const form bind the legacy default stream, so any launch
    // below such a binding ignores the stream the caller asked for.
    let offenders: Vec<_> = kernel_entry_point_sources()
        .into_iter()
        .filter(|path| {
            let source = read(path);
            source.contains("cudaStream_t stream = 0")
                || source.contains("const cudaStream_t stream = 0")
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "these kernel sources bind the legacy default stream instead of taking the caller's \
         stream as a parameter: {offenders:?}"
    );
}

#[test]
fn flash_attention_entry_point_takes_a_stream() {
    let source = read(&crate_root().join("kernels/flash_api.cu"));

    let run_mha = source
        .split_once("extern \"C\" void run_mha(")
        .expect("flash_api.cu must declare `extern \"C\" void run_mha(`")
        .1;
    let params = run_mha
        .split_once(')')
        .expect("run_mha parameter list must terminate")
        .0;
    assert!(
        params.contains("cudaStream_t"),
        "run_mha must accept the caller's stream as a parameter, got: {params}"
    );
}

#[test]
fn no_rust_source_uses_a_stream_the_caller_did_not_supply() {
    // `fork_default_stream` creates a stream nothing waits on; `wait_for` and the raw `.stream`
    // field are the device-side halves of that same pattern, all removed by the cudarc 0.19 port.
    const BANNED: [&str; 3] = [
        "fork_default_stream",
        ".wait_for(",
        "device_ptr_write_target",
    ];

    let offenders: Vec<_> = workspace_rust_sources()
        .into_iter()
        .filter_map(|path| {
            let source = read(&path);
            let hits: Vec<_> = BANNED
                .iter()
                .filter(|pattern| source.contains(**pattern))
                .collect();
            (!hits.is_empty()).then_some((path, hits))
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "these sources drive a stream the caller did not supply; recover the caller's stream from \
         candle tensor storage instead: {offenders:?}"
    );
}
