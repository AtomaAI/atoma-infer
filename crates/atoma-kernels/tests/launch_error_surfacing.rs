//! Guards the launch-error invariant: a failing CUDA call returns a status that Rust turns into a
//! typed error, and never terminates the process.
//!
//! Like `stream_explicitness.rs`, these checks read the sources rather than running kernels, so
//! they hold in CPU CI where no CUDA toolkit or GPU exists. On a rig the invariant is only
//! observable when a launch actually fails, which no ordinary test run reaches.

use std::path::{Path, PathBuf};

/// The FFI entry points that launch a kernel, with the label passed to `check_launch`.
const LAUNCHERS: [&str; 3] = [
    "run_mha",
    "copy_blocks_cache",
    "reshape_and_cache_flash_cache",
];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Collects every kernel source and header. The vendored CUTLASS submodule sits outside this
/// directory and is therefore not scanned.
fn kernel_sources() -> Vec<PathBuf> {
    let kernels_dir = crate_root().join("kernels");
    let entries = std::fs::read_dir(&kernels_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", kernels_dir.display()));
    entries
        .map(|entry| entry.expect("failed to read dir entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "cu" || extension == "h")
        })
        .collect()
}

#[test]
fn no_kernel_source_terminates_the_process_on_a_cuda_error() {
    let offenders: Vec<_> = kernel_sources()
        .into_iter()
        .filter(|path| read(path).contains("exit("))
        .collect();

    assert!(
        offenders.is_empty(),
        "these kernel sources end the process on a CUDA failure instead of returning a status the \
         caller can surface: {offenders:?}"
    );
}

#[test]
fn every_kernel_entry_point_returns_a_status() {
    // `flash_cuda_error_string` resolves a status rather than producing one, so it is the single
    // entry point allowed a different return type.
    const RESOLVER: &str = "extern \"C\" const char *flash_cuda_error_string(";

    let offenders: Vec<_> = kernel_sources()
        .into_iter()
        .flat_map(|path| {
            let source = read(&path).replace(RESOLVER, "");
            source
                .match_indices("extern \"C\"")
                .filter(|(index, _)| {
                    let tail = source[*index..]
                        .trim_start_matches("extern \"C\"")
                        .trim_start();
                    // `extern "C" {` opens a block whose members are declared on their own lines.
                    !tail.starts_with('{') && !tail.starts_with("cudaError_t")
                })
                .map(|(index, _)| {
                    let tail = &source[index..];
                    format!("{}: {}", path.display(), &tail[..tail.len().min(60)])
                })
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "these kernel entry points do not return a `cudaError_t`, so a failure inside them cannot \
         reach Rust: {offenders:#?}"
    );
}

#[test]
fn every_launcher_declares_and_checks_a_status() {
    let ffi = read(&crate_root().join("src/ffi.rs"));
    let sources: String = ["src/flash_attention.rs", "src/cache_manager.rs"]
        .iter()
        .map(|relative| read(&crate_root().join(relative)))
        .collect();

    for launcher in LAUNCHERS {
        let declaration = ffi
            .split_once(&format!("fn {launcher}("))
            .unwrap_or_else(|| panic!("src/ffi.rs must declare {launcher}"))
            .1;
        let signature = declaration
            .split_once(';')
            .expect("an extern declaration ends with a semicolon")
            .0;
        assert!(
            signature.contains("-> c_int"),
            "the {launcher} FFI declaration must return the CUDA status, got: {signature}"
        );
        assert!(
            sources.contains(&format!("check_launch(\"{launcher}\"")),
            "every {launcher} launch must have its status checked with check_launch"
        );
    }
}

#[test]
fn every_ffi_argument_label_names_a_real_parameter() {
    let ffi = read(&crate_root().join("src/ffi.rs"));
    let declaration = ffi
        .split_once("pub(crate) fn run_mha(")
        .expect("src/ffi.rs must declare run_mha")
        .1
        .split_once("\n    ) -> c_int;")
        .expect("the run_mha declaration must end with its return type")
        .0;
    let parameters: Vec<&str> = declaration
        .lines()
        .filter_map(|line| line.trim().split_once(':'))
        .map(|(name, _)| name)
        .collect();

    let source = read(&crate_root().join("src/flash_attention.rs"));
    let call_sites: Vec<&str> = source
        .split("ffi::run_mha(")
        .skip(1)
        .map(|tail| {
            tail.split_once("\n            )")
                .expect("a run_mha call ends with its closing parenthesis")
                .0
        })
        .collect();
    assert_eq!(call_sites.len(), 3, "expected one call per attention path");

    let offenders: Vec<&str> = call_sites
        .iter()
        .flat_map(|call| {
            call.match_indices("/* ")
                .filter_map(|(index, _)| call[index + 3..].split_once(" */").map(|(name, _)| name))
        })
        .filter(|label| !parameters.contains(label))
        .collect();

    assert!(
        offenders.is_empty(),
        "these argument labels do not name a run_mha parameter, so they describe the call wrongly: \
         {offenders:?}"
    );
}
