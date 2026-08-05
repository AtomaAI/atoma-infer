//! Guards the launch-error invariant: a failing CUDA call reaches Rust as a typed error, and never
//! terminates the process.
//!
//! Like `stream_explicitness.rs`, these checks read the sources rather than running kernels, so
//! they hold in CPU CI where no CUDA toolkit or GPU exists. On a rig the invariant is only
//! observable when a launch actually fails, which no ordinary test run reaches.
//!
//! Two mechanisms carry a failure, because the sources have two owners. `cache_manager.cu` is ours
//! and returns a status directly. The flash-attention dispatch is vendored, so its launchers record
//! into `flash_error.h` and Rust reads the result through `flash_last_error`.

use std::path::{Path, PathBuf};

/// The FFI entry points that launch a kernel, with the label passed to `check_launch`.
const LAUNCHERS: [&str; 3] = [
    "run_mha",
    "copy_blocks_cache",
    "reshape_and_cache_flash_cache",
];

/// The launchers we own, which return their status directly.
const STATUS_RETURNING_LAUNCHERS: [&str; 2] =
    ["copy_blocks_cache", "reshape_and_cache_flash_cache"];

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
        "these kernel sources end the process on a CUDA failure instead of leaving it for the \
         caller to surface: {offenders:?}"
    );
}

#[test]
fn the_vendored_dispatch_records_failures_for_rust_to_read() {
    let template = read(&crate_root().join("kernels/flash_fwd_launch_template.h"));
    assert!(
        template.contains("flash_record_error("),
        "the CUDA_CHECK the vendored launchers call must record the failure, or a failing launch \
         is only printed and then forgotten"
    );

    let api = read(&crate_root().join("kernels/flash_api.cu"));
    assert!(
        api.contains("flash_clear_error()"),
        "run_mha must clear the recorded failure on entry, or it reports an earlier call's error"
    );
    assert!(
        api.contains(r#"extern "C" int flash_last_error()"#),
        "flash_api.cu must export flash_last_error so Rust can read the recorded failure"
    );
}

#[test]
fn every_launcher_has_its_status_checked() {
    let ffi = read(&crate_root().join("src/ffi.rs"));
    let sources: String = ["src/flash_attention.rs", "src/cache_manager.rs"]
        .iter()
        .map(|relative| read(&crate_root().join(relative)))
        .collect();

    for launcher in LAUNCHERS {
        assert!(
            sources.contains(&format!("check_launch(\"{launcher}\"")),
            "every {launcher} launch must have its status checked with check_launch"
        );
    }

    for launcher in STATUS_RETURNING_LAUNCHERS {
        let signature = ffi
            .split_once(&format!("fn {launcher}("))
            .unwrap_or_else(|| panic!("src/ffi.rs must declare {launcher}"))
            .1
            .split_once(';')
            .expect("an extern declaration ends with a semicolon")
            .0;
        assert!(
            signature.contains("-> c_int"),
            "the {launcher} FFI declaration must return the CUDA status, got: {signature}"
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
        .split_once("\n    );")
        .expect("the run_mha declaration must end with a semicolon")
        .0;
    let parameters: Vec<&str> = declaration
        .lines()
        .filter_map(|line| line.trim().split_once(':'))
        .map(|(name, _)| name)
        .collect();

    // Argument labels are single identifiers; the other block comments in the file are prose.
    let source = read(&crate_root().join("src/flash_attention.rs"));
    let offenders: Vec<&str> = source
        .match_indices("/* ")
        .filter_map(|(index, _)| source[index + 3..].split_once(" */").map(|(name, _)| name))
        .filter(|label| {
            label
                .chars()
                .all(|character| character.is_ascii_lowercase() || character == '_')
        })
        .filter(|label| !parameters.contains(label))
        .collect();

    assert!(
        offenders.is_empty(),
        "these argument labels do not name a run_mha parameter, so they describe the call wrongly: \
         {offenders:?}"
    );
}
