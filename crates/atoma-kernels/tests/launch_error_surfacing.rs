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

/// The FFI entry points that do not launch a kernel, and so have no launch status to check.
///
/// Every other declaration in the `extern` block is treated as a launcher, so adding one without
/// checking its status fails [`every_launch_site_has_its_status_checked`].
const NON_LAUNCHING_FFI: [&str; 2] = ["flash_last_error", "flash_cuda_error_string"];

/// The launchers we own, which return their status directly.
const STATUS_RETURNING_LAUNCHERS: [&str; 9] = [
    "copy_blocks_cache",
    "reshape_and_cache_flash_cache",
    "decode_embedding_gather_bf16",
    "decode_rmsnorm_bf16",
    "decode_rope_bf16",
    "decode_silu_mul_bf16",
    "decode_add_bf16",
    "sampler_sample_f32",
    "sampler_gather_u32",
];

/// The modules that call into the FFI. Every launch site lives in one of these.
const CALLER_SOURCES: [&str; 5] = [
    "src/flash_attention.rs",
    "src/cache_manager.rs",
    "src/paged_decode.rs",
    "src/decode_ops.rs",
    "src/sampler.rs",
];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// The body of the `extern "C"` block in `src/ffi.rs`.
fn extern_block(ffi: &str) -> &str {
    ffi.split_once("extern \"C\" {")
        .expect("src/ffi.rs must declare an extern block")
        .1
        .split_once("\n}")
        .expect("the extern block must be closed")
        .0
}

/// The FFI entry points that launch a kernel, read from the `extern` block rather than listed here
/// so that a newly declared launcher is covered without anyone remembering to add it.
fn declared_launchers(ffi: &str) -> Vec<&str> {
    extern_block(ffi)
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_prefix("pub(crate) fn ")
                .or_else(|| line.strip_prefix("fn "))
        })
        .filter_map(|declaration| declaration.split_once('(').map(|(name, _)| name))
        .filter(|name| !NON_LAUNCHING_FFI.contains(name))
        .collect()
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
fn every_launch_site_has_its_status_checked() {
    let ffi = read(&crate_root().join("src/ffi.rs"));
    let launchers = declared_launchers(&ffi);
    assert!(
        !launchers.is_empty(),
        "src/ffi.rs declares no launchers, so this test would hold vacuously"
    );

    let sources: String = CALLER_SOURCES
        .iter()
        .map(|relative| read(&crate_root().join(relative)))
        .collect();

    for launcher in launchers {
        // Counted rather than merely found: one checked launch must not vouch for the others.
        let launches = sources.matches(&format!("ffi::{launcher}(")).count();
        let checks = sources
            .matches(&format!("check_launch(\"{launcher}\""))
            .count();
        assert!(
            launches > 0,
            "{launcher} is declared in src/ffi.rs but never called as `ffi::{launcher}(` in \
             {CALLER_SOURCES:?}; if it moved or is now called through an import, this test can no \
             longer see its launch sites"
        );
        assert_eq!(
            launches, checks,
            "{launcher} is launched {launches} times but only {checks} of those launches have \
             their status checked with check_launch"
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
