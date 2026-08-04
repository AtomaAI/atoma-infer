// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the ATOMA_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
#[cfg(feature = "cuda")]
use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::process::Command;

#[cfg(feature = "cuda")]
const KERNEL_FILES: [&str; 66] = [
    "kernels/cache_manager.cu",
    "kernels/flash_api.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_sm80.cu",
];

#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    println!("cargo:rerun-if-changed=kernels/rotary.h");
    println!("cargo:rerun-if-changed=kernels/alibi.h");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("ATOMA_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => PathBuf::from(build_dir)
            .canonicalize()
            .context("Failed to canonicalize build directory")?,
    };
    println!("cargo:warning=Build directory: {:?}", build_dir.display());

    compile_cuda_files(&build_dir)?;

    // Link libraries
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=msvcprt");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}

/// The CUTLASS header the flash-attention kernels include first.
#[cfg(feature = "cuda")]
const CUTLASS_SENTINEL_HEADER: &str = "cutlass/include/cutlass/cutlass.h";

/// Fails with checkout instructions when the CUTLASS submodule is empty.
///
/// Without this, nvcc reports a missing-header error for each of the 66 kernel translation units
/// and never names the submodule.
#[cfg(feature = "cuda")]
fn check_cutlass_checkout() -> Result<()> {
    if Path::new(CUTLASS_SENTINEL_HEADER).is_file() {
        return Ok(());
    }
    anyhow::bail!(
        "CUTLASS headers are missing ({CUTLASS_SENTINEL_HEADER} not found). CUTLASS is a Git \
         submodule that a plain `git clone` leaves empty; check it out with \
         `git submodule update --init --depth 1 crates/atoma-kernels/cutlass`."
    )
}

/// Fails with an actionable message when the toolchain the kernel build needs is missing.
///
/// `bindgen_cuda` resolves the target architecture from `CUDA_COMPUTE_CAP` or `nvidia-smi`, then
/// panics with `Failed to get compute_cap` when neither answers, naming none of the causes.
#[cfg(feature = "cuda")]
fn check_cuda_toolchain() -> Result<()> {
    if Command::new("nvcc").arg("--version").output().is_err() {
        anyhow::bail!(
            "`nvcc` was not found on PATH; the CUDA toolkit is required to build the `cuda` feature."
        )
    }
    if std::env::var_os("CUDA_COMPUTE_CAP").is_some() {
        return Ok(());
    }
    let compute_cap = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv"])
        .output();
    match compute_cap {
        Ok(output) if output.status.success() => Ok(()),
        Ok(_) | Err(_) => anyhow::bail!(
            "Could not determine the target compute capability: `nvidia-smi` did not answer and \
             CUDA_COMPUTE_CAP is unset. Set CUDA_COMPUTE_CAP (for example `CUDA_COMPUTE_CAP=80`) \
             to build for a specific architecture without a GPU present."
        ),
    }
}

#[cfg(feature = "cuda")]
fn compile_cuda_files(build_dir: &Path) -> Result<()> {
    check_cutlass_checkout()?;
    check_cuda_toolchain()?;

    let kernels: Vec<_> = KERNEL_FILES.iter().map(|&s| s.to_string()).collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.to_path_buf())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Icutlass/include")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-w");

    println!("cargo:info={builder:?}");

    let out_file = if cfg!(target_os = "windows") {
        build_dir.join("flashattention.lib")
    } else {
        build_dir.join("libflashattention.a")
    };

    builder.build_lib(&out_file);

    Ok(())
}
