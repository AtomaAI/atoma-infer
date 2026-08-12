//! NVRTC compilation and raw launches of the throwaway step kernels.
//!
//! Everything is raw (`sys::CUfunction`, `result::launch_kernel`) because the capture stream
//! only exposes its raw handle: `CaptureStream`'s safe surface deliberately has no launch, and
//! cudarc's safe launcher requires the `Arc<CudaStream>` it hides. How much raw plumbing this
//! costs is itself a harness finding for the production executor API.

use std::ffi::{c_int, c_void, CString};

use anyhow::{anyhow, Context, Result};
use atoma_runtime::context::RuntimeContext;
use cudarc::driver::{result, sys};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const KERNEL_SRC: &str = include_str!("kernels.cu");
const THREADS: u32 = 256;

/// Grid size for one thread per element.
fn grid_1d(n: usize) -> u32 {
    u32::try_from(n.div_ceil(THREADS as usize)).expect("grid fits u32")
}

/// The NVRTC arch string for a compute capability, when it is one the harness knows; `None` falls
/// back to NVRTC's baseline PTX, which the driver JITs.
fn arch_for(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        (_, _) => None,
    }
}

/// The compiled step kernels. The module is never unloaded: captured graphs reference its
/// functions for the process lifetime.
pub struct StepKernels {
    fill_random_bf16: sys::CUfunction,
    embedding_gather: sys::CUfunction,
    rmsnorm_bf16: sys::CUfunction,
    rope_qk: sys::CUfunction,
    silu_mul: sys::CUfunction,
    add_bf16: sys::CUfunction,
    argmax_bf16: sys::CUfunction,
    bf16_to_f32_arr: sys::CUfunction,
    f32_to_bf16_arr: sys::CUfunction,
}

impl StepKernels {
    /// Compiles `kernels.cu` for the context's device and loads every kernel.
    pub fn compile_and_load(ctx: &RuntimeContext) -> Result<Self> {
        let (major, minor) = ctx
            .cuda()
            .compute_capability()
            .map_err(|e| anyhow!("querying compute capability: {:?}", e.0))?;
        let opts = CompileOptions {
            arch: arch_for(major, minor),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| anyhow!("NVRTC rejected kernels.cu: {e:?}"))?;
        let image = CString::new(ptx.to_src()).context("PTX contains an interior NUL")?;

        ctx.cuda()
            .bind_to_thread()
            .map_err(|e| anyhow!("binding context: {:?}", e.0))?;
        let module = unsafe { result::module::load_data(image.as_ptr().cast::<c_void>()) }
            .map_err(|e| anyhow!("loading step-kernel PTX module: {:?}", e.0))?;

        let function = |name: &str| -> Result<sys::CUfunction> {
            let c_name = CString::new(name).expect("kernel names have no NUL");
            unsafe { result::module::get_function(module, c_name) }
                .map_err(|e| anyhow!("kernel {name} missing from module: {:?}", e.0))
        };
        Ok(Self {
            fill_random_bf16: function("fill_random_bf16")?,
            embedding_gather: function("embedding_gather")?,
            rmsnorm_bf16: function("rmsnorm_bf16")?,
            rope_qk: function("rope_qk")?,
            silu_mul: function("silu_mul")?,
            add_bf16: function("add_bf16")?,
            argmax_bf16: function("argmax_bf16")?,
            bf16_to_f32_arr: function("bf16_to_f32_arr")?,
            f32_to_bf16_arr: function("f32_to_bf16_arr")?,
        })
    }

    /// Fills `n` bf16 elements at `out` with deterministic uniforms in `[-scale, scale]`.
    ///
    /// # Safety
    /// `out` must address at least `n` bf16 elements on the context's device.
    pub unsafe fn fill_random_bf16(
        &self,
        stream: sys::CUstream,
        out: sys::CUdeviceptr,
        n: usize,
        seed: u64,
        scale: f32,
    ) -> Result<()> {
        let mut out = out;
        let mut n = n as u64;
        let mut seed = seed;
        let mut scale = scale;
        let grid = grid_1d((n as usize).min(1 << 20));
        let mut params = [
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
            (&raw mut seed).cast::<c_void>(),
            (&raw mut scale).cast::<c_void>(),
        ];
        launch(self.fill_random_bf16, grid, stream, &mut params)
    }

    /// Gathers `n_tokens` embedding rows by token id into `out`.
    ///
    /// # Safety
    /// All pointers must be live device addresses of the documented shapes.
    pub unsafe fn embedding_gather(
        &self,
        stream: sys::CUstream,
        table: sys::CUdeviceptr,
        token_ids: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        hidden: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let (mut table, mut token_ids, mut out) = (table, token_ids, out);
        let mut hidden = c_int::try_from(hidden)?;
        let mut n_tokens = c_int::try_from(n_tokens)?;
        let grid = grid_1d((hidden * n_tokens) as usize);
        let mut params = [
            (&raw mut table).cast::<c_void>(),
            (&raw mut token_ids).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut hidden).cast::<c_void>(),
            (&raw mut n_tokens).cast::<c_void>(),
        ];
        launch(self.embedding_gather, grid, stream, &mut params)
    }

    /// RMS-normalizes `n_tokens` rows of width `hidden` from `x` into `out` with gains `gamma`.
    ///
    /// # Safety
    /// All pointers must be live device addresses of the documented shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_bf16(
        &self,
        stream: sys::CUstream,
        x: sys::CUdeviceptr,
        gamma: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        hidden: usize,
        n_tokens: usize,
        eps: f32,
    ) -> Result<()> {
        let (mut x, mut gamma, mut out) = (x, gamma, out);
        let mut hidden = c_int::try_from(hidden)?;
        let mut eps = eps;
        let grid = u32::try_from(n_tokens)?;
        let mut params = [
            (&raw mut x).cast::<c_void>(),
            (&raw mut gamma).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut hidden).cast::<c_void>(),
            (&raw mut eps).cast::<c_void>(),
        ];
        launch(self.rmsnorm_bf16, grid, stream, &mut params)
    }

    /// Applies RoPE in place to the q and k segments of the fused qkv buffer; positions come
    /// from `seqlens_k` minus one.
    ///
    /// # Safety
    /// All pointers must be live device addresses of the documented shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_qk(
        &self,
        stream: sys::CUstream,
        qkv: sys::CUdeviceptr,
        seqlens_k: sys::CUdeviceptr,
        n_tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<()> {
        let (mut qkv, mut seqlens_k) = (qkv, seqlens_k);
        let mut n_tokens_c = c_int::try_from(n_tokens)?;
        let mut num_q_heads_c = c_int::try_from(num_q_heads)?;
        let mut num_kv_heads_c = c_int::try_from(num_kv_heads)?;
        let mut head_dim_c = c_int::try_from(head_dim)?;
        let mut theta = theta;
        let grid = grid_1d(n_tokens * (num_q_heads + num_kv_heads) * (head_dim / 2));
        let mut params = [
            (&raw mut qkv).cast::<c_void>(),
            (&raw mut seqlens_k).cast::<c_void>(),
            (&raw mut n_tokens_c).cast::<c_void>(),
            (&raw mut num_q_heads_c).cast::<c_void>(),
            (&raw mut num_kv_heads_c).cast::<c_void>(),
            (&raw mut head_dim_c).cast::<c_void>(),
            (&raw mut theta).cast::<c_void>(),
        ];
        launch(self.rope_qk, grid, stream, &mut params)
    }

    /// `out = silu(gate) * up` over `n` elements.
    ///
    /// # Safety
    /// All pointers must be live device addresses covering `n` bf16 elements.
    pub unsafe fn silu_mul(
        &self,
        stream: sys::CUstream,
        gate: sys::CUdeviceptr,
        up: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<()> {
        let (mut gate, mut up, mut out) = (gate, up, out);
        let mut n = n as u64;
        let grid = grid_1d(n as usize);
        let mut params = [
            (&raw mut gate).cast::<c_void>(),
            (&raw mut up).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
        ];
        launch(self.silu_mul, grid, stream, &mut params)
    }

    /// `out = a + b` over `n` bf16 elements, accumulated in f32.
    ///
    /// # Safety
    /// All pointers must be live device addresses covering `n` bf16 elements.
    pub unsafe fn add_bf16(
        &self,
        stream: sys::CUstream,
        a: sys::CUdeviceptr,
        b: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<()> {
        let (mut a, mut b, mut out) = (a, b, out);
        let mut n = n as u64;
        let grid = grid_1d(n as usize);
        let mut params = [
            (&raw mut a).cast::<c_void>(),
            (&raw mut b).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
        ];
        launch(self.add_bf16, grid, stream, &mut params)
    }

    /// Writes the argmax index of each sequence's logits row into `out`.
    ///
    /// # Safety
    /// `logits` must address `n_sequences * vocab` bf16 elements and `out` `n_sequences` i32s.
    pub unsafe fn argmax_bf16(
        &self,
        stream: sys::CUstream,
        logits: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        vocab: usize,
        n_sequences: usize,
    ) -> Result<()> {
        let (mut logits, mut out) = (logits, out);
        let mut vocab = c_int::try_from(vocab)?;
        let grid = u32::try_from(n_sequences)?;
        let mut params = [
            (&raw mut logits).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut vocab).cast::<c_void>(),
        ];
        launch(self.argmax_bf16, grid, stream, &mut params)
    }

    /// Widens `n` bf16 elements to f32.
    ///
    /// # Safety
    /// Pointers must be live device addresses covering `n` elements of their type.
    pub unsafe fn bf16_to_f32(
        &self,
        stream: sys::CUstream,
        input: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<()> {
        let (mut input, mut out) = (input, out);
        let mut n = n as u64;
        let grid = grid_1d(n as usize);
        let mut params = [
            (&raw mut input).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
        ];
        launch(self.bf16_to_f32_arr, grid, stream, &mut params)
    }

    /// Narrows `n` f32 elements to bf16 with round-to-nearest-even.
    ///
    /// # Safety
    /// Pointers must be live device addresses covering `n` elements of their type.
    pub unsafe fn f32_to_bf16(
        &self,
        stream: sys::CUstream,
        input: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<()> {
        let (mut input, mut out) = (input, out);
        let mut n = n as u64;
        let grid = grid_1d(n as usize);
        let mut params = [
            (&raw mut input).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
        ];
        launch(self.f32_to_bf16_arr, grid, stream, &mut params)
    }
}

/// One raw 1-D launch with 256-thread blocks.
///
/// # Safety
/// `params` must match the kernel's parameter list exactly.
unsafe fn launch(
    function: sys::CUfunction,
    grid: u32,
    stream: sys::CUstream,
    params: &mut [*mut c_void],
) -> Result<()> {
    unsafe { result::launch_kernel(function, (grid, 1, 1), (THREADS, 1, 1), 0, stream, params) }
        .map_err(|e| anyhow!("kernel launch failed: {:?}", e.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_every_element() {
        assert_eq!(grid_1d(1), 1);
        assert_eq!(grid_1d(256), 1);
        assert_eq!(grid_1d(257), 2);
    }

    #[test]
    fn known_architectures_map_and_unknown_fall_back() {
        assert_eq!(arch_for(9, 0), Some("compute_90"));
        assert_eq!(arch_for(12, 0), None);
    }

    #[test]
    fn kernel_source_declares_every_loaded_symbol() {
        for name in [
            "fill_random_bf16",
            "embedding_gather",
            "rmsnorm_bf16",
            "rope_qk",
            "silu_mul",
            "add_bf16",
            "argmax_bf16",
            "bf16_to_f32_arr",
            "f32_to_bf16_arr",
        ] {
            assert!(
                KERNEL_SRC.contains(&format!("__global__ void {name}(")),
                "kernels.cu lost the {name} kernel"
            );
        }
    }
}
