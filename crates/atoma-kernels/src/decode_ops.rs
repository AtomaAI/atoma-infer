//! The decode step's own kernels: embedding gather, RMSNorm, rotary embedding, SiLU-gated
//! multiply and the residual add, over bf16 rows at raw device addresses.
//!
//! The sources are in-house (`kernels/decode_ops.cu`), compiled by nvcc under the `cuda` feature
//! into a library of their own, apart from the vendored flash-attention build and its fast-math
//! flags. Each launcher returns its launch status and takes the caller's stream; the wrappers
//! here turn the status into [`KernelError`]. Without the feature the same functions return
//! [`KernelError::NotCompiled`], so the crates built on this one compile and test without a
//! toolkit.
//!
//! The rotary embedding reads cos and sin from tables indexed by position rather than computing
//! them: the frequencies a model scales (Llama 3) are the caller's to compute on the host, once.

use core::ffi::c_void;

use crate::error::KernelError;

/// Gathers one embedding row per token: `out[t] = table[token_ids[t]]`.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingGatherCall {
    /// bf16 `[vocab, hidden]`.
    pub table: u64,
    /// u32 `[n_tokens]`.
    pub token_ids: u64,
    /// bf16 `[n_tokens, hidden]`.
    pub out: u64,
    pub hidden: usize,
    pub n_tokens: usize,
    pub stream: *mut c_void,
}

/// Normalizes each token row by its root mean square and scales it by the gain.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormCall {
    /// bf16 `[n_tokens, hidden]`.
    pub x: u64,
    /// bf16 `[hidden]`.
    pub gain: u64,
    /// bf16 `[n_tokens, hidden]`.
    pub out: u64,
    pub hidden: usize,
    pub n_tokens: usize,
    pub eps: f32,
    pub stream: *mut c_void,
}

/// Rotates the first `rot_heads` heads of each fused qkv row in place, by the token's position.
#[derive(Debug, Clone, Copy)]
pub struct RopeCall {
    /// bf16 `[n_tokens, row_width]`: the q heads first, then the k heads, then the v heads.
    pub qkv: u64,
    /// i32 `[n_tokens]`: each token's position, an index into the tables.
    pub positions: u64,
    /// f32 `[max_position, head_dim / 2]`.
    pub cos_table: u64,
    /// f32 `[max_position, head_dim / 2]`.
    pub sin_table: u64,
    pub n_tokens: usize,
    /// The q heads plus the k heads.
    pub rot_heads: usize,
    pub head_dim: usize,
    /// Elements per fused row.
    pub row_width: usize,
    pub stream: *mut c_void,
}

/// `out = silu(gate) * up` over `len` elements.
#[derive(Debug, Clone, Copy)]
pub struct SiluMulCall {
    pub gate: u64,
    pub up: u64,
    pub out: u64,
    pub len: usize,
    pub stream: *mut c_void,
}

/// `out = lhs + rhs` over `len` elements, summed in f32.
#[derive(Debug, Clone, Copy)]
pub struct AddCall {
    pub lhs: u64,
    pub rhs: u64,
    pub out: u64,
    pub len: usize,
    pub stream: *mut c_void,
}

#[cfg(feature = "cuda")]
mod real {
    use core::ffi::c_void;

    use super::{AddCall, EmbeddingGatherCall, RmsNormCall, RopeCall, SiluMulCall};
    use crate::error::KernelError;
    use crate::ffi;

    /// A count that must fit the FFI's 64-bit parameter.
    fn count(argument: &'static str, value: usize) -> Result<i64, KernelError> {
        i64::try_from(value).map_err(|_| KernelError::ArgumentOverflow { argument, value })
    }

    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; every token id must index the table.
    pub unsafe fn embedding_gather(call: &EmbeddingGatherCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::decode_embedding_gather_bf16(
                call.table as *const c_void,
                call.token_ids as *const c_void,
                call.out as *mut c_void,
                count("hidden", call.hidden)?,
                count("n_tokens", call.n_tokens)?,
                call.stream,
            )
        };
        ffi::check_launch("decode_embedding_gather_bf16", status)
    }

    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape.
    pub unsafe fn rmsnorm(call: &RmsNormCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::decode_rmsnorm_bf16(
                call.x as *const c_void,
                call.gain as *const c_void,
                call.out as *mut c_void,
                count("hidden", call.hidden)?,
                count("n_tokens", call.n_tokens)?,
                call.eps,
                call.stream,
            )
        };
        ffi::check_launch("decode_rmsnorm_bf16", status)
    }

    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; every position must index the tables.
    pub unsafe fn rope(call: &RopeCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::decode_rope_bf16(
                call.qkv as *mut c_void,
                call.positions as *const c_void,
                call.cos_table as *const c_void,
                call.sin_table as *const c_void,
                count("n_tokens", call.n_tokens)?,
                count("rot_heads", call.rot_heads)?,
                count("head_dim", call.head_dim)?,
                count("row_width", call.row_width)?,
                call.stream,
            )
        };
        ffi::check_launch("decode_rope_bf16", status)
    }

    /// # Safety
    /// Every address in `call` must be live on the stream's device and cover `len` elements.
    pub unsafe fn silu_mul(call: &SiluMulCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::decode_silu_mul_bf16(
                call.gate as *const c_void,
                call.up as *const c_void,
                call.out as *mut c_void,
                count("len", call.len)?,
                call.stream,
            )
        };
        ffi::check_launch("decode_silu_mul_bf16", status)
    }

    /// # Safety
    /// Every address in `call` must be live on the stream's device and cover `len` elements.
    pub unsafe fn add(call: &AddCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::decode_add_bf16(
                call.lhs as *const c_void,
                call.rhs as *const c_void,
                call.out as *mut c_void,
                count("len", call.len)?,
                call.stream,
            )
        };
        ffi::check_launch("decode_add_bf16", status)
    }
}

#[cfg(feature = "cuda")]
pub use real::{add, embedding_gather, rmsnorm, rope, silu_mul};

/// Named refusal: this build carries no kernels.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn embedding_gather(_call: &EmbeddingGatherCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "decode_embedding_gather_bf16",
    })
}

/// Named refusal: this build carries no kernels.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn rmsnorm(_call: &RmsNormCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "decode_rmsnorm_bf16",
    })
}

/// Named refusal: this build carries no kernels.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn rope(_call: &RopeCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "decode_rope_bf16",
    })
}

/// Named refusal: this build carries no kernels.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn silu_mul(_call: &SiluMulCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "decode_silu_mul_bf16",
    })
}

/// Named refusal: this build carries no kernels.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn add(_call: &AddCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "decode_add_bf16",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel sources, so the launcher list and the sources are held to the same names.
    const KERNEL_SOURCE: &str = include_str!("../kernels/decode_ops.cu");

    /// Every launcher the Rust side declares, as the sources must export it.
    const LAUNCHERS: [&str; 5] = [
        "decode_embedding_gather_bf16",
        "decode_rmsnorm_bf16",
        "decode_rope_bf16",
        "decode_silu_mul_bf16",
        "decode_add_bf16",
    ];

    #[test]
    fn the_sources_export_every_launcher_with_a_status_and_a_stream() {
        for launcher in LAUNCHERS {
            let declaration = format!("extern \"C\" cudaError_t {launcher}(");
            let Some((_, rest)) = KERNEL_SOURCE.split_once(&declaration) else {
                panic!("decode_ops.cu does not export {launcher}");
            };
            let parameters = rest.split_once(')').expect("a parameter list closes").0;
            assert!(
                parameters.ends_with("cudaStream_t stream"),
                "{launcher} must take the caller's stream last, got: {parameters}"
            );
        }
    }

    #[test]
    fn no_kernel_launch_ignores_its_status() {
        let launches = KERNEL_SOURCE.matches("<<<").count();
        let checks = KERNEL_SOURCE.matches("return cudaGetLastError();").count();
        assert_eq!(launches, LAUNCHERS.len());
        assert_eq!(checks, LAUNCHERS.len());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_build_without_kernels_refuses_each_by_name() {
        let stream = core::ptr::null_mut();
        // SAFETY: the stubs dereference nothing.
        let refusals = unsafe {
            [
                embedding_gather(&EmbeddingGatherCall {
                    table: 0,
                    token_ids: 0,
                    out: 0,
                    hidden: 8,
                    n_tokens: 1,
                    stream,
                })
                .unwrap_err(),
                rmsnorm(&RmsNormCall {
                    x: 0,
                    gain: 0,
                    out: 0,
                    hidden: 8,
                    n_tokens: 1,
                    eps: 1e-5,
                    stream,
                })
                .unwrap_err(),
                rope(&RopeCall {
                    qkv: 0,
                    positions: 0,
                    cos_table: 0,
                    sin_table: 0,
                    n_tokens: 1,
                    rot_heads: 2,
                    head_dim: 4,
                    row_width: 12,
                    stream,
                })
                .unwrap_err(),
                silu_mul(&SiluMulCall {
                    gate: 0,
                    up: 0,
                    out: 0,
                    len: 8,
                    stream,
                })
                .unwrap_err(),
                add(&AddCall {
                    lhs: 0,
                    rhs: 0,
                    out: 0,
                    len: 8,
                    stream,
                })
                .unwrap_err(),
            ]
        };
        for (refusal, launcher) in refusals.iter().zip(LAUNCHERS) {
            assert_eq!(refusal, &KernelError::NotCompiled { kernel: launcher });
        }
    }
}
