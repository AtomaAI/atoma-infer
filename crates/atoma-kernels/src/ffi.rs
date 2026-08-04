use crate::error::KernelError;
use core::ffi::{c_char, c_int, c_void};
use std::ffi::CStr;

extern "C" {
    /// Returns the `cudaError_t` of the launch, or of the setup call that failed before it.
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        is_seqlens_k_cumulative: bool,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        num_splits: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,
        scale_softmax_log2: f32,

        block_table: *const c_int,
        block_table_batch_stride: u32,
        page_block_size: c_int,

        seqused_k: *const c_int,
        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,

        window_size_left: c_int,
        window_size_right: c_int,
        softcap: f32,
        unpadded_lse: bool,
        force_split_kernel: bool,

        softmax_lseaccum_ptr: *const c_void,
        oaccum_ptr: *const c_void,

        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn copy_blocks_cache(
        cache: *mut c_void,
        block_mapping: *const c_void,
        num_pairs: i64,
        numel_per_block: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn reshape_and_cache_flash_cache(
        source: *const c_void,
        cache: *mut c_void,
        slot_mapping: *const i64,
        block_stride: i64,
        num_tokens: i64,
        num_heads: i64,
        head_size: i64,
        block_size: i64,
        source_stride: i64,
        dtype: u32,
        stream: *mut c_void,
    ) -> c_int;

    /// `cudaGetErrorString` for a `cudaError_t` value, as a static NUL-terminated string.
    fn flash_cuda_error_string(code: c_int) -> *const c_char;
}

/// Turns the status returned by an FFI launcher into a typed error.
///
/// A launch is asynchronous, so a success here means the kernel was accepted by the driver, not
/// that it has run; faults raised during execution surface on a later synchronization.
///
/// # Arguments
///
/// * `kernel` - Name of the FFI entry point, used to identify the failure.
/// * `status` - The `cudaError_t` value the entry point returned.
pub(crate) fn check_launch(kernel: &'static str, status: c_int) -> Result<(), KernelError> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: `cudaGetErrorString` returns a pointer to a static string for every input, including
    // unrecognized error codes.
    let message = unsafe { CStr::from_ptr(flash_cuda_error_string(status)) }
        .to_string_lossy()
        .into_owned();
    Err(KernelError::LaunchFailed {
        kernel,
        code: status,
        message,
    })
}
