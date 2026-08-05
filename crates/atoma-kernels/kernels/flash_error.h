// Records CUDA failures raised inside the vendored flash-attention launchers so Rust can surface
// them, instead of the process ending on the first one.
//
// This file is ours, not part of flash-attention. It exists to keep the patch against the vendored
// sources down to two lines in `flash_fwd_launch_template.h`: an include, and the process-ending
// call becoming `flash_record_error`. Threading a status return through the dispatch templates
// instead would touch every one of the 66 instantiation files, which
// `docs/plan/audit/repo-kernels.md` has already marked for replacement.
//
// `cudaGetLastError` cannot carry the failure on its own: the launchers call it themselves and
// consume the state before Rust could read it.

#pragma once

#include <cuda_runtime.h>

// The first failure recorded since `flash_last_error` was last called, per thread because launches
// on different threads target different streams.
inline cudaError_t &flash_error_slot() {
    static thread_local cudaError_t error = cudaSuccess;
    return error;
}

// Records `code` unless an earlier failure is still pending, so the first cause survives.
inline void flash_record_error(cudaError_t code) {
    if (code != cudaSuccess && flash_error_slot() == cudaSuccess) {
        flash_error_slot() = code;
    }
}

// Discards any pending failure. Entry points call this before dispatching so what `flash_last_error`
// reports belongs to that call alone.
inline void flash_clear_error() { flash_error_slot() = cudaSuccess; }
