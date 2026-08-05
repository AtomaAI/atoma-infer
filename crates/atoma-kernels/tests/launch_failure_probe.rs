//! Manual probe for acceptance criterion 5 of #156: a launch that the driver actually rejects must
//! reach Rust as `KernelError::LaunchFailed` carrying the driver's message, and the process must
//! survive to observe it.
//!
//! No ordinary test reaches this path, because every other test launches a well-formed grid. This
//! one oversubscribes `gridDim.y`: `copy_blocks_cache` launches `dim3 grid(1, num_pairs)`, and the
//! hardware caps `gridDim.y` at 65535, so a mapping with more pairs than that is rejected by the
//! driver at launch time rather than by any check of ours.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};

const NUM_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = 64;
const NUM_HEADS: usize = 2;
const HEAD_SIZE: usize = 8;

/// One past the hardware limit on `gridDim.y`.
const PAIRS_OVER_GRID_LIMIT: usize = 65_536;

#[test]
fn a_launch_the_driver_rejects_surfaces_as_an_error_and_the_process_survives() {
    let device = Device::new_cuda(0).unwrap();

    let mut key_cache = Tensor::zeros(
        &[NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE],
        DType::F16,
        &device,
    )
    .unwrap();
    let mut value_cache = Tensor::zeros(
        &[NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE],
        DType::F16,
        &device,
    )
    .unwrap();

    // Every pair copies block 0 onto itself, so the mapping is in bounds. The only thing wrong with
    // this launch is the size of the grid it asks for.
    let block_mapping = Tensor::zeros(&[PAIRS_OVER_GRID_LIMIT, 2], DType::I64, &device).unwrap();

    let error = atoma_kernels::copy_blocks(&[&mut key_cache], &[&mut value_cache], block_mapping)
        .expect_err("a grid of 65536 in y must be rejected by the driver");

    let rendered = error.to_string();
    println!("observed error: {rendered}");

    assert!(
        rendered.contains("copy_blocks_cache"),
        "the error must name the FFI entry point that failed, got: {rendered}"
    );
    assert!(
        rendered.contains("failed with CUDA error"),
        "the error must carry the driver's code, got: {rendered}"
    );
    assert!(
        rendered.contains("invalid configuration argument"),
        "the error must carry the driver's own message, got: {rendered}"
    );

    // Reaching this line at all is the point: the process was not terminated by the failing launch.
    println!("process survived the failing launch");
}
