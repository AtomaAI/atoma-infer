//! The device-facing half of the probe. Everything here type-checks GPU-free under the
//! workspace's `fallback-dynamic-loading` cudarc pin; only execution needs a driver.

pub mod blas;
pub mod harness;
pub mod kernels;
pub mod step;
