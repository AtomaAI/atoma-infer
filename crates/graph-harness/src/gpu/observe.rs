//! Reading the device back after work: synchronous D2H reads for the identity comparison, and
//! the free-memory query behind the graph-overhead and soak-delta measurements.

use anyhow::{anyhow, Result};
use cudarc::driver::result;

/// Synchronous D2H read of `buf.len()` bytes from `src`. Call only after the stream that wrote
/// `src` has been synchronized.
pub fn read_back(src: u64, buf: &mut [u8]) -> Result<()> {
    // SAFETY: `src` is a live device address the caller sized `buf` for, and the copy is
    // synchronous, so `buf` outlives it.
    unsafe { result::memcpy_dtoh_sync(buf, src) }.map_err(|e| anyhow!("D2H read: {:?}", e.0))
}

/// The device's free memory in bytes.
pub fn free_memory() -> Result<i64> {
    let (free, _total) = result::mem_get_info().map_err(|e| anyhow!("mem_get_info: {:?}", e.0))?;
    Ok(free as i64)
}
