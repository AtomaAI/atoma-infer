//! The decode step on the runtime tensor path: the host side that needs no device.
//!
//! A keyed step command is checked against the bucket its key names, laid out as the fixed
//! inputs the step reads at the widths the captured graphs bake, and carried to the device
//! through the descriptor seam. Nothing here needs candle or a compiled kernel, so it builds and
//! tests without a device; the model step itself sits behind the `cuda` feature.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`batch`] | A keyed batch held to its bucket, and the buckets the tensor path serves |
//! | [`staging`] | One step's inputs written into staging at full width, ready to upload |
//! | [`inputs`] | Pinned staging and fixed device buffers per input; the upload and fence descriptors |

pub mod batch;
pub mod inputs;
pub mod staging;
