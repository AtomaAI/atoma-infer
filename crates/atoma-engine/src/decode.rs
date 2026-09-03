//! The decode step on the runtime tensor path: the host side that needs no device.
//!
//! A keyed step command is checked against the bucket its key names and laid out as the fixed
//! inputs the step reads, at the widths the captured graphs bake. What is here is pure host
//! arithmetic over the batch layout; the device half sits behind the `cuda` feature.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`batch`] | A keyed batch held to its bucket, and the buckets the tensor path serves |
//! | [`staging`] | One step's inputs written into staging at full width, ready to upload |

pub mod batch;
pub mod staging;
