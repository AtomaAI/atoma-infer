//! Holding an operand to what a kernel or a table expects.
//!
//! Every launch and every table in this crate reads its counts off tensor views and trusts them
//! afterwards, so what a view must be — the dtype the kernel reads, the rank it indexes, one
//! unbroken row-major buffer, the widths the model's dimensions fix — is checked here, once, and
//! refused by the operand's name with both numbers. The checks take layouts, so the GEMM shapes,
//! which are derived from layouts, use the same ones as the calls assembled from tensors.

use std::fmt;

use atoma_runtime::tensor::{Dtype, Layout, MAX_RANK};
use thiserror::Error;

/// Which tensor a refusal is about: one the model has, or one of a layer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub what: &'static str,
    pub layer: Option<usize>,
}

impl Operand {
    /// A tensor the model has one of.
    #[must_use]
    pub const fn model(what: &'static str) -> Self {
        Self { what, layer: None }
    }

    /// A tensor each layer has one of.
    #[must_use]
    pub const fn layer(layer: usize, what: &'static str) -> Self {
        Self {
            what,
            layer: Some(layer),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layer {
            Some(layer) => write!(f, "layer {layer}'s {}", self.what),
            None => f.write_str(self.what),
        }
    }
}

/// A shape, for refusals that show both the one held and the one needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: usize,
}

impl Shape {
    /// The shape `dims`.
    ///
    /// # Panics
    ///
    /// Panics when `dims` has more than [`MAX_RANK`] dimensions: no layout holds such a shape, so
    /// an expected shape that long is a mistake at the call site, not a mismatch to report.
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        let rank = dims.len();
        assert!(
            rank <= MAX_RANK,
            "an expected shape of rank {rank} exceeds the {MAX_RANK} dimensions a layout holds"
        );
        let mut shape = Self {
            dims: [0; MAX_RANK],
            rank,
        };
        shape.dims[..rank].copy_from_slice(dims);
        shape
    }

    /// The shape of `layout`.
    #[must_use]
    pub fn of(layout: &Layout) -> Self {
        Self::new(layout.dims())
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

/// An operand that is not what was expected of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OperandError {
    #[error("{operand} is {dtype:?}, not {expected:?}")]
    Dtype {
        operand: Operand,
        dtype: Dtype,
        expected: Dtype,
    },
    #[error("{operand} is rank {rank}, not {expected}")]
    Rank {
        operand: Operand,
        rank: usize,
        expected: usize,
    },
    #[error("{operand} has strides {strides:?}; it must be one contiguous buffer")]
    NotContiguous {
        operand: Operand,
        strides: [usize; MAX_RANK],
    },
    #[error("{operand} has inner stride {stride}; its last dimension must be contiguous")]
    InnerStride { operand: Operand, stride: usize },
    #[error("{operand} is {shape}, not {expected}")]
    Shape {
        operand: Operand,
        shape: Shape,
        expected: Shape,
    },
    #[error("{operand} holds {len} elements, not {expected}")]
    Length {
        operand: Operand,
        len: usize,
        expected: usize,
    },
    #[error("{operand} holds {len} rows; {needed} are read")]
    TooShort {
        operand: Operand,
        len: usize,
        needed: usize,
    },
}

/// Holds `layout` to `dtype`.
///
/// # Errors
///
/// Returns [`OperandError::Dtype`] otherwise.
pub fn dtype(operand: Operand, layout: &Layout, expected: Dtype) -> Result<(), OperandError> {
    if layout.dtype() == expected {
        return Ok(());
    }
    Err(OperandError::Dtype {
        operand,
        dtype: layout.dtype(),
        expected,
    })
}

/// Holds `layout` to `rank`.
///
/// # Errors
///
/// Returns [`OperandError::Rank`] otherwise.
pub fn rank(operand: Operand, layout: &Layout, expected: usize) -> Result<(), OperandError> {
    if layout.rank() == expected {
        return Ok(());
    }
    Err(OperandError::Rank {
        operand,
        rank: layout.rank(),
        expected,
    })
}

/// Holds `layout` to being one unbroken row-major buffer.
///
/// # Errors
///
/// Returns [`OperandError::NotContiguous`] with the strides otherwise.
pub fn contiguous(operand: Operand, layout: &Layout) -> Result<(), OperandError> {
    if layout.is_contiguous() {
        return Ok(());
    }
    let mut strides = [0; MAX_RANK];
    for (slot, &stride) in strides.iter_mut().zip(layout.strides()) {
        *slot = stride;
    }
    Err(OperandError::NotContiguous { operand, strides })
}

/// Holds `layout`'s last dimension to being contiguous: what a matrix multiplication needs of
/// an operand whose rows may sit further apart.
///
/// # Errors
///
/// Returns [`OperandError::InnerStride`] otherwise.
pub fn inner_contiguous(operand: Operand, layout: &Layout) -> Result<(), OperandError> {
    let Some(&stride) = layout.strides().last() else {
        return Ok(());
    };
    if stride == 1 {
        return Ok(());
    }
    Err(OperandError::InnerStride { operand, stride })
}

/// Holds `layout` to exactly `dims`.
///
/// # Errors
///
/// Returns [`OperandError::Shape`] with both shapes otherwise.
pub fn shape(operand: Operand, layout: &Layout, dims: &[usize]) -> Result<(), OperandError> {
    if layout.dims() == dims {
        return Ok(());
    }
    Err(OperandError::Shape {
        operand,
        shape: Shape::of(layout),
        expected: Shape::new(dims),
    })
}

/// A contiguous `[rows, columns]` operand of `dtype`, both dimensions fixed.
///
/// # Errors
///
/// Returns [`OperandError`] naming the first of dtype, rank, contiguity and shape that fails.
pub fn matrix(
    operand: Operand,
    layout: &Layout,
    expected: Dtype,
    rows: usize,
    columns: usize,
) -> Result<(), OperandError> {
    dtype(operand, layout, expected)?;
    rank(operand, layout, 2)?;
    contiguous(operand, layout)?;
    shape(operand, layout, &[rows, columns])
}

/// A contiguous `[rows, columns]` operand of `dtype` whose row count is the batch's to say;
/// returns it.
///
/// # Errors
///
/// Returns [`OperandError`] naming the first of dtype, rank, contiguity and width that fails.
pub fn rows(
    operand: Operand,
    layout: &Layout,
    expected: Dtype,
    columns: usize,
) -> Result<usize, OperandError> {
    dtype(operand, layout, expected)?;
    rank(operand, layout, 2)?;
    contiguous(operand, layout)?;
    let held = layout.dim(0);
    shape(operand, layout, &[held, columns])?;
    Ok(held)
}

/// A contiguous vector of `dtype`; returns its length.
///
/// # Errors
///
/// Returns [`OperandError`] naming the first of dtype, rank and contiguity that fails.
pub fn vector_len(
    operand: Operand,
    layout: &Layout,
    expected: Dtype,
) -> Result<usize, OperandError> {
    dtype(operand, layout, expected)?;
    rank(operand, layout, 1)?;
    contiguous(operand, layout)?;
    Ok(layout.dim(0))
}

/// A contiguous vector of `dtype` holding `len` elements.
///
/// # Errors
///
/// Returns [`OperandError`] naming the first of dtype, rank, contiguity and length that fails.
pub fn vector(
    operand: Operand,
    layout: &Layout,
    expected: Dtype,
    len: usize,
) -> Result<(), OperandError> {
    let held = vector_len(operand, layout, expected)?;
    if held == len {
        return Ok(());
    }
    Err(OperandError::Length {
        operand,
        len: held,
        expected: len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(dims: &[usize]) -> Layout {
        Layout::contiguous(dims, Dtype::Bf16).unwrap()
    }

    #[test]
    #[should_panic(expected = "rank 5 exceeds the 4 dimensions")]
    fn an_expected_shape_longer_than_a_layout_holds_is_a_mistake_not_a_mismatch() {
        let _ = Shape::new(&[1, 2, 3, 4, 5]);
    }

    #[test]
    fn an_operand_names_itself_with_its_layer_when_it_has_one() {
        assert_eq!(Operand::model("the gain").to_string(), "the gain");
        assert_eq!(
            Operand::layer(3, "key projection").to_string(),
            "layer 3's key projection"
        );
    }

    #[test]
    fn a_matrix_is_refused_by_the_first_thing_that_fails_in_order() {
        let operand = Operand::model("x");
        assert_eq!(
            matrix(
                operand,
                &Layout::contiguous(&[2, 4], Dtype::F32).unwrap(),
                Dtype::Bf16,
                2,
                4
            )
            .unwrap_err(),
            OperandError::Dtype {
                operand,
                dtype: Dtype::F32,
                expected: Dtype::Bf16
            }
        );
        assert_eq!(
            matrix(operand, &bf16(&[8]), Dtype::Bf16, 2, 4).unwrap_err(),
            OperandError::Rank {
                operand,
                rank: 1,
                expected: 2
            }
        );
        let gapped = Layout::strided(&[2, 4], &[8, 1], Dtype::Bf16).unwrap();
        assert_eq!(
            matrix(operand, &gapped, Dtype::Bf16, 2, 4).unwrap_err(),
            OperandError::NotContiguous {
                operand,
                strides: [8, 1, 0, 0]
            }
        );
        assert_eq!(
            matrix(operand, &bf16(&[2, 5]), Dtype::Bf16, 2, 4).unwrap_err(),
            OperandError::Shape {
                operand,
                shape: Shape::new(&[2, 5]),
                expected: Shape::new(&[2, 4])
            }
        );
        assert_eq!(matrix(operand, &bf16(&[2, 4]), Dtype::Bf16, 2, 4), Ok(()));
    }

    #[test]
    fn rows_are_free_and_vectors_are_held_to_their_length() {
        let operand = Operand::model("x");
        assert_eq!(rows(operand, &bf16(&[7, 4]), Dtype::Bf16, 4), Ok(7));
        assert_eq!(
            rows(operand, &bf16(&[7, 3]), Dtype::Bf16, 4).unwrap_err(),
            OperandError::Shape {
                operand,
                shape: Shape::new(&[7, 3]),
                expected: Shape::new(&[7, 4])
            }
        );
        assert_eq!(vector_len(operand, &bf16(&[9]), Dtype::Bf16), Ok(9));
        assert_eq!(
            vector(operand, &bf16(&[9]), Dtype::Bf16, 8).unwrap_err(),
            OperandError::Length {
                operand,
                len: 9,
                expected: 8
            }
        );
    }

    #[test]
    fn a_gapped_matrix_may_still_have_contiguous_rows() {
        let operand = Operand::model("x");
        let gapped = Layout::strided(&[2, 4], &[8, 1], Dtype::Bf16).unwrap();
        assert_eq!(inner_contiguous(operand, &gapped), Ok(()));
        let strided = Layout::strided(&[2, 4], &[8, 2], Dtype::Bf16).unwrap();
        assert_eq!(
            inner_contiguous(operand, &strided).unwrap_err(),
            OperandError::InnerStride { operand, stride: 2 }
        );
        assert_eq!(
            inner_contiguous(operand, &Layout::contiguous(&[], Dtype::F32).unwrap()),
            Ok(())
        );
    }

    #[test]
    fn errors_say_the_operand_and_both_numbers() {
        let refused = OperandError::TooShort {
            operand: Operand::model("the positions"),
            len: 4,
            needed: 8,
        };
        assert_eq!(
            refused.to_string(),
            "the positions holds 4 rows; 8 are read"
        );
        let shape = OperandError::Shape {
            operand: Operand::layer(1, "key projection"),
            shape: Shape::new(&[4096, 4096]),
            expected: Shape::new(&[1024, 4096]),
        };
        assert_eq!(
            shape.to_string(),
            "layer 1's key projection is [4096, 4096], not [1024, 4096]"
        );
    }
}
