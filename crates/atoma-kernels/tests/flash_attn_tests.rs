#![cfg(feature = "cuda")]

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use serial_test::serial;

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

/// Builds `q`, `k` and `v` of shape `[1, num_heads, seq_len, head_dim]`, with values spread over
/// `[0, 1)` so f16 rounding does not dominate the comparison against the reference.
fn qkv(
    device: &Device,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let elem_count = num_heads * seq_len * head_dim;
    let base = (Tensor::arange(0u32, elem_count as u32, device)?.to_dtype(DType::F32)?
        / elem_count as f64)?
        .reshape((1, num_heads, seq_len, head_dim))?
        .to_dtype(DType::F16)?;
    let k = (&base * 0.75)?;
    let v = (&base * 0.5)?;
    Ok((base, k, v))
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
#[serial]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        atoma_kernels::flash_attn(&q, &k, &v, 0.5, false)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys1, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    assert_eq!(ys2.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys2, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);

    Ok(())
}

#[test]
#[serial]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 2u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        atoma_kernels::flash_attn_varlen(&q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false)?
            .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    Ok(())
}

#[test]
#[serial]
fn flash_attn_varlen_with_block_table() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let block_size = 16;
    let num_blocks = 2;
    let q = Tensor::arange(0u32, 512, &device)?
        .to_dtype(DType::F16)?
        .reshape((32, 2, 8))?;
    let k = (&q / 40.)?.reshape((num_blocks, block_size, 2, 8))?;
    let v = (&q / 50.)?.reshape((num_blocks, block_size, 2, 8))?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0u32, 32u32, 64u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 32u32, 64u32], &device)?;

    let ys = {
        let block_table = Some(Tensor::arange(0u32, 4, &device)?.reshape((2, 2))?);
        atoma_kernels::flash_attn_varlen_with_block_table(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            32,
            32,
            0.5,
            None,
            None,
            block_table.as_ref(),
        )?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[32, 2, 8]);

    let q = Tensor::arange(0u32, 512, &device)?
        .to_dtype(DType::F16)?
        .reshape((32, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let should_be_ys =
        atoma_kernels::flash_attn_varlen(&q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false)?;
    let should_be_ys = should_be_ys.to_dtype(DType::F32)?;

    assert_eq!(should_be_ys.dims(), &[32, 2, 8]);
    assert_eq!(to_vec3_round(ys, 10)?, to_vec3_round(should_be_ys, 10)?);

    Ok(())
}

#[test]
#[serial]
fn flash_attn_kv_cache() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_k = Tensor::new(&[2u32], &device)?;

    let ys = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        atoma_kernels::flash_attn_kv_cache_full(
            &q,
            &k,
            &v,
            None,
            0.5,
            None,
            Some(&seqlens_k),
            false,
        )?
        .transpose(1, 2)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[1, 3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys.squeeze(0)?, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    Ok(())
}

#[test]
#[serial]
fn test_flash_attn_kv_cache_with_block_table() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let block_size = 16;
    let num_blocks = 2;
    let q = Tensor::arange(0u32, 512, &device)?
        .to_dtype(DType::F16)?
        .reshape((32, 1, 2, 8))?;
    let k = (&q / 40.)?.reshape((num_blocks, block_size, 2, 8))?;
    let v = (&q / 50.)?.reshape((num_blocks, block_size, 2, 8))?;
    let q = (&q / 30.)?;

    let seqlens_k = Tensor::new(&[1u32; 32], &device)?;

    let ys = {
        let block_table = Some(Tensor::arange(0u32, 64, &device)?.reshape((32, 2))?);
        atoma_kernels::flash_attn_kv_cache_full(
            &q,
            &k,
            &v,
            None,
            0.5,
            block_table.as_ref(),
            Some(&seqlens_k),
            false,
        )?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[32, 1, 2, 8]);
    let ys = ys.squeeze(1)?;

    let q = Tensor::arange(0u32, 512, &device)?
        .to_dtype(DType::F16)?
        .reshape((32, 2, 8))?;
    let k = (&q / 40.)?.reshape((num_blocks, block_size, 2, 8))?;
    let v = (&q / 50.)?.reshape((num_blocks, block_size, 2, 8))?;
    let q = (&q / 30.)?;

    let seqlens_k = Tensor::from_vec((0u32..=32).collect::<Vec<_>>(), (33,), &device)?;

    let should_be_ys = {
        let block_table = Some(Tensor::arange(0u32, 64, &device)?.reshape((32, 2))?);
        atoma_kernels::flash_attn_varlen_with_block_table(
            &q,
            &k,
            &v,
            None,
            &seqlens_k,
            &seqlens_k,
            32,
            32,
            0.5,
            None,
            None,
            block_table.as_ref(),
        )?
    };
    let should_be_ys = should_be_ys.to_dtype(DType::F32)?;

    assert_eq!(should_be_ys.dims(), &[32, 2, 8]);
    assert_eq!(to_vec3_round(ys, 6)?, to_vec3_round(should_be_ys, 6)?);

    Ok(())
}

/// Attention over `q`, `k` and `v` of shape `[1, num_heads, seq_len, head_dim]`, masked so each
/// query only attends to keys at or before its own position.
fn fa_causal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (_, _, seq_len, _) = q.dims4()?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|row| (0..seq_len).map(move |column| f32::from(column > row)))
        .collect();
    let mask = (Tensor::from_vec(mask, (seq_len, seq_len), q.device())? * f64::from(f32::MIN))?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?.broadcast_add(&mask)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    Ok(att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?)
}

/// Asserts that the flash output matches a reference computed the naive way.
fn assert_close(expected: &Tensor, actual: &Tensor, what: &str) -> Result<()> {
    let diff = expected
        .to_dtype(DType::F32)?
        .sub(&actual.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_vec0::<f32>()?;
    assert!(diff < 2e-3, "{what} deviates from the reference by {diff}");
    Ok(())
}

/// Head dimensions that are not a multiple of 32 take the uneven-K kernels, which
/// `models::FlashAttention::supported_head_sizes` advertises and #161 re-enabled. Every entry
/// point that serves those head dims is covered, because each dispatches differently.
fn assert_uneven_head_dim_matches_reference(head_dim: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (num_heads, seq_len) = (2, 8);
    let (q, k, v) = qkv(&device, num_heads, seq_len, head_dim)?;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let (qt, kt, vt) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);

    for causal in [false, true] {
        let expected = if causal {
            fa_causal(&q, &k, &v, softmax_scale)?
        } else {
            fa_acausal(&q, &k, &v, softmax_scale)?
        }
        .i(0)?;
        let actual = atoma_kernels::flash_attn(&qt, &kt, &vt, softmax_scale, causal)?
            .transpose(1, 2)?
            .i(0)?;
        assert_eq!(actual.dims(), &[num_heads, seq_len, head_dim]);
        assert_close(
            &expected,
            &actual,
            &format!("head dim {head_dim}, causal {causal}"),
        )?;
    }

    // The varlen entry point takes `[total_q, num_heads, head_dim]` and a single sequence here.
    let seqlens = Tensor::new(&[0u32, seq_len as u32], &device)?;
    let expected = fa_acausal(&q, &k, &v, softmax_scale)?
        .i(0)?
        .transpose(0, 1)?;
    let actual = atoma_kernels::flash_attn_varlen(
        &q.i(0)?.transpose(0, 1)?.contiguous()?,
        &k.i(0)?.transpose(0, 1)?.contiguous()?,
        &v.i(0)?.transpose(0, 1)?.contiguous()?,
        &seqlens,
        &seqlens,
        seq_len,
        seq_len,
        softmax_scale,
        false,
    )?;
    assert_eq!(actual.dims(), &[seq_len, num_heads, head_dim]);
    assert_close(&expected, &actual, &format!("head dim {head_dim}, varlen"))?;

    Ok(())
}

#[test]
#[serial]
fn flash_attn_head_dim_80() -> Result<()> {
    assert_uneven_head_dim_matches_reference(80)
}

#[test]
#[serial]
fn flash_attn_head_dim_112() -> Result<()> {
    assert_uneven_head_dim_matches_reference(112)
}

#[test]
#[serial]
fn flash_attn_rejects_softcap() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (q, k, v) = qkv(&device, 2, 8, 64)?;
    let alibi_slopes = Tensor::zeros(2, DType::F32, &device)?;

    let result = atoma_kernels::flash_attn_alibi_windowed_with_softcap(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        &alibi_slopes,
        0.5,
        None,
        Some(0),
        Some(30.0),
    );

    let error = result
        .expect_err("softcap must be rejected, not silently dropped")
        .to_string();
    assert!(error.contains("softcap"), "{error}");
    assert!(error.contains("FLASHATTENTION_DISABLE_SOFTCAP"), "{error}");

    Ok(())
}

#[test]
#[serial]
fn flash_attn_rejects_a_sliding_window() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (q, k, v) = qkv(&device, 2, 8, 64)?;
    let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);

    for (window_size_left, window_size_right) in
        [(Some(4), None), (None, Some(4)), (Some(4), Some(4))]
    {
        let result = atoma_kernels::flash_attn_windowed(
            &q,
            &k,
            &v,
            0.5,
            window_size_left,
            window_size_right,
        );
        let error = result
            .expect_err("a sliding window must be rejected, not silently widened")
            .to_string();
        assert!(error.contains("sliding-window attention"), "{error}");
        assert!(error.contains("FLASHATTENTION_DISABLE_LOCAL"), "{error}");
    }

    Ok(())
}

#[test]
#[serial]
fn flash_attn_accepts_full_and_causal_attention() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (q, k, v) = qkv(&device, 2, 8, 64)?;
    let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);

    atoma_kernels::flash_attn_windowed(&q, &k, &v, 0.5, None, None)?;
    atoma_kernels::flash_attn_windowed(&q, &k, &v, 0.5, None, Some(0))?;

    Ok(())
}

#[test]
#[serial]
fn flash_attn_varlen_rejects_seqlens_the_kernels_cannot_read_as_i32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0i64, 2i64], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 2u32], &device)?;

    let result = atoma_kernels::flash_attn_varlen(
        &q.transpose(0, 1)?,
        &k.transpose(0, 1)?,
        &v.transpose(0, 1)?,
        &seqlens_q,
        &seqlens_k,
        32,
        32,
        0.5,
        false,
    );

    let error = result
        .expect_err("i64 sequence lengths must be rejected, not reinterpreted as i32")
        .to_string();
    assert!(error.contains("seqlens_q"), "{error}");

    Ok(())
}

/// CPU log-sum-exp reference: `logsumexp(q @ k^T * softmax_scale)` per query row, with an
/// optional causal mask, in f32 from the same values the kernel sees.
///
/// `q` and `k` are `[num_heads, seq_len, head_dim]`; returns `[num_heads, seq_len_q]`.
fn lse_reference(q: &Tensor, k: &Tensor, softmax_scale: f32, causal: bool) -> Result<Tensor> {
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let scores = (q.matmul(&k.t()?.contiguous()?)? * softmax_scale as f64)?;
    let scores = if causal {
        let (_num_heads, seqlen_q, seqlen_k) = scores.dims3()?;
        let mask: Vec<f32> = (0..seqlen_q)
            .flat_map(|i| (0..seqlen_k).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, seqlen_q, seqlen_k), scores.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };
    let row_max = scores.max_keepdim(candle_core::D::Minus1)?;
    let sum = scores
        .broadcast_sub(&row_max)?
        .exp()?
        .sum(candle_core::D::Minus1)?;
    let lse = (sum.log()? + row_max.squeeze(candle_core::D::Minus1)?)?;
    Ok(lse)
}

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    Ok(lhs
        .sub(rhs)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?)
}

/// The fixed-sequence-length LSE layout is `[batch, num_heads, seqlen_q]`; the values must be
/// the log-sum-exp of the masked, scaled scores — the quantity split-attention merging combines
/// on, previously written to a buffer nothing could read (#166).
#[test]
#[serial]
fn flash_attn_lse_matches_cpu_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (q, k, v) = qkv(&device, 3, 6, 64)?;

    for causal in [false, true] {
        let (output, lse) = atoma_kernels::flash_attn_with_lse(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            0.5,
            causal,
        )?;

        // The attention output itself is unchanged by observing the LSE.
        let baseline = atoma_kernels::flash_attn(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            0.5,
            causal,
        )?;
        let output_diff = max_abs_diff(
            &output.to_dtype(DType::F32)?,
            &baseline.to_dtype(DType::F32)?,
        )?;
        assert_eq!(output_diff, 0.0, "causal={causal}");

        assert_eq!(lse.dims(), &[1, 3, 6], "causal={causal}");
        let reference = lse_reference(&q.i(0)?, &k.i(0)?, 0.5, causal)?;
        let diff = max_abs_diff(&lse.i(0)?, &reference)?;
        assert!(diff < 5e-3, "causal={causal}: lse diverges by {diff}");
    }
    Ok(())
}

/// The varlen LSE layout is `[num_heads, total_q]` strided by `total_q`, so with unequal
/// sequence lengths each sequence's columns sit at its `seqlens_q` offsets — a wrong `total_q`
/// stride lands sequence 1's values in the wrong columns and fails this comparison.
#[test]
#[serial]
fn flash_attn_varlen_lse_matches_cpu_reference_with_unequal_lengths() -> Result<()> {
    let device = Device::new_cuda(0)?;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const LENGTHS: [usize; 2] = [3, 5];
    let total: usize = LENGTHS.iter().sum();

    let elem_count = total * NUM_HEADS * HEAD_DIM;
    let base = (Tensor::arange(0u32, elem_count as u32, &device)?.to_dtype(DType::F32)?
        / elem_count as f64)?
        .reshape((total, NUM_HEADS, HEAD_DIM))?
        .to_dtype(DType::F16)?;
    let k = (&base * 0.75)?;
    let v = (&base * 0.5)?;
    let q = base;
    let seqlens = Tensor::new(&[0u32, 3u32, 8u32], &device)?;

    let (_output, lse) = atoma_kernels::flash_attn_varlen_with_lse(
        &q,
        &k,
        &v,
        &seqlens,
        &seqlens,
        *LENGTHS.iter().max().expect("lengths are non-empty"),
        *LENGTHS.iter().max().expect("lengths are non-empty"),
        0.5,
        true,
    )?;
    assert_eq!(lse.dims(), &[NUM_HEADS, total]);

    let mut start = 0;
    for (sequence_index, &len) in LENGTHS.iter().enumerate() {
        // [len, h, d] -> [h, len, d] for the reference; columns start..start+len in the LSE.
        let q_seq = q.i(start..start + len)?.transpose(0, 1)?.contiguous()?;
        let k_seq = k.i(start..start + len)?.transpose(0, 1)?.contiguous()?;
        let reference = lse_reference(&q_seq, &k_seq, 0.5, true)?;
        let lse_seq = lse.i((.., start..start + len))?;
        let diff = max_abs_diff(&lse_seq, &reference)?;
        assert!(
            diff < 5e-3,
            "sequence {sequence_index}: lse diverges by {diff}"
        );
        start += len;
    }
    Ok(())
}

/// The exact-size check is what turns a mis-allocated LSE buffer into a loud error: before the
/// oracle existed, halving this buffer wrote out of bounds without any test noticing, and the
/// historical `* 128` over-allocation was equally invisible (#166). The internal buffer of the
/// non-LSE entry points is still unobservable — that is exactly why the `_with_lse` paths carry
/// the caller's tensor instead.
#[test]
#[serial]
fn flash_attn_rejects_a_mis_sized_lse_buffer() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (q, k, v) = qkv(&device, 3, 6, 64)?;
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;

    // Required: 1 * 3 * 6 = 18 elements. Halving and the `* 128` inflation must both fail.
    for wrong_elems in [9usize, 18 * 128] {
        let wrong = Tensor::zeros(wrong_elems, DType::F32, &device)?;
        let op = atoma_kernels::FlashAttention {
            softmax_lse: Some(wrong),
            softmax_scale: 0.5,
            alibi_slopes: None,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        };
        let error = q
            .apply_op3(&k, &v, op)
            .expect_err("a mis-sized LSE buffer must be rejected before the launch")
            .to_string();
        assert!(
            error.contains("softmax_lse must hold exactly 18 elements"),
            "{error}"
        );
    }
    Ok(())
}
