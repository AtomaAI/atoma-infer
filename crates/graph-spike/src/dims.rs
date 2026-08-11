//! Llama-8B-shaped model dimensions and the sizes the spike derives from them.

/// Size of one bf16 element in bytes; every weight and activation the step touches is bf16.
pub const BF16_BYTES: usize = 2;

/// The model shape every spike step runs: Llama-8B-shaped per the #143 runsheet, with the layer
/// count the only configurable axis — 2–4 layers reproduce the full op mix without the full
/// memory bill.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ModelDims {
    pub num_layers: usize,
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
}

impl ModelDims {
    /// The Llama-8B shape (hidden 4096, 32 q / 8 kv heads, head_dim 128, FFN 14336, vocab
    /// 128256) with `num_layers` layers.
    pub fn llama_8b_shaped(num_layers: usize) -> Self {
        assert!(
            (1..=32).contains(&num_layers),
            "num_layers {num_layers} out of range: the spike runs 1..=32 layers"
        );
        let dims = Self {
            num_layers,
            hidden: 4096,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            ffn: 14336,
            vocab: 128256,
        };
        assert_eq!(dims.hidden, dims.num_q_heads * dims.head_dim);
        dims
    }

    /// Output width of the fused qkv projection in elements: q heads plus one k and one v head
    /// group.
    pub fn qkv_out(&self) -> usize {
        (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    /// Width of the packed k (or v) segment inside the qkv projection, in elements.
    pub fn kv_width(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// The attention softmax scale, `1 / sqrt(head_dim)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Bytes of one layer's weights in bf16: qkv, o, gate, up, down projections and the two
    /// RMSNorm gains.
    pub fn layer_weight_bytes(&self) -> usize {
        let qkv = self.hidden * self.qkv_out();
        let o = self.hidden * self.hidden;
        let gate_up = 2 * self.hidden * self.ffn;
        let down = self.ffn * self.hidden;
        let norms = 2 * self.hidden;
        (qkv + o + gate_up + down + norms) * BF16_BYTES
    }

    /// Bytes of every weight the step touches: all layers plus the embedding table, final norm,
    /// and lm head.
    pub fn total_weight_bytes(&self) -> usize {
        let embedding = self.vocab * self.hidden;
        let lm_head = self.vocab * self.hidden;
        let final_norm = self.hidden;
        self.num_layers * self.layer_weight_bytes()
            + (embedding + lm_head + final_norm) * BF16_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_out_covers_q_and_both_kv_groups() {
        let dims = ModelDims::llama_8b_shaped(4);
        assert_eq!(dims.qkv_out(), (32 + 8 + 8) * 128);
        assert_eq!(dims.kv_width(), 8 * 128);
    }

    #[test]
    fn layer_weight_bytes_match_the_hand_computed_8b_shape() {
        let dims = ModelDims::llama_8b_shaped(4);
        // qkv 4096*6144 + o 4096*4096 + gate/up 2*4096*14336 + down 14336*4096 + norms 2*4096
        // elements, at 2 bytes each.
        let elements = 4096 * 6144 + 4096 * 4096 + 2 * 4096 * 14336 + 14336 * 4096 + 2 * 4096;
        assert_eq!(dims.layer_weight_bytes(), elements * 2);
    }

    #[test]
    fn total_weight_bytes_add_embedding_and_lm_head() {
        let dims = ModelDims::llama_8b_shaped(2);
        let expected = 2 * dims.layer_weight_bytes() + (2 * 128_256 * 4096 + 4096) * 2;
        assert_eq!(dims.total_weight_bytes(), expected);
    }

    #[test]
    #[should_panic(expected = "num_layers 0 out of range")]
    fn zero_layers_are_rejected() {
        ModelDims::llama_8b_shaped(0);
    }
}
