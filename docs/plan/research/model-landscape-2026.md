# Open-Weight Model Landscape (mid-2026) vs the Plan's Shape-Family Ladder

**Date:** 2026-07-30. **Question (wayfinder #139):** which currently-relevant open-weight models
does the §3 ladder cover, and which families fall outside it? **Method:** every shape claim below
is sourced from HF `config.json` (fetched 2026-07-30), official tech reports, or vendor
announcements; third-party numbers are flagged. The ladder reference is
[../README.md](../README.md) §3 (rungs), §2.8 (quant ladder), §2.13 (model scope).

**Headline:** the ladder covers the mid-2026 *installed base* almost exactly — but three of the
four largest open releases of H1 2026 (DeepSeek-V4, Kimi K3, Qwen3.5) are **not expressible on
any rung as scoped**. The frontier forked into three attention families the plan does not name:
top-k sparse index attention (DSA-class), sequence-compressed attention (CSA/HCA), and hybrid
linear attention (GDN/KDA-class). Details in §3; verdict in §4.

---

## 1. Coverage table

Shape facts → the rung/family that serves the model, or the named gap. "Rung 2" = dense GQA;
"rung 3" = grouped-GEMM MoE + single-node EP + W4 (incl. MXFP4); "rung 4" = MLA + wide-EP + MTP
+ FP8; "rung 5" = Blackwell/NVFP4/PD; "PL-SWA" = post-launch SWA-hybrid (Gemma-class) family;
"PL-Kimi" = post-launch Kimi-class scale.

| Model (release) | Params / MoE | Attention | Positional | Released quant | Spec head | Ladder fit |
|---|---|---|---|---|---|---|
| DeepSeek-V3 / V3.1 / R1 (2024-12→2025-08) | 671B A37B; 256+1sh, top-8, `noaux_tc` | MLA (`kv_lora_rank` 512, nope 128 / rope 64 / v 128) | YaRN 128K | FP8 e4m3, block [128,128] | MTP-1 | **Rung 4 — exact.** The rung's design model |
| DeepSeek-V3.2 (2025-09-29) | same as V3.1 | MLA + **DSA**: lightning indexer, `index_topk` 2048 | YaRN 128K | FP8 | MTP-1 | Rung 4 + **GAP-1** (indexer + sparse gather) |
| DeepSeek-V4-Pro (2026-04; config verified) | 1.6T A49B; 61L, h7168; 384+1sh, top-6, `noaux_tc` | **CSA/HCA** — MLA is gone: `num_key_value_heads: 1`, `head_dim: 512`, per-layer `compress_ratios` ∈ {0,4,128}, `index_topk: 1024`, mHC residuals | YaRN ×16 → 1,048,576 | FP8 e4m3, block [128,128] | MTP-1 (`num_nextn_predict_layers: 1`) | **GAP-2 — no rung** |
| DeepSeek-V4-Flash (2026-04) | 284B A13B; 43L, h4096; 256+1sh, top-6 | same family: 1 KV head ×512, `q_lora_rank`/`o_lora_rank` 1024, `qk_rope_head_dim` 64, `index_topk` 512, `index_n_heads` 64, `compress_ratios` [0,0,4,128,…] | YaRN ×16 → 1M | FP8 e4m3 [128,128] | MTP-1 | **GAP-2 — no rung** |
| Qwen3 dense 0.6–32B (2025-04) | dense | GQA | RoPE/YaRN ≤128K | BF16 (+FP8 variants) | — | **Rung 2 — exact** |
| Qwen3-30B-A3B / 235B-A22B (2025) | 128 experts top-8, no shared | GQA | RoPE/YaRN | BF16 + official FP8 | — | **Rung 3 — exact.** The rung's gate models |
| Qwen3-Next-80B-A3B (2025-09) | 512+1sh, top-10 | **hybrid 3:1 Gated DeltaNet : gated attention** | partial RoPE 0.25 | BF16 | MTP | **GAP-3** |
| Qwen3.5-397B-A17B (2026-02-16; config verified) | 60L; 512 experts top-10 + shared expert (`moe_intermediate_size` 1024) | hybrid: `full_attention_interval: 4`; GDN linear layers (16 K-heads / 64 V-heads @128, conv 4); full layers GQA 32 heads / **2 KV heads** @256 | `partial_rotary_factor` 0.25, θ 1e7, 262,144 ctx | BF16 (+FP8 variants) | **MTP** (`mtp_num_hidden_layers: 1`) | **GAP-3 — no rung.** This is the *current* Qwen line |
| Qwen3.6 (2026-04) | family continuation | same hybrid GDN architecture (vendor repo) | — | — | — | **GAP-3** |
| Kimi K2 / K2-Thinking (2025-07 / 2025-11) | 1T A32B; 61L; 384+1sh, top-8 | MLA (DeepSeek-V3 dims) | 128K→256K | FP8; K2-Thinking **native INT4 QAT** (compressed-tensors, group 32) | — | **Rung 4 family @ PL-Kimi scale** — as planned |
| Kimi K2.5 / K2.6 / K2.7 (2026-01→06) | 1.1T (HF listing); same 61L/384-expert MLA core + 400M MoonViT vision | MLA | 256K | modified-MIT weights | — | **Rung 4 family @ PL-Kimi** (text path; vision out of scope) |
| Kimi K3 (2026-07-16, weights 07-26; config verified) | 2.8T A104B; 93L text, h7168; **896 experts top-16 + 2 shared**, `moe_intermediate_size` 3072 | **hybrid ~3:1 KDA : MLA** — 69 KDA linear layers (head_dim 128, `short_conv_kernel_size` 4), 24 full-attn MLA layers (`kv_lora_rank` 512, nope 128/rope 64/v 128) | 1,048,576 ctx | **native MXFP4** (`mxfp4-pack-quantized`, group 32; 1.56 TB) | — (not in fetched config) | **GAP-3 + PL-Kimi — no rung.** Kimi-class no longer means pure MLA |
| GLM-4.5 / 4.6 (2025) | 355B A32B; 160 experts top-8 | GQA, partial RoPE | 128K | BF16 + FP8 | MTP | Rung 3 shape + MTP (rung-4 item) — serviceable |
| GLM-5 / 5.2 (2026-02-11; config verified) | 744B A~40B; 78L; 256+1sh top-8, `noaux_tc`, `routed_scaling_factor` 2.5 | **MLA + DSA**: `kv_lora_rank` 512, `q_lora_rank` 2048, nope 192 / rope 64 / **v 256**; `index_topk` 2048, `index_n_heads` 32; 5.2 adds IndexShare (indexer shared across 4 layers) | θ 1e6 interleaved; 202,752 ctx | BF16 + official FP8 repo | MTP-1 | **Rung 4 + GAP-1.** MIT license; the most rung-4-shaped 2026 frontier model |
| gpt-oss-120b / 20b (2025-08; **no 2026 refresh**) | 117B A5.1B (36L, 128 experts top-4) / 21B A3.6B (24L, 32 experts) | alternating **SWA(128) + full**, GQA 64/8 @64, **learned attention sinks** | YaRN 131,072 | **MXFP4 MoE weights** (4.25 bpp) | — | Rung 3 quant ✓; attention → PL-SWA + **GAP-5** (sinks flag) |
| Llama 4 Scout / Maverick (2025-04; last open Llama — Meta's 2026 successor "Avocado" is reported closed) | 109B A17B, 16 experts / 400B A17B, 128 experts | **iRoPE**: 3:1 chunked local attention (8192, RoPE) : global **NoPE** layers | 10M / 1M (claimed) | BF16 + FP8 | — | Rung 3 MoE + chunked-local variant of PL-SWA. Family dead-ended; low priority |
| Mistral Large 3 (2025-12-02) | 675B A41B MoE | full attention (vendor docs; gated repo — not config-verified) | 256K | **BF16 + FP8 + official NVFP4**; official **EAGLE draft** repo | EAGLE (vendor-shipped) | Rung 3 shape at rung-4 scale; NVFP4/EAGLE flags (§2, GAP-7) |
| Mistral Small 4 119B (2026-03), Medium 3.5 128B, Leanstral-1.5-119B-A6B | dense + MoE mix | — | — | NVFP4 + EAGLE variants shipped | EAGLE | Rungs 2/3 + same flags |
| MiniMax-M2 / M2.1 / M2.5 / M2.7 (2025-12→2026-03) | 230B A10B MoE | **plain GQA full attention** — no SWA, no linear | RoPE | BF16 | — | **Rung 3 — exact.** The friendliest frontier family for the ladder |
| MiniMax-M3 (2026-06; config verified) | 427B (HF); 60L; 128 experts top-4 + 1 shared | GQA 64/4 @128, per-head QK-norm + **block-sparse selection**: `sparse_topk_blocks: 16` × `sparse_block_size: 128`, 4 index heads, sparse from layer 3 | `partial_rotary_factor` 0.5, θ 5e6; 1,048,576 ctx | BF16 + **MXFP8** repo | **`num_mtp_modules: 7`**, `num_nextn_predict_layers: 1` | **GAP-1 (block variant) + multi-depth MTP (GAP-7)** |
| Gemma 3 (2025-03) | dense 1–27B | 5:1 SWA(1024) : global, dual θ | 128K | BF16 (+QAT int4) | — | **PL-SWA — the family's anchor, as planned** |
| Gemma 4 (2026-04; 31B config verified) | E2B/E4B (PLE), 12B, 31B dense, **26B-A4B MoE (128 experts top-8)** | 5:1 `sliding_attention`(1024) : `full_attention`; **global layers have different geometry**: `global_head_dim: 512`, 4 global KV heads vs local 16 @256; `num_kv_shared_layers` field (cross-layer KV sharing in E-variants) | dual-θ: local 1e4 default, global 1e6 `rope_type: "proportional"` (p-RoPE), partial 0.25; 262,144 ctx | BF16, Apache-2.0 | official **MTP drafter** checkpoints (2026-04-16) | PL-SWA + **GAP-6** (heterogeneous head geometry / shared-KV accounting) |
| Nemotron 3 Nano / Super / Ultra (2025-12 / 2026-03 / 2026-06) | 31.6B A3.2B (128 experts top-6) / 120B A12B (LatentMoE) / 550B A55B | **hybrid Mamba-2 : GQA transformer** | 1M ctx (Nano, Ultra) | **Super pre-trained in NVFP4** | MTP layers (Super) | **GAP-4 — no rung** |
| Granite 4.0 (2025-10), Falcon-H1 (2025) | hybrid incl. MoE (e.g. Small 32B-A9B) | Mamba-2 / transformer hybrid, NoPE (Granite) | — | BF16 | — | **GAP-4** |
| Ling/Ring 2.5 1T (Ant, 2026) | 1T MoE | hybrid Lightning (linear) + MLA | — | — | — | **GAP-3 variant** (third-party sourced; verify before acting) |
| Arcee Trinity Large (2026-01) | 400B A13B | 3:1 SWA(4096) : global, gated attention, **NoPE in global layers**, QK-norm | RoPE local / NoPE global | — | — | PL-SWA (window 4096 variant) |
| Step 3.5 Flash (2026) | 196B A11B | gated attention | RoPE | — | **MTP-3** (multi-depth, used at inference) | Rung 3 shape + GAP-7 |

Notes on the R-series: there is no standalone DeepSeek "R2" — reasoning merged into the V3.1+
hybrid-thinking line and continues in V4; "DeepSeek-class" tracking means tracking the V-line.

## 2. Quant-format landscape vs the §2.8 ladder

| Format | Mid-2026 reality | Ladder status |
|---|---|---|
| **FP8 e4m3, block-scale [128,128]** | Still *the* frontier release format: DeepSeek V3.x **and V4**, GLM-5-FP8, Qwen FP8 variants | **Covered** — rung 2 core path. Correct bet |
| **MXFP4** (group 32, E8M0 scale) | No longer just gpt-oss: **Kimi K3 ships native MXFP4 as its primary 2.8T checkpoint** (`mxfp4-pack-quantized`) | **Covered** as rung-3 W4 weight-only — and now load-bearing for a flagship, not a curiosity |
| **NVFP4** (group 16, FP8-E4M3 scale + tensor scale) | Vendors ship official NVFP4 checkpoints (Mistral Large 3 / Small 4); **Nemotron 3 Super is pre-trained in NVFP4** — NVFP4 as *primary* artifact, not a derivative | Rung 5 covers NVFP4 *compute on Blackwell*. **Small gap:** serving vendor NVFP4 checkpoints on Hopper needs NVFP4 *weight-only dequant* in the rung-3 Marlin-class kernel (group-16/FP8-scale variant of the MXFP4 path) |
| **MXFP8** (group 32) | New: MiniMax-M3-MXFP8 official repo | Not in ladder. Small delta on the FP8 path (per-32 scales vs [128,128] blocks) — absorb when/if M3-class is in scope |
| **INT4 QAT** (compressed-tensors, group 32) | Kimi K2-Thinking native release format | **Covered** by rung-3 W4 |
| **GGUF** | unchanged non-goal | ✓ |

Context lengths: 1M-token default is now standard at the frontier (V4, K3, M3, Nemotron Ultra —
all `max_position_embeddings: 1048576`, V4 via YaRN ×16). Not a shape gap, but rung-4/5 KV
capacity planning and the bench harness's long-context workloads should assume 1M, not 128K.

## 3. Gaps the ladder cannot express

**GAP-1 — Top-k sparse index attention (DSA-class).**
Models: DeepSeek-V3.2, GLM-5/5.2 (`index_topk: 2048`, 32 index heads, IndexShare in 5.2),
MiniMax-M3 (block variant: top-16 × 128-token blocks). What's missing: a lightning-indexer
kernel (small MQA-style FP8 scoring head), top-k selection, and sparse-gather attention over
selected tokens (FlashMLA-class kernels have sparse variants upstream). KV accounting is
*unchanged* — full KV is retained; DSA is compute-side sparsity. Which rung it perturbs:
**rung 4** — the two most rung-4-shaped frontier checkpoints of 2026 (GLM-5, V3.2) both carry an
indexer, so "DeepSeek-class MLA serving" without DSA serves only 2025 checkpoints.
**Fog-graduation: YES** — as a rung-4 scope amendment (MLA + optional indexer), not a new rung.

**GAP-2 — Sequence-compressed attention (CSA/HCA, DeepSeek-V4).**
The launch-bar family itself left MLA. V4 attention per config: a single 512-dim KV head
(`num_key_value_heads: 1`, `head_dim: 512`, `q_lora_rank`/`o_lora_rank`), per-layer
`compress_ratios` ∈ {0, 4, 128} (CSA ≈ 4×, HCA ≈ 128× sequence compression), top-k index
selection over the compressed stream, mHC residual connections. Result (paper): ~10% of V3.2's
KV, 1M ctx. Napkin: 61L × 512 dims × mixed {¼, 1/128} retention ≈ 4–8 KB/token BF16 vs ~70
KB/token for V3-MLA. What's missing: everything — compression kernels, indexer, and **KV
accounting the block manager cannot currently represent**: layers fill blocks at *different
token rates* (a ratio-128 layer stores one slot per 128 tokens), so tokens-per-block, prefix
block hashing, and chunked-prefill boundaries all become per-layer-group quantities. Which rung
it perturbs: **rung 4's definition and the §9 launch gate** — on the plan's 24–30-month horizon,
"DeepSeek-class at launch" (2028) will mean V4-successors, not V3.x. The cross-rung insurance
(layer-group KV accounting, `AttentionBackend` capability trait) is pointed in exactly the right
direction but must support per-group *fill rates*, not just per-group geometry.
**Fog-graduation: YES — the largest candidate on the map.**

**GAP-3 — Hybrid linear attention (GDN/KDA-class) + recurrent-state cache.**
Models: **Qwen3.5/3.6 (the current Qwen line)**, Qwen3-Next, **Kimi K3**, Ling/Ring 2.5. Shape:
3:1 linear:full interleave; linear layers keep a fixed-size recurrent state (delta-rule S matrix
+ short-conv state), not a KV cache; full layers are GQA (Qwen3.5: 2 KV heads) or MLA (K3).
What's missing: chunked delta-rule scan kernels (prefill) + recurrent-step kernels (decode);
a per-layer-group cache manager holding {KV blocks} for full layers and {state snapshots} for
linear layers; **prefix caching semantics change** — a radix hit on a linear layer requires a
stored state snapshot at the block boundary, not block reuse, and preemption-by-recompute must
re-materialize states. Which rungs it perturbs: **rungs 2–3's assumption that "Qwen3 family" is
a stable reference** — Qwen3-235B/30B remain the correct 2025-generation gate models, but the
*current* Qwen generation is unservable by the ladder; post-launch **Kimi-class now means KDA
hybrid** (K3), not pure-MLA scale-up. **Fog-graduation: YES.**

**GAP-4 — Mamba-2/SSM-hybrid MoE (Nemotron 3 Nano/Super/Ultra, Granite 4.0, Falcon-H1).**
Same cache-shape problem as GAP-3 (fixed-size SSM state per layer) with different kernels
(selective-scan). NVIDIA is committing hard (550B Ultra, NVFP4-pretrained Super, open weights +
data). Perturbs no pre-launch rung; competes for post-launch breadth with PL-SWA. Note that
solving GAP-3's heterogeneous cache manager solves ~80% of GAP-4's engine work; only kernels
differ. **Fog-graduation: launch scope NO; post-launch watchlist YES** (demand-driven, after
GAP-3 infrastructure exists).

**GAP-5 — Attention sinks + tiny-window alternating SWA (gpt-oss).**
gpt-oss (unchanged since 2025-08, still the most-deployed open MoE) needs learned per-head sink
logits in the softmax and SWA(128) on alternating layers. Rung 2's FA3 work re-enables SWA;
sinks are a small kernel + `AttentionBackend` capability flag. Perturbs rung 3 (gpt-oss is
otherwise a perfect rung-3 citizen: MoE top-4 + MXFP4). **Fog-graduation: NO — absorb into
rung 2/3 scope now** (cheap, and the plan already cites MXFP4-for-gpt-oss as a rung-3 target).

**GAP-6 — Heterogeneous per-layer head geometry + cross-layer KV sharing (Gemma 4).**
Global layers: 4 KV heads @512; local layers: 16 KV heads @256; `num_kv_shared_layers` lets
some layers own **zero** KV (share earlier layers' K/V); dual per-layer-type RoPE (θ 1e4 default
vs θ 1e6 "proportional"/partial 0.25). The planned layer-group KV accounting must support
different bytes-per-token per group and zero-KV groups, and rope config must be per-layer-type.
Perturbs PL-SWA only — but the accounting API freezes at **PR-5**, so the design check is now.
**Fog-graduation: NO — verify in PR-5's layer-group API design instead.**

**GAP-7 — Speculative-decoding heads are now universal, and multi-depth.**
Every 2026 frontier release ships MTP: Qwen3.5 (`mtp_num_hidden_layers: 1`), GLM-5, V4,
Nemotron 3 Super, Step 3.5 (MTP-3), and MiniMax-M3 with **seven** MTP modules; vendors also ship
ready-made draft heads (deepseek-ai publishes EAGLE3 and DFlash repos; Mistral ships `-Eagle`
variants of every flagship; Google ships Gemma 4 MTP drafters). The plan's decision #12 ("MTP at
rung 4 only; EAGLE-3 out entirely") was priced when EAGLE required training a draft — that cost
is now zero. Perturbs: rung 3 (serving Qwen3.5-class competitively wants its MTP head) and the
scheduler's `num_tokens = 1+k` design, which should allow k>1 chains (multi-depth) from day one.
**Fog-graduation: partial — keep MTP execution at rung 4, but (a) make the scheduler's k
variable, (b) log a decision-review trigger: if launch-era baselines default-enable vendor
EAGLE/DFlash drafts, "no EAGLE-3" stops being competitive-neutral.**

## 4. Verdict

The shape-family thesis (§2.13: "shape families, not model count") **survives — the specific
family list does not.** Rungs 0–3 are untouched: Qwen3 dense/MoE, gpt-oss (+ sinks flag),
MiniMax-M2.x, GLM-4.5, Mistral Large 3, and Llama 3.x/4 are all expressible, and the rung-3
gate models remain the right ones. Rung 4 as scoped (pure MLA) exactly serves DeepSeek-V3.x/R1,
Kimi K2.x, and GLM-5-minus-indexer — a large, real, still-deployed cohort — and remains the
right *teaching* rung. But every flagship released after 2026-01 moved: GLM-5 needs DSA (GAP-1,
small), DeepSeek-V4 needs CSA/HCA (GAP-2, large), and Qwen3.5/3.6 + Kimi K3 need hybrid-linear
(GAP-3, large). On a 24–30-month runway, the launch-gate phrase "DeepSeek-class" will point at a
V4-successor, so rung 4 should absorb GAP-1 now and the fog map should carry GAP-2 and GAP-3 as
graduation candidates with the layer-group KV accounting API (PR-5) designed for per-group
**fill rates and cache kinds**, not just per-group geometry — that single design decision is
what keeps all three gaps absorbable without an engine rewrite.

## Sources

**Configs (fetched 2026-07-30):**
[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) ·
[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) ·
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) ·
[GLM-5](https://huggingface.co/zai-org/GLM-5) ·
[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) ·
[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) ·
[gemma-4-31B](https://huggingface.co/google/gemma-4-31B) ·
org listings: [deepseek-ai](https://huggingface.co/deepseek-ai) (incl. EAGLE3/DFlash repos) ·
[moonshotai](https://huggingface.co/moonshotai) ·
[MiniMaxAI](https://huggingface.co/MiniMaxAI) (incl. M3-MXFP8) ·
[mistralai](https://huggingface.co/mistralai) (incl. -NVFP4/-Eagle repos)

**Papers / vendor:**
[DeepSeek-V4 report (arXiv 2606.19348)](https://arxiv.org/abs/2606.19348) — CSA/HCA, mHC, 32T
tokens, sizes ·
[GLM-5 report (arXiv 2602.15763)](https://arxiv.org/html/2602.15763v1) ·
[Kimi-K2.5 repo](https://github.com/MoonshotAI/Kimi-K2.5) ·
[Kimi K3 coverage (VentureBeat)](https://venturebeat.com/technology/chinas-moonshot-ai-releases-kimi-k3-the-largest-open-source-model-ever-rivaling-top-u-s-systems) ·
[Qwen3.6 repo](https://github.com/QwenLM/Qwen3.6) ·
[gpt-oss announcement](https://openai.com/index/introducing-gpt-oss/) + [repo](https://github.com/openai/gpt-oss) ·
[Llama 4 announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) ·
[Meta "Avocado" closed-successor reporting](https://seekingalpha.com/article/4851635-wall-street-lunch-meta-to-launch-avocado-llm-to-take-on-google-openai) ·
[Mistral 3 announcement](https://mistral.ai/news/mistral-3) ·
[Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) ·
[Nemotron 3 Nano (arXiv 2512.20848)](https://arxiv.org/abs/2512.20848) ·
[Nemotron 3 white paper](https://arxiv.org/pdf/2512.20856) ·
[Nemotron 3 Super tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) ·
[Nemotron 3 Ultra coverage](https://www.marktechpost.com/2026/06/04/nvidia-ai-releases-nemotron-3-ultra-an-open-550b-mixture-of-experts-hybrid-mamba-transformer-for-long-running-agents/) ·
[MiniMax-M1 (arXiv 2506.13585)](https://arxiv.org/pdf/2506.13585) ·
[MiniMax-M2 series report (arXiv 2605.26494)](https://arxiv.org/pdf/2605.26494) ·
[Granite 4.0 announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

**Third-party (dates/roundup only, flagged where load-bearing):**
[Raschka: architectures Jan–Feb 2026](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight)
(Trinity, Step 3.5, Ling/Ring 2.5, Nanbeige, Tiny Aya, Sarvam) ·
[mlabonne: Qwen3.5](https://huggingface.co/blog/mlabonne/qwen35) ·
[mlabonne: MiniMax-M2.5](https://huggingface.co/blog/mlabonne/minimax-m25) ·
[Kimi K3 MXFP4 overview (HF community)](https://huggingface.co/blog/ResterChed/kimi-k3-model-overview-mxfp4-quantization-open-wei)
