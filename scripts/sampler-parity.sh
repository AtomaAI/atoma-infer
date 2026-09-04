#!/usr/bin/env bash
#
# The device sampler against its host reference, on a CUDA rig: runs the ignored integration
# tests that build the sampler over synthetic logits and compare every row's token with the
# reference, check that a seeded request's tokens do not depend on its batch, its row or its
# slot, measure the draw frequencies against the distribution the filters leave, and check that
# the gather overwrites a decoding row's token with what its slot sampled.
#
# Needs a device and a CUDA 12.x toolkit; no checkpoint and no model.
#
# Usage, from a checkout on the rig:
#   scripts/sampler-parity.sh
#
# The flash-attention kernels are a long nvcc build the first time; set FLASH_ATTN_BUILD_DIR to
# keep the build across checkouts.

set -euo pipefail

export CARGO_TERM_COLOR=never

die() {
	echo "sampler-parity: error: $*" >&2
	exit 1
}

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver is not installed"
nvidia-smi >/dev/null || die "nvidia-smi failed — no visible GPU"
command -v nvcc >/dev/null || die "nvcc not on PATH — install a CUDA 12.x toolkit"
command -v cargo >/dev/null || die "cargo not found — install rustup; rust-toolchain.toml pins it"

cutlass_header=crates/atoma-kernels/cutlass/include/cutlass/cutlass.h
if [[ ! -f $cutlass_header ]]; then
	echo "==> CUTLASS submodule is empty; fetching"
	git submodule update --init --depth 1 crates/atoma-kernels/cutlass
fi

echo "==> commit: $(git rev-parse HEAD)"
echo "==> gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"

cargo test -p atoma-engine --locked --features cuda --test sampler_parity -- --ignored --nocapture
