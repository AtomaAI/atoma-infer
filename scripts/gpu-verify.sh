#!/usr/bin/env bash
#
# Manual GPU verification for atoma-infer — run from a checkout on a CUDA rig.
#
# There is no GPU CI runner; this script is the verification path. It preflights the rig
# (driver, CUDA toolkit, NCCL, OpenSSL build dependencies, CUTLASS submodule), builds the
# cuda feature, runs the cuda and cuda,nccl test suites, runs clippy over all features,
# and prints an evidence block to paste into the tracking ticket.
#
# A build failure aborts the run immediately. Test and clippy failures are recorded and
# the run continues, so the evidence block reports every suite; the exit status is
# non-zero if any step failed.

set -euo pipefail

# Logs are parsed for the evidence block and pasted into tickets; keep them free of
# escape sequences.
export CARGO_TERM_COLOR=never

log_dir=""
nccl_header=""
failed_steps=()

die() {
	echo "gpu-verify: error: $*" >&2
	exit 1
}

preflight() {
	command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver is not installed"
	nvidia-smi >/dev/null || die "nvidia-smi failed — no visible GPU"
	command -v nvcc >/dev/null || die "nvcc not on PATH — install a CUDA 12.x toolkit"
	command -v cargo >/dev/null || die "cargo not found — install rustup; rust-toolchain.toml pins the version"

	local candidate
	for candidate in /usr/include/nccl.h /usr/local/include/nccl.h /usr/local/cuda/include/nccl.h; do
		if [[ -r $candidate ]]; then
			nccl_header=$candidate
			break
		fi
	done
	[[ -n $nccl_header ]] || die "nccl.h not found — install libnccl-dev (the cuda,nccl test run needs it to link)"
	ldconfig -p | grep -q libnccl || die "libnccl not in the linker cache — install libnccl2"

	# Without pkg-config and the OpenSSL headers the build dies in openssl-sys before any
	# kernel compiles.
	command -v pkg-config >/dev/null || die "pkg-config not found — install pkg-config"
	pkg-config --exists openssl || die "OpenSSL headers not found — install libssl-dev"

	local cutlass_header=crates/atoma-kernels/cutlass/include/cutlass/cutlass.h
	if [[ ! -f $cutlass_header ]]; then
		echo "==> CUTLASS submodule is empty; fetching"
		git submodule update --init --depth 1 crates/atoma-kernels/cutlass
	fi
	[[ -f $cutlass_header ]] || die "CUTLASS submodule checkout failed — $cutlass_header is still missing"

	echo "==> preflight ok"
}

run_recorded() {
	local name=$1
	shift
	echo
	echo "==> $name: $*"
	if "$@" 2>&1 | tee "$log_dir/$name.log"; then
		echo "==> $name: ok"
	else
		failed_steps+=("$name")
		echo "==> $name: FAILED — continuing so the evidence block reports every suite"
	fi
}

suite_results() {
	awk '
		/^[[:space:]]*Running / {
			line = $0
			sub(/^[[:space:]]*Running /, "", line)
			target = line
			sub(/ \(.*/, "", target)
			bin = line
			sub(/^.*\(/, "", bin)
			sub(/\).*/, "", bin)
			sub(/^.*\//, "", bin)
			sub(/-[0-9a-f]*$/, "", bin)
			suite = bin " (" target ")"
		}
		/^[[:space:]]*Doc-tests / { suite = "doc-tests " $2 }
		/^test result:/ {
			result = $0
			sub(/^test result: /, "", result)
			sub(/; finished in.*/, "", result)
			printf "  %-52s %s\n", suite, result
			suite = "(unknown suite)"
		}
	' "$log_dir/$1.log"
}

evidence() {
	local gpu_count gpu_name driver compute_cap cuda_release nccl_version commit
	gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
	gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd, -)
	driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | paste -sd, -)
	compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd, -)
	cuda_release=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)
	nccl_version=$(awk '
		/^#define NCCL_MAJOR/ { major = $3 }
		/^#define NCCL_MINOR/ { minor = $3 }
		/^#define NCCL_PATCH/ { patch = $3 }
		END { print major "." minor "." patch }
	' "$nccl_header")
	commit=$(git rev-parse HEAD)
	git diff-index --quiet HEAD -- || commit="$commit (dirty)"

	local clippy_status=ok failed
	for failed in "${failed_steps[@]}"; do
		if [[ $failed == clippy ]]; then
			clippy_status=FAILED
		fi
	done

	echo
	echo "=============== GPU verification evidence ==============="
	echo "commit:       $commit"
	echo "gpu:          ${gpu_count}x $gpu_name"
	echo "driver:       $driver"
	echo "compute cap:  $compute_cap"
	echo "cuda toolkit: $cuda_release"
	echo "nccl:         $nccl_version"
	echo "logs:         $log_dir"
	echo
	echo "cargo test --workspace --features cuda:"
	suite_results test-cuda
	echo
	echo "cargo test --workspace --features cuda,nccl:"
	suite_results test-cuda-nccl
	echo
	echo "clippy (--all-targets --all-features -D warnings): $clippy_status"
	if ((${#failed_steps[@]} == 0)); then
		echo "result: PASS"
	else
		echo "result: FAIL (${failed_steps[*]})"
	fi
	echo "========================================================="
}

main() {
	if (($# > 0)); then
		die "gpu-verify.sh takes no arguments"
	fi

	local repo_root
	repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
	cd "$repo_root" || die "cannot cd to $repo_root"

	preflight

	log_dir=$(mktemp -d "${TMPDIR:-/tmp}/gpu-verify-$(date +%Y%m%d-%H%M%S)-XXXX")

	# An explicit build first: a regression in the GPU build path must fail the run here,
	# in minutes, as an unambiguous build failure — not surface later inside a test run.
	echo
	echo "==> build-cuda: cargo build --workspace --features cuda --locked"
	cargo build --workspace --features cuda --locked 2>&1 | tee "$log_dir/build-cuda.log" ||
		die "the cuda feature no longer builds — see $log_dir/build-cuda.log"

	# --no-fail-fast is required: without it cargo stops at the first failing target, and
	# later suites (flash-attn, the guard tests, crates/models) silently never run.
	run_recorded test-cuda cargo test --workspace --features cuda --locked --no-fail-fast
	run_recorded test-cuda-nccl cargo test --workspace --features cuda,nccl --locked --no-fail-fast
	run_recorded clippy cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

	evidence

	((${#failed_steps[@]} == 0)) || exit 1
}

main "$@"
