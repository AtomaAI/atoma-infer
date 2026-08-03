# atoma-infer

<img src="assets/atoma-cover.png" alt="Atoma Logo" width="800"/>

`atoma-infer` is a Rust and CUDA project for large-language-model inference.

> **Learn more about Atoma:** Visit [atoma.ai](https://atoma.ai/) for information about Atoma's secure AI infrastructure platform.

## Status

The current implementation is under revival and is **not production-ready**. The repository does not yet have a verified build, test, or serving baseline. OpenAI API compatibility, model support, and single-node or distributed GPU topologies are not verified capabilities of this checkout. Do not rely on it for production workloads.

Rung 0 will restore a trustworthy build and test baseline before feature or performance claims are reintroduced. See the canonical [rung-0 specification](https://github.com/AtomaAI/atoma-infer/issues/147) for its bounded scope.

## Launch-gate parity target

Launch-gate parity is a future measured target, not a description of the current implementation. The gate requires DeepSeek-class goodput within 10% of the better of vLLM or SGLang, plus at least two outright headline benchmark wins. Measurements cover 2×8×H100 with FP8 and 8×B200 with NVFP4, in aggregated and prefill/decode-disaggregated topologies.

The canonical [revival decision map](https://github.com/AtomaAI/atoma-infer/issues/138) records the decisions and measurement destination. The [rung-0 specification](https://github.com/AtomaAI/atoma-infer/issues/147) defines the initial recovery work. These GitHub issues are the public sources of truth for the revival plan.

## Contributor setup

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/atoma-infer.git`.
3. Enter the checkout: `cd atoma-infer`.
4. Install Rust using [rustup](https://www.rust-lang.org/tools/install). The repository's toolchain file selects the required Rust version.
5. Initialize dependencies: `git submodule update --init --recursive`.

Run the GPU-less workspace checks with:

```shell
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Install [prek](https://prek.j178.dev/), then install and run the repository hooks with `prek install` and `prek run --all-files`.

## Configuration and running

Copy the example configuration and provide the Hugging Face API key, model, cache path, and GPU device IDs for your environment:

```shell
cp config.example.toml config.toml
```

`config.toml` is ignored by Git. The server requires the opt-in `cuda` feature and accepts the configuration path through `--config-path`:

```shell
RUST_LOG=info cargo run --release --features cuda --bin atoma-api -- --config-path config.toml
```

CUDA compilation and live GPU execution remain unverified until the remaining rung-0 CUDA checkpoints are complete.

## Contributing

Keep each pull request focused on one purpose, such as one bug fix, feature, or performance improvement. Unrelated changes belong in separate pull requests so each change can be reviewed independently.

A narrow exception applies to canonical roadmap integration pull requests. Such a pull request may integrate multiple planned changes only when it identifies their canonical tickets and preserves reviewable commit ranges for each ticket. This exception does not apply to unrelated cleanup or opportunistic changes.

Pull request descriptions should state the problem, the chosen approach, and the verification performed. Bug fixes and features should include tests at a public behavior seam. Performance changes should identify one bottleneck, describe the benchmark and hardware, and report speed and memory results.

## License

Licensed under the [Apache License 2.0](LICENSE).
