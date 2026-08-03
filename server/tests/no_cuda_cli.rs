#![cfg(not(feature = "cuda"))]

use std::process::Command;

#[test]
fn default_binary_exits_with_actionable_cuda_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_atoma-infer-server"))
        .output()
        .expect("server binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("CUDA support is disabled"));
    assert!(stderr.contains("--features cuda"));
}
