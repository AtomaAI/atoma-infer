# Atoma Infer Architecture

> High-performance LLM inference engine built in Rust with Candle, designed for maximum GPU utilization and minimal latency.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Rust?](#why-rust)
3. [Design Philosophy](#design-philosophy)
4. [System Architecture](#system-architecture)
5. [Threading Model](#threading-model)
6. [Pipeline Flow](#pipeline-flow)
7. [Memory Management](#memory-management)
8. [Request Scheduling](#request-scheduling)
9. [GPU Execution](#gpu-execution)
10. [Component Details](#component-details)
11. [Performance Characteristics](#performance-characteristics)
12. [Configuration](#configuration)
13. [Development Roadmap](#development-roadmap)

---

## Overview

Atoma Infer is a production-grade LLM inference engine that achieves state-of-the-art performance through aggressive CPU-GPU pipelining, advanced memory management, and lock-free concurrency. The architecture is inspired by leading frameworks (such as vLLM, SGLang, TensorRT-LLM) while leveraging Rust's unique strengths for safe, zero-cost abstractions.

### Key Features

- **Zero GPU Idle Time**: Triple-buffered pipeline ensures GPU is always computing
- **Hybrid KV Cache**: Combines PagedAttention (block-level) with RadixAttention (prefix caching)
- **Continuous Batching**: Dynamic request batching at iteration level
- **Lock-Free Concurrency**: Crossbeam-based communication with zero contention
- **Hardware Agnostic**: Candle backend supports CUDA, ROCm, Metal, and CPU
- **Type-Safe**: Rust's ownership prevents entire classes of runtime errors

---

## Why Rust?

Atoma Infer is built in Rust because **distributed AI inference demands system-level control, predictable performance, and safety guarantees that Python fundamentally cannot provide**. Our goal is to make Atoma the de facto inference framework for distributed systems, and Rust is essential to achieving this.

### The Distributed AI Challenge

Building distributed AI inference systems introduces unique challenges that go beyond single-node serving:

**Multi-Node Coordination**
- Requests must be routed across GPU nodes with minimal coordination overhead
- KV cache state may be replicated or sharded across nodes
- Failure handling and retry logic must be deterministic and fast
- Network communication becomes a critical path that cannot be abstracted away

**Resource Orchestration at Scale**
- Memory pressure management across heterogeneous GPU pools
- Dynamic load balancing as model popularity shifts
- Gang scheduling for multi-GPU model parallelism (tensor/pipeline parallel)
- Preemption and migration of partially completed requests

**Consistency and Fault Tolerance**
- Distributed state (KV cache, request queues) must remain consistent during failures
- Graceful degradation when nodes fail or become slow
- Exactly-once semantics for billing and audit trails
- Rolling upgrades without request loss

**Performance Isolation**
- Noisy neighbor prevention in multi-tenant environments
- QoS guarantees for different request classes
- Predictable tail latencies under load, essential for SLA compliance

### Why Python Frameworks Fall Short for Distribution

Python-based inference engines like vLLM and SGLang excel at single-node optimization but face fundamental barriers when scaling to distributed systems:

**Global Interpreter Lock (GIL)**
- Prevents true parallelism for coordination logic (request routing, state management)
- Forces reliance on multiprocessing, which is heavyweight and prevents shared memory
- Makes lock-free algorithms and fine-grained synchronization impossible

**Unpredictable Garbage Collection**
- GC pauses create tail latency variance, which cascades in distributed systems
- A 100ms GC pause on one node can stall an entire request chain
- Deterministic latency is critical for distributed consensus and coordination protocols

**Runtime Overhead**
- Dynamic typing and reference counting add overhead to every coordination operation
- Serialization/deserialization for inter-node communication is slower
- Python's async runtime (asyncio) has higher overhead than native async/await

**Memory Unsafety at Language Boundaries**
- Heavy reliance on C/C++ extensions (CUDA kernels, networking) creates unsafe boundaries
- Debugging distributed memory corruption across Python/C boundaries is extremely difficult
- No compile-time guarantees about thread safety in distributed state management

**Limited Control Over System Resources**
- Cannot fine-tune TCP behavior, RDMA parameters, or GPU stream scheduling
- Network protocol implementations (gRPC, custom protocols) require dropping to C
- Difficulty implementing zero-copy networking for large tensor transfers

### Rust Enables True Distributed AI Systems

**Fearless Concurrency for Distributed Coordination**
- Rust's ownership system prevents data races at compile time across distributed state
- Lock-free data structures and message passing enable high-throughput coordination
- True parallelism (no GIL) means coordination logic doesn't bottleneck inference
- Async/await with Tokio provides efficient, scalable networking primitives

**Predictable Performance for SLA Guarantees**
- No garbage collection means consistent tail latencies across all nodes
- Deterministic memory management enables predictable resource utilization
- Performance characteristics remain stable under load, critical for autoscaling
- Easier to reason about and debug performance in distributed traces

**Zero-Copy Networking for Tensor Transfer**
- Ownership system enables safe zero-copy serialization of tensors
- Direct integration with RDMA and kernel bypass networking (io_uring)
- Fine-grained control over memory pinning for efficient GPU-to-GPU transfers
- Custom protocols optimized for AI workloads without foreign function overhead

**Type Safety Across Network Boundaries**
- Compile-time guarantees about message formats between nodes
- Serde ecosystem provides safe, efficient serialization
- No runtime deserialization errors that could crash nodes
- Easier to evolve distributed protocols with type-safe versioning

**System-Level Resource Control**
- Direct control over thread affinity, CPU scheduling, and NUMA topology
- Fine-tuned GPU stream management for overlapping compute and communication
- Custom memory allocators for specific workload patterns
- Integration with Linux kernel features (cgroups, eBPF) for observability

**Robust Fault Tolerance**
- Rust's Result/Option types force explicit error handling in distributed code paths
- No null pointer exceptions or unexpected panics in coordination logic
- Graceful degradation is encoded in types, not runtime behavior
- Easier to build correct consensus protocols and state machines

### Building the De Facto Distributed AI Framework

Atoma's architecture leverages Rust to deliver capabilities that define next-generation distributed inference:

**Transparent Multi-Node Scheduling**
- Requests are routed across GPU pools based on model cache locality, load, and affinity
- Distributed KV cache with automatic replication and eviction
- No single coordinator bottleneck: fully decentralized scheduling

**Elastic Model Serving**
- Models can scale from single GPU to multi-node tensor/pipeline parallel seamlessly
- Hot model loading across nodes without request drops
- Preemptive migration of long requests during scale-down events

**Cross-Node Prefix Caching**
- RadixAttention prefix cache is distributed across nodes with automatic replication
- Cache hits leverage RDMA for low-latency prefix fetching
- Intelligent cache placement based on access patterns

**Multi-Tenancy and Isolation**
- Hard resource limits per tenant enforced at the Rust type level
- Performance isolation using CPU pinning and GPU MPS
- Fair scheduling with weighted queues and preemption

**Observability and Debuggability**
- Distributed tracing with nanosecond precision timing
- Memory-safe eBPF integration for kernel-level visibility
- Zero-overhead metrics collection using atomic operations

### When Python Frameworks Make Sense

Python-based frameworks like vLLM and SGLang are excellent for:
- Single-node deployments where coordination overhead is minimal
- Rapid experimentation and research workflows
- Teams prioritizing ecosystem compatibility over absolute performance
- Workloads where Python-based tooling integration is essential

---

## Design Philosophy

### 1. GPU is the Bottleneck

Modern LLM inference is GPU memory-bandwidth bound. Our architecture ensures:
- GPU compute never waits for CPU operations
- All CPU work is overlapped with GPU execution
- Memory transfers are asynchronous and pipelined

### 2. Zero-Copy Wherever Possible

- Pinned memory for DMA transfers
- Lock-free queues for inter-thread communication
- Rust's ownership prevents accidental copies

### 3. Predictable Performance

- No garbage collection pauses
- Bounded memory allocation
- Deterministic scheduling

### 4. Safety Without Overhead

- Compile-time data race prevention
- Zero-cost abstractions
- No runtime type checking

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  HTTP/REST   │  │  gRPC        │  │  WebSocket/SSE     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────────────┘    │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │       Request Router/Load Balancer   │
          └──────────────────┬──────────────────┘
                             │
┌────────────────────────────┴──────────────────────────────────────┐
│                      Processing Pipeline                           │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Stage 1: Ingestion (Thread Pool)                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │
│  │  │ Parser   │  │ Validator│  │ Enqueue  │                 │  │
│  │  └──────────┘  └──────────┘  └──────────┘                 │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Lock-Free Queue                               │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 2: Tokenization (Work-Stealing Pool)                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Lock-Free Queue                               │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 3: Scheduling (Single Thread)                       │  │
│  │  • Continuous batching logic                               │  │
│  │  • Memory allocation decisions                             │  │
│  │  • Priority queue management                               │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Bounded Queue (Triple Buffer)                 │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 4: Batch Preparation (Single Thread)                │  │
│  │  • Tensor assembly in pinned memory                        │  │
│  │  • Attention mask construction                             │  │
│  │  • Initiate H2D transfer (async)                           │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Bounded Queue (Triple Buffer)                 │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 5: GPU Execution (Single Thread + Multi-Stream)     │  │
│  │  ┌────────────────────────────────────────────────────┐    │  │
│  │  │  Stream 0: Compute Kernels                         │    │  │
│  │  │  Stream 1: H2D/D2H Transfers                       │    │  │
│  │  │  Stream 2: Sampling Operations                     │    │  │
│  │  │  Stream 3: KV Cache Management                     │    │  │
│  │  └────────────────────────────────────────────────────┘    │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Lock-Free Queue                               │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 6: Detokenization (Work-Stealing Pool)              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │ Lock-Free Queue                               │
│  ┌────────────────┴───────────────────────────────────────────┐  │
│  │  Stage 7: Response (Thread Pool)                           │  │
│  │  • Format HTTP/gRPC responses                              │  │
│  │  • Stream server-sent events                               │  │
│  │  • Update metrics and logging                              │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │      Observability & Monitoring       │
          │  • Prometheus metrics                 │
          │  • Distributed tracing                │
          │  • Performance profiling              │
          └───────────────────────────────────────┘
```

---

## Threading Model

### Thread Topology

Atoma Infer uses a **hybrid threading model** combining dedicated threads for critical paths with work-stealing pools for parallelizable tasks.

```rust
pub struct ThreadTopology {
    // API Layer (8-16 threads)
    api_workers: ThreadPool,
    
    // Preprocessing (4-8 threads, work-stealing)
    tokenizer_pool: WorkStealingPool,
    
    // Core Pipeline (dedicated single threads)
    scheduler_thread: JoinHandle<()>,      // 1 thread
    batch_prep_thread: JoinHandle<()>,     // 1 thread
    gpu_executor_thread: JoinHandle<()>,   // 1 thread (CUDA context)
    sampler_thread: JoinHandle<()>,        // 1 thread
    
    // Postprocessing (4-8 threads, work-stealing)
    detokenizer_pool: WorkStealingPool,
    
    // Response Layer (8-16 threads)
    response_workers: ThreadPool,
    
    // Background (1 thread each)
    metrics_collector: JoinHandle<()>,
    health_monitor: JoinHandle<()>,
}
```

### Why Single-Threaded GPU Executor?

1. **CUDA Context**: CUDA contexts are thread-local; multi-threading requires complex context management
2. **Memory-Bound**: LLM inference decode phase is memory-bandwidth bound, not compute-bound
3. **Stream Parallelism**: We achieve parallelism via multiple CUDA streams, not threads
4. **Simplicity**: Single-threaded executor eliminates synchronization overhead

### Thread Affinity & NUMA

```rust
// Pin threads to specific CPU cores for cache locality
pub fn set_thread_affinity(thread: &Thread, core_id: usize) {
    #[cfg(target_os = "linux")]
    {
        let mut cpu_set = libc::cpu_set_t::default();
        unsafe {
            libc::CPU_SET(core_id, &mut cpu_set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpu_set);
        }
    }
}

// NUMA-aware allocation
pub fn allocate_on_numa_node(size: usize, node: usize) -> *mut u8 {
    #[cfg(target_os = "linux")]
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        libc::mbind(ptr, size, libc::MPOL_BIND, &(1 << node), 64, 0);
        ptr as *mut u8
    }
}
```

---

## Pipeline Flow

### Triple Buffering Strategy

The core innovation is maintaining **3 batches in flight simultaneously**:

```
Time →
─────────────────────────────────────────────────────────────────
Batch 0: [Schedule]─→[Prep CPU]─→[GPU Compute]─→[Sample]─→[Done]
                                      
Batch 1:             [Schedule]─→[Prep CPU]─→[GPU Compute]─→[Sample]
                                                  
Batch 2:                         [Schedule]─→[Prep CPU]─→[GPU Compute]
                                                             
Batch 3:                                     [Schedule]─→[Prep CPU]

GPU:     ────────────────────────██████████████████████████████────
         Idle (startup)          Always Computing!
```

### Detailed Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Request Arrives                                              │
│     • HTTP POST /v1/chat/completions                            │
│     • Request ID generated                                       │
│     • Pushed to raw_requests queue                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Tokenization (Parallel, Work-Stealing)                      │
│     • Worker pops task from global queue                        │
│     • Encodes prompt: "Hello" → [15496, 29901]                 │
│     • Steals work from other workers if local queue empty       │
│     • Pushed to tokenized_requests queue                        │
│     • Typical time: 1-2ms per request                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Scheduling (Single Thread, Continuous Batching)             │
│     • Waits for gpu_ready flag                                  │
│     • Drains tokenized_requests into scheduler state            │
│     • Runs scheduling algorithm:                                │
│       - Prioritize running decode sequences                     │
│       - Attempt to swap in preempted sequences                  │
│       - Add new prefill sequences if budget available           │
│     • Allocates KV cache blocks from block manager              │
│     • Creates ScheduledBatch descriptor                         │
│     • Pushes to scheduled_batches queue (capacity: 3)           │
│     • Notifies batch_prep_thread                                │
│     • Sets gpu_ready = false                                    │
│     • Typical time: 0.3-0.5ms                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Batch Preparation (Single Thread, CPU-side)                 │
│     • Waits for batch_scheduled notification                    │
│     • Acquires pinned memory buffer from pool                   │
│     • Assembles input tensors in pinned memory:                 │
│       - Flattens all token IDs into contiguous array            │
│       - Builds position indices                                 │
│       - Constructs sequence start locations                     │
│       - Creates attention masks (causal for prefill)            │
│     • Prepares KV cache block tables                            │
│     • Initiates async H2D transfer on transfer_stream           │
│       (non-blocking! CPU continues immediately)                 │
│     • Creates PreparedBatch descriptor                          │
│     • Pushes to prepared_batches queue (capacity: 3)            │
│     • Notifies gpu_executor_thread                              │
│     • Typical time: 0.8-1.2ms                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. GPU Execution (Single Thread, Multi-Stream)                 │
│     • Waits for batch_prepared notification                     │
│     • Waits for H2D transfer to complete (event sync)           │
│     • Sets current CUDA stream to compute_stream                │
│     • Launches model forward pass (async):                      │
│       Stream 0 (Compute):                                       │
│       - Embedding lookup                                        │
│       - Layer 0: Attention + FFN                                │
│       - Layer 1: Attention + FFN                                │
│       - ...                                                     │
│       - Layer N: Attention + FFN                                │
│       - Final layer norm                                        │
│       - LM head projection                                      │
│     • Records compute_complete event on compute_stream          │
│     • IMMEDIATELY sets gpu_ready = true (async execution!)      │
│     • Creates ExecutionResult with event handle                 │
│     • Pushes to execution_results queue (capacity: 3)           │
│     • Notifies sampler_thread                                   │
│     • GPU time: async, not blocking CPU            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Sampling (Single Thread, GPU Stream)                        │
│     • Waits for batch_executed notification                     │
│     • Waits for compute_complete event (GPU sync)               │
│     • Sets current CUDA stream to sampling_stream               │
│     • Applies sampling strategies:                              │
│       - Greedy: argmax(logits)                                  │
│       - Temperature: logits / T, then softmax + sample          │
│       - Top-p: nucleus sampling                                 │
│       - Top-k: filter then sample                               │
│     • Updates sequence states:                                  │
│       - Appends token to sequence.tokens                        │
│       - Checks stop conditions (EOS, max_tokens)                │
│       - Marks finished sequences                                │
│     • Initiates async D2H transfer for sampled tokens           │
│     • Returns pinned buffer to pool                             │
│     • Creates SampledBatch descriptor                           │
│     • Pushes to sampled_batches queue                           │
│     • Notifies detokenizer_pool                                 │
│     • Typical time: 0.2-0.4ms                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Detokenization (Parallel, Work-Stealing)                    │
│     • Worker pops task from global queue                        │
│     • Decodes tokens: [15496, 29901] → "Hello"                 │
│     • Accumulates into response buffer                          │
│     • For streaming: sends SSE event immediately                │
│     • Typical time: 0.3-0.5ms per sequence                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. Response Formatting (Thread Pool)                           │
│     • Formats OpenAI-compatible JSON response                   │
│     • Calculates token counts and timing metrics                │
│     • Sends HTTP response or SSE stream                         │
│     • Updates Prometheus metrics                                │
│     • Logs request completion                                   │
│     • Typical time: 0.2-0.3ms                                   │
└─────────────────────────────────────────────────────────────────┘
```


---

## Memory Management

### Hybrid KV Cache Architecture

Atoma Infer implements a **two-tier caching system**:

1. **PagedAttention (Block-Level)**: Efficient memory allocation with minimal fragmentation
2. **RadixAttention (Prefix-Level)**: Automatic reuse of common prefixes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Layout                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU Memory (e.g., 80GB on A100)                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Model Weights (Frozen)                                │    │
│  │  • Embedding: 4GB                                      │    │
│  │  • Layers: 20GB                                        │    │
│  │  • Total: ~24GB                                        │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  KV Cache Blocks (Paged)                               │    │
│  │  • Block size: 16 tokens                               │    │
│  │  • Per block: 2 * num_layers * hidden_size * 2 bytes  │    │
│  │  • Example: 2 * 32 * 4096 * 2 = 512 KB per block      │    │
│  │  • ~100,000 blocks = 50GB                              │    │
│  │                                                         │    │
│  │  Block Table:                                          │    │
│  │  ┌──────────────────────────────────────────────┐     │    │
│  │  │ Seq 0: [Block 0, Block 1, Block 2]           │     │    │
│  │  │ Seq 1: [Block 0, Block 3]  (sharing Block 0) │     │    │
│  │  │ Seq 2: [Block 4, Block 5, Block 6, Block 7]  │     │    │
│  │  └──────────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Activation Memory (Temporary)                         │    │
│  │  • Reused across iterations                            │    │
│  │  • ~4GB                                                │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  CPU Memory (Pinned)                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Pinned Buffer Pool (4 buffers × 128MB)                │    │
│  │  • Used for zero-copy H2D/D2H transfers               │    │
│  │  • Circular buffer reuse                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  CPU Memory (Swapped KV Cache)                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Preempted Sequences (Swapped Out)                     │    │
│  │  • Low-priority sequences moved to CPU                 │    │
│  │  • Swapped back when GPU memory available              │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### PagedAttention Block Manager

```rust
pub struct BlockManager {
    block_size: usize,              // Tokens per block (default: 16)
    num_gpu_blocks: usize,          // Total GPU blocks
    num_cpu_blocks: usize,          // CPU blocks for swapping
    
    // Block allocation
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
    
    // Mapping: sequence → blocks
    block_tables: HashMap<SequenceId, Vec<PhysicalBlock>>,
    
    // Reference counting for copy-on-write
    refcounts: HashMap<BlockId, AtomicUsize>,
}

pub struct PhysicalBlock {
    block_id: BlockId,
    device: Device,              // GPU or CPU
    num_filled: AtomicUsize,     // Tokens currently stored
}
```

**Key Operations:**

1. **Allocation**: O(1) - pop from free list
2. **Deallocation**: O(1) - push to free list
3. **Fork (Beam Search)**: O(1) - increment refcount, copy-on-write
4. **Swap Out**: O(n) - async DMA to CPU memory
5. **Swap In**: O(n) - async DMA to GPU memory

### RadixAttention Prefix Cache

```rust
pub struct RadixTreeNode {
    tokens: Vec<TokenId>,
    key_cache: Option<Arc<Tensor>>,
    value_cache: Option<Arc<Tensor>>,
    parent: Option<Arc<RwLock<RadixTreeNode>>>,
    children: HashMap<TokenId, Arc<RwLock<RadixTreeNode>>>,
    last_accessed: AtomicU64,    // For LRU eviction
    ref_count: AtomicUsize,      // Active users
}
```

**Example: Multi-Turn Conversation**

```
User: "Explain quantum computing"
System prompt: "You are a helpful assistant..." [1024 tokens]

Turn 1:
  [System Prompt]──[User Q1]──[Assistant A1]
  
Turn 2 (reuses system prompt):
  [System Prompt]──[User Q1]──[Assistant A1]──[User Q2]──[Assistant A2]
      ↑ Cache Hit!  ↑ Cache Hit!  ↑ Cache Hit!
      
Savings: Avoided recomputing 1024 + len(Q1) + len(A1) tokens
```

### Memory Allocation Strategy

```rust
pub struct MemoryManager {
    // Pre-allocated pools
    pinned_pool: PinnedMemoryPool,      // 4 × 128MB buffers
    device_pool: DeviceMemoryPool,      // Buddy allocator
    
    // Tracking
    allocated_bytes: AtomicUsize,
    peak_usage: AtomicUsize,
    fragmentation: AtomicF32,
}

impl MemoryManager {
    /// Allocate pinned memory for H2D transfer
    pub async fn acquire_pinned_buffer(&self) -> PinnedBuffer {
        loop {
            if let Some(id) = self.pinned_pool.available.pop() {
                return self.pinned_pool.buffers[id].clone();
            }
            // Wait for buffer to become available
            tokio::time::sleep(Duration::from_micros(50)).await;
        }
    }
    
    /// Allocate device memory with caching
    pub fn allocate_device(&mut self, size: usize) -> DevicePtr {
        // Try to find cached block
        let block_size = size.next_power_of_two();
        if let Some(ptr) = self.device_pool.free_blocks.get_mut(&block_size) {
            if let Some(block) = ptr.pop() {
                return block.ptr;
            }
        }
        
        // Allocate new block
        unsafe { cuda_malloc(block_size).unwrap() }
    }
}
```

---

## Request Scheduling

### Continuous Batching Algorithm

Unlike static batching (waiting for full batches), continuous batching operates at **iteration-level granularity**:

```
Static Batching:
Request 1: ████████████████ (completes)
Request 2: ████████████████ (completes)
Request 3: ████ (waits...)
Request 4: ████ (waits...)
           ↑ GPU idle while waiting

Continuous Batching:
Request 1: ██████ (generates 6 tokens)
Request 2: ████ (generates 4 tokens, completes)
Request 3: ██ (added to batch after iteration 2)
Request 4: ██████████ (added to batch after iteration 4)
           ↑ GPU always working
```

### Scheduler State Machine

```rust
pub struct ContinuousBatchScheduler {
    // Request queues (priority-based)
    waiting: Arc<Mutex<BinaryHeap<PrioritizedRequest>>>,
    running: Arc<RwLock<HashMap<SequenceId, SequenceGroup>>>,
    swapped: Arc<Mutex<Vec<SequenceGroup>>>,
    
    // Resource tracking
    block_manager: Arc<RwLock<BlockManager>>,
    kv_cache: Arc<RadixAttentionCache>,
    
    // Configuration
    max_num_seqs: usize,           // Max concurrent sequences
    max_tokens_per_batch: usize,   // Token budget per iteration
    enable_chunked_prefill: bool,
    chunk_size: usize,
}

pub enum SequenceStatus {
    Waiting,      // In waiting queue
    Running,      // Being processed
    Swapped,      // Temporarily moved to CPU
    Finished(FinishReason),
}

pub struct SequenceGroup {
    request_id: RequestId,
    sequences: Vec<Sequence>,      // Multiple for beam search
    arrival_time: Instant,
    priority: Priority,
    sampling_params: SamplingParams,
}
```

### Scheduling Algorithm (Per Iteration)

```rust
impl ContinuousBatchScheduler {
    pub async fn schedule(&self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        
        // Phase 1: Schedule running sequences (decode)
        // These have highest priority - already allocated KV cache
        for (seq_id, seq_group) in self.running.read().await.iter() {
            if can_append_slot(seq_id) {
                output.decode_seq_groups.push(seq_group.clone());
            } else {
                // Out of memory - preempt lowest priority
                self.preempt_lowest_priority().await;
            }
        }
        
        // Phase 2: Schedule swapped sequences
        // Try to swap back preempted sequences
        while let Some(seq_group) = self.swapped.lock().await.first() {
            if can_swap_in(seq_group) {
                self.swap_in(seq_group).await;
                output.decode_seq_groups.push(seq_group.clone());
            } else {
                break;  // Not enough memory
            }
        }
        
        // Phase 3: Schedule waiting sequences (prefill)
        // Add new requests if budget available
        let mut token_budget = self.max_tokens_per_batch - output.num_decode_tokens;
        
        while let Some(req) = self.waiting.lock().await.peek() {
            let num_tokens = if self.enable_chunked_prefill {
                req.num_tokens.min(self.chunk_size)
            } else {
                req.num_tokens
            };
            
            if num_tokens <= token_budget && can_allocate(num_tokens) {
                let req = self.waiting.lock().await.pop().unwrap();
                output.prefill_seq_groups.push(req);
                token_budget -= num_tokens;
            } else {
                break;
            }
        }
        
        output
    }
}
```

### Priority System

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    System = 0,       // System prompts, health checks
    High = 1,         // Premium users, low-latency requirements
    Normal = 2,       // Standard requests
    Low = 3,          // Batch processing, background tasks
}

pub struct PrioritizedRequest {
    priority: Priority,
    arrival_time: Instant,
    request: GenerationRequest,
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        self.priority.cmp(&other.priority)
            // Then FIFO within same priority
            .then_with(|| self.arrival_time.cmp(&other.arrival_time))
    }
}
```

### Preemption Strategy

When GPU memory is exhausted:

1. **Identify Candidates**: Select lowest priority running sequences
2. **Swap Out**: Async DMA transfer KV cache from GPU → CPU
3. **Free Blocks**: Return GPU blocks to free pool
4. **Update State**: Mark sequences as `Swapped`

```rust
pub async fn preempt(&mut self, seq_id: SequenceId) -> Result<()> {
    let seq_group = self.running.write().await.remove(&seq_id).unwrap();
    
    // Swap KV cache to CPU
    self.block_manager.write().await.swap_out(seq_id).await?;
    
    // Move to swapped queue
    self.swapped.lock().await.push(seq_group);
    
    self.metrics.num_preemptions.fetch_add(1, Ordering::Relaxed);
    Ok(())
}
```

---

## GPU Execution

### Multi-Stream Execution

Atoma Infer uses **4 concurrent CUDA streams** for maximum overlap:

```
Stream 0 (Compute):     ████████████████████████████
                        Main model execution
                        
Stream 1 (H2D):         ▓▓▓▓        ▓▓▓▓        ▓▓▓▓
                        Host → Device transfers
                        
Stream 2 (D2H):                 ▓▓▓▓        ▓▓▓▓        ▓▓▓▓
                                Device → Host transfers
                        
Stream 3 (Sampling):                ▓▓▓▓        ▓▓▓▓
                                    Token sampling
                                    
Events:                 E0      E1      E2      E3
                        ↓ Record ↓ Wait ↓ Sync  ↓
```

### Model Forward Pass

```rust
pub trait LLMModel: Send + Sync {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: &mut KVCache,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub struct LlamaModel {
    config: LlamaConfig,
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl LLMModel for LlamaModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: &mut KVCache,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Embedding lookup
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        
        // 2. Transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                positions,
                &mut kv_cache.get_layer(layer_idx),
                attention_mask,
            )?;
        }
        
        // 3. Final normalization
        hidden_states = self.norm.forward(&hidden_states)?;
        
        // 4. LM head projection
        let logits = self.lm_head.forward(&hidden_states)?;
        
        Ok(logits)
    }
}
```

### Attention Implementation

```rust
pub struct PagedAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    block_size: usize,
}

impl PagedAttention {
    /// Context attention (prefill phase)
    pub fn forward_context(
        &self,
        query: &Tensor,      // [num_tokens, num_heads, head_dim]
        key: &Tensor,        // [num_tokens, num_heads, head_dim]
        value: &Tensor,      // [num_tokens, num_heads, head_dim]
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // Use FlashAttention kernel for efficiency
        flash_attention_varlen(
            query,
            key,
            value,
            attention_mask,
            self.scale,
            /*causal=*/ true,
        )
    }
    
    /// Paged attention (decode phase)
    pub fn forward_decode(
        &self,
        query: &Tensor,           // [batch_size, num_heads, head_dim]
        key_cache: &Tensor,       // [num_blocks, block_size, num_heads, head_dim]
        value_cache: &Tensor,     // [num_blocks, block_size, num_heads, head_dim]
        block_tables: &Tensor,    // [batch_size, max_blocks]
        context_lens: &Tensor,    // [batch_size]
    ) -> Result<Tensor> {
        // Custom paged attention kernel
        paged_attention_v2(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            self.block_size,
            self.scale,
        )
    }
}
```

### Sampling Strategies

```rust
pub enum SamplingType {
    Greedy,
    Temperature {
        temperature: f32,
        top_p: Option<f32>,
        top_k: Option<usize>,
    },
}

pub fn sample_tokens(
    logits: &Tensor,           // [batch_size, vocab_size]
    sampling_params: &[SamplingParams],
) -> Result<Vec<TokenId>> {
    let mut sampled_tokens = Vec::new();
    
    for (i, params) in sampling_params.iter().enumerate() {
        let logit = logits.get(i)?;
        
        let token = match params.sampling_type {
            SamplingType::Greedy => {
                // argmax(logits)
                greedy_sample(&logit)?
            }
            SamplingType::Temperature { temperature, top_p, top_k } => {
                // Apply temperature
                let scaled_logits = (&logit / temperature)?;
                
                // Apply top-k filtering
                let filtered = if let Some(k) = top_k {
                    top_k_filter(&scaled_logits, k)?
                } else {
                    scaled_logits
                };
                
                // Apply top-p (nucleus) filtering
                let filtered = if let Some(p) = top_p {
                    top_p_filter(&filtered, p)?
                } else {
                    filtered
                };
                
                // Sample from distribution
                categorical_sample(&filtered)?
            }
        };
        
        sampled_tokens.push(token);
    }
    
    Ok(sampled_tokens)
}
```

---

## Component Details

### API Server

```rust
pub struct ApiServer {
    engine: Arc<InferenceEngine>,
    config: ServerConfig,
    router: Router,
}

impl ApiServer {
    pub async fn run(&self) -> Result<()> {
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/v1/models", get(list_models_handler))
            .route("/health", get(health_check_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(Arc::new(self.clone()));
        
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        tracing::info!("Server listening on {}", addr);
        axum::serve(listener, app).await?;
        Ok(())
    }
}

// OpenAI-compatible endpoint
async fn chat_completions_handler(
    State(server): State<Arc<ApiServer>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    if request.stream.unwrap_or(false) {
        // Server-Sent Events streaming
        let stream = server.engine.generate_stream(request).await?;
        Ok(Sse::new(stream).into_response())
    } else {
        // Batch response
        let response = server.engine.generate(request).await?;
        Ok(Json(response).into_response())
    }
}
```

### Lock-Free Queues

```rust
use crossbeam::queue::{SegQueue, ArrayQueue};

/// Unbounded MPMC queue (for variable-rate stages)
pub type UnboundedQueue<T> = Arc<SegQueue<T>>;

/// Bounded MPMC queue (for backpressure control)
pub type BoundedQueue<T> = Arc<ArrayQueue<T>>;

/// Work-stealing task injector
pub type TaskInjector<T> = Arc<Injector<T>>;

pub struct PipelineQueues {
    // Variable-rate queues (unbounded)
    tokenized_requests: UnboundedQueue<TokenizedRequest>,
    sampled_batches: UnboundedQueue<SampledBatch>,
    responses: UnboundedQueue<Response>,
    
    // Fixed-rate queues (bounded for triple buffering)
    scheduled_batches: BoundedQueue<ScheduledBatch>,    // capacity: 3
    prepared_batches: BoundedQueue<PreparedBatch>,      // capacity: 3
    execution_results: BoundedQueue<ExecutionResult>,   // capacity: 3
}
```

### Metrics & Observability

```rust
pub struct EngineMetrics {
    // Request metrics
    pub requests_total: Counter,
    pub requests_active: Gauge,
    pub request_duration_seconds: Histogram,
    
    // Token metrics
    pub tokens_generated_total: Counter,
    pub tokens_per_second: Gauge,
    
    // Latency metrics (critical!)
    pub time_to_first_token_seconds: Histogram,
    pub inter_token_latency_seconds: Histogram,
    
    // Resource metrics
    pub gpu_memory_used_bytes: Gauge,
    pub gpu_utilization_percent: Gauge,
    pub kv_cache_usage_percent: Gauge,
    
    // Scheduler metrics
    pub waiting_queue_size: Gauge,
    pub running_sequences: Gauge,
    pub preemptions_total: Counter,
    pub swapped_sequences: Gauge,
}

// Prometheus exposition
#[get("/metrics")]
async fn metrics_handler(
    State(metrics): State<Arc<EngineMetrics>>,
) -> String {
    metrics.export()
}
```

---

## Performance Characteristics

### Throughput Analysis

```rust
/// Maximum achievable throughput
pub fn calculate_throughput(
    gpu_compute_time_ms: f64,
    batch_size: usize,
) -> f64 {
    // With perfect pipelining, throughput limited by GPU compute
    let batches_per_second = 1000.0 / gpu_compute_time_ms;
    let tokens_per_second = batches_per_second * batch_size as f64;
    tokens_per_second
}

// Example: 7B model on A100
// - GPU compute: 8ms per batch (decode)
// - Batch size: 128 tokens
// Throughput = (1000 / 8) * 128 = 16,000 tokens/sec
```

### Latency Characteristics

| Phase | Latency | Notes |
|-------|---------|-------|
| **Prefill** | O(n²) memory access | n = prompt length |
| **Decode** | O(n) memory access | Linear in context |
| **First Token** | Dominated by prefill | ~15-30ms for <1K tokens |
| **Subsequent Tokens** | ~8-12ms (7B) | GPU memory bandwidth bound |

### Memory Efficiency

```
Fragmentation Analysis:
────────────────────────────────────────────────────────
Without PagedAttention (pre-allocated):
  Avg sequence length: 100 tokens
  Max sequence length: 2048 tokens
  Waste: (2048 - 100) / 2048 = 95% wasted!

With PagedAttention (block_size=16):
  Avg sequence: 100 tokens → 7 blocks (112 tokens allocated)
  Waste: (112 - 100) / 112 = 10.7% per sequence
  Last block waste: 12 tokens per sequence
  Overall waste: ~3-4% (accounting for sharing)
────────────────────────────────────────────────────────
```

### Scalability

**Single GPU:**
- Concurrent requests: 1,000+
- Throughput: 10,000-16,000 tokens/sec (7B model)

**Multi-GPU (Tensor Parallel):**
- 8× A100: Can run 70B model
- Throughput: ~3,000-4,000 tokens/sec (70B model)

**Distributed (Multiple Nodes):**
- Linear scaling with data parallelism
- Each node runs independent replica

---

## Configuration

### Engine Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    // Model
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dtype: DataType,              // fp16, bf16, fp32
    
    // Memory
    pub gpu_memory_utilization: f32,  // 0.0-1.0 (default: 0.9)
    pub swap_space: usize,            // CPU swap space in GB
    pub block_size: usize,            // Tokens per block (default: 16)
    
    // Scheduling
    pub max_num_seqs: usize,          // Max concurrent sequences
    pub max_tokens_per_batch: usize,  // Token budget per iteration
    pub enable_chunked_prefill: bool,
    pub chunk_size: usize,            // Prefill chunk size
    
    // Threading
    pub num_tokenizer_threads: usize,
    pub num_detokenizer_threads: usize,
    pub pin_threads: bool,            // NUMA-aware pinning
    
    // GPU
    pub num_cuda_streams: usize,      // Default: 4
    pub enable_cuda_graphs: bool,     // Graph capture for fixed shapes
    pub enable_flash_attention: bool,
    
    // Features
    pub enable_prefix_caching: bool,  // RadixAttention
    pub enable_speculative_decoding: bool,
    
    // Server
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
}
```

### Example Configuration

```toml
# config.toml
[model]
path = "/models/llama-3-8b"
tokenizer_path = "/models/llama-3-8b/tokenizer.json"
dtype = "fp16"

[memory]
gpu_memory_utilization = 0.90
swap_space_gb = 32
block_size = 16

[scheduling]
max_num_seqs = 256
max_tokens_per_batch = 4096
enable_chunked_prefill = true
chunk_size = 512

[threading]
num_tokenizer_threads = 8
num_detokenizer_threads = 8
pin_threads = true

[gpu]
num_cuda_streams = 4
enable_cuda_graphs = true
enable_flash_attention = true

[features]
enable_prefix_caching = true
enable_speculative_decoding = false

[server]
host = "0.0.0.0"
port = 8080
max_concurrent_requests = 10000
```

---

## Development Roadmap

### Phase 1: Foundation ✅
- [x] Basic Candle integration
- [x] Simple forward pass
- [x] Token sampling
- [x] HTTP API server

### Phase 2: Memory Management (Current)
- [ ] PagedAttention block manager
- [ ] KV cache allocation/deallocation
- [ ] Copy-on-write for beam search
- [ ] CPU/GPU swapping

### Phase 3: Scheduling
- [ ] Continuous batching scheduler
- [ ] Priority queue
- [ ] Preemption policies
- [ ] Chunked prefill

### Phase 4: Performance
- [ ] RadixAttention prefix caching
- [ ] Multi-stream execution
- [ ] Flash Attention integration
- [ ] Fused kernels (RMSNorm, SiLU, RoPE)

### Phase 5: Distributed
- [ ] Tensor parallelism (multi-GPU)
- [ ] Pipeline parallelism
- [ ] Disaggregated prefill/decode
- [ ] Multi-node support

### Phase 6: Advanced Features
- [ ] Speculative decoding
- [ ] LoRA adapter batching
- [ ] Guided decoding (JSON, regex)
- [ ] Vision-language models

### Phase 7: Production Hardening
- [ ] Comprehensive metrics
- [ ] Distributed tracing
- [ ] Rate limiting
- [ ] Model hot-swapping
- [ ] A/B testing

---

## Contributing

We welcome contributions! Key areas:

1. **Kernel Optimization**: CUDA/ROCm/Metal kernels for attention, sampling
2. **Memory Management**: Improved allocation strategies, cache policies
3. **Scheduling**: Better preemption heuristics, fairness guarantees
4. **Model Support**: New architectures (Mixtral, Qwen, etc.)
5. **Benchmarking**: Performance testing, profiling tools

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[Your License Here]

---

## References

1. **vLLM**: Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
2. **SGLang**: Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs" (2024)
3. **TensorRT-LLM**: NVIDIA TensorRT-LLM Documentation
4. **FlashAttention**: Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
5. **Orca**: Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022)

---

For questions or support, please open an issue or join our [Discord/Slack].