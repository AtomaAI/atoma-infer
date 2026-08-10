//! The capture arena: engine-owned addresses for every captured step's activations.
//!
//! One arena is shared by every bucket — buckets never replay concurrently, so the activation
//! footprint is the largest bucket's, not the sum across the bucket ladder. A slot's address is a
//! pure function of (bucket, layer, role): every bucket bumps from the same base, so address
//! stability across replays is guaranteed by construction and needs no pinning, weak references,
//! or write-back copies (ADR 0004).
//!
//! Addresses are exposed only through [`CaptureArena::offset`] — never an open-coded formula at a
//! call site — so a smarter allocation plan (e.g. layer-parity ping-pong) can replace the naive
//! one without touching any caller. The arena has no model knowledge: roles enter as
//! caller-declared per-token widths and the bucket ladder as uninterpreted entries.

/// Index into the bucket ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketIdx(pub usize);

/// Zero-based model layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerIdx(pub usize);

/// Index into the caller-declared role table.
///
/// A role names one activation tensor produced per layer (the caller decides what that means);
/// the arena only knows its per-token width in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRole(pub usize);

/// Every slot address is aligned to this many bytes, matching the alignment CUDA guarantees for
/// device allocations, so a slot can back any tensor a kernel would otherwise get from
/// `cuMemAlloc`.
pub const SLOT_ALIGN: usize = 256;

/// Element type of an activation, used only to turn per-token element counts into byte widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
    F8,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F8 => 1,
        }
    }

    /// Per-token width in bytes of a role holding `elements_per_token` elements of this type.
    pub fn width_bytes(self, elements_per_token: usize) -> usize {
        elements_per_token * self.size_in_bytes()
    }
}

/// Address layout for every captured step's activations. See the module docs for the invariants.
#[derive(Debug, Clone)]
pub struct CaptureArena {
    num_layers: usize,
    /// Per-token width in bytes of each role, indexed by [`TensorRole`].
    role_widths: Vec<usize>,
    /// Tokens per bucket, indexed by [`BucketIdx`]. Uninterpreted: never sorted, deduplicated,
    /// or assumed monotonic.
    ladder: Vec<usize>,
}

impl CaptureArena {
    /// Builds the arena layout from `num_layers` layers, caller-declared per-token role widths
    /// in bytes, and the bucket ladder in tokens per bucket.
    pub fn new(num_layers: usize, role_widths: &[usize], ladder: &[usize]) -> Self {
        Self {
            num_layers,
            role_widths: role_widths.to_vec(),
            ladder: ladder.to_vec(),
        }
    }

    /// Byte offset of the slot holding `role`'s activation for `layer` under `bucket`.
    ///
    /// One slot per layer, no liveness-based reuse: a wrong reuse decision cannot corrupt
    /// numbers silently inside a replay (ADR 0004).
    pub fn offset(&self, bucket: BucketIdx, layer: LayerIdx, role: TensorRole) -> usize {
        assert!(
            layer.0 < self.num_layers,
            "layer index {} out of range: the arena was built for {} layers",
            layer.0,
            self.num_layers
        );
        assert!(
            role.0 < self.role_widths.len(),
            "role index {} out of range: {} roles were declared",
            role.0,
            self.role_widths.len()
        );
        let per_layer: usize = self.layer_extent(bucket);
        let within_layer: usize = (0..role.0)
            .map(|r| self.slot_size(bucket, TensorRole(r)))
            .sum();
        layer.0 * per_layer + within_layer
    }

    /// Size in bytes of the slot holding `role`'s activation under `bucket`.
    pub fn slot_size(&self, bucket: BucketIdx, role: TensorRole) -> usize {
        let tokens = self.ladder[bucket.0];
        (self.role_widths[role.0] * tokens).next_multiple_of(SLOT_ALIGN)
    }

    /// Total bytes `bucket` addresses: every layer's slots for every role.
    pub fn bucket_footprint(&self, bucket: BucketIdx) -> usize {
        self.num_layers * self.layer_extent(bucket)
    }

    /// Bytes of device memory backing the arena: the largest bucket's footprint, since every
    /// bucket bumps from the same base and buckets never replay concurrently.
    pub fn total_size(&self) -> usize {
        (0..self.ladder.len())
            .map(|b| self.bucket_footprint(BucketIdx(b)))
            .max()
            .unwrap_or(0)
    }

    fn layer_extent(&self, bucket: BucketIdx) -> usize {
        (0..self.role_widths.len())
            .map(|r| self.slot_size(bucket, TensorRole(r)))
            .sum()
    }
}

/// The workspace-ownership contract for kernels: the caller owns a preallocated workspace, and
/// the kernel may not allocate.
///
/// A kernel that allocates inside a captured region invalidates the capture — or bakes a
/// pool-owned address into the graph — so allocation-freedom is the backend's contract with
/// this crate. A kernel-call descriptor implements this trait to declare how much caller-owned
/// workspace one invocation needs; the caller allocates at least that much before capture
/// (a [`GraphEntry`](crate::graph_entry::GraphEntry) workspace buffer) and hands it to every
/// launch. Defined here, implemented by the attention backend that will exist later; the
/// current FlashAttention-2 wrapper deliberately does not implement it.
pub trait WorkspaceRequirement {
    /// Bytes of caller-owned workspace one invocation of this call needs.
    fn workspace_bytes(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two roles at 100 and 300 bytes per token; slot sizes below are hand-computed:
    /// bucket 0 (2 tokens): 200 -> 256, 600 -> 768 (layer extent 1024);
    /// bucket 1 (8 tokens): 800 -> 1024, 2400 -> 2560 (layer extent 3584).
    fn two_role_arena() -> CaptureArena {
        CaptureArena::new(2, &[100, 300], &[2, 8])
    }

    #[test]
    fn offsets_match_hand_computed_layout() {
        let arena = two_role_arena();

        assert_eq!(arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(0)), 0);
        assert_eq!(arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(1)), 256);
        assert_eq!(arena.offset(BucketIdx(0), LayerIdx(1), TensorRole(0)), 1024);
        assert_eq!(arena.offset(BucketIdx(0), LayerIdx(1), TensorRole(1)), 1280);

        assert_eq!(arena.offset(BucketIdx(1), LayerIdx(0), TensorRole(1)), 1024);
        assert_eq!(arena.offset(BucketIdx(1), LayerIdx(1), TensorRole(0)), 3584);
        assert_eq!(arena.offset(BucketIdx(1), LayerIdx(1), TensorRole(1)), 4608);
    }

    #[test]
    fn total_size_is_the_largest_bucket_footprint() {
        let arena = two_role_arena();

        assert_eq!(arena.bucket_footprint(BucketIdx(0)), 2048);
        assert_eq!(arena.bucket_footprint(BucketIdx(1)), 7168);
        assert_eq!(arena.total_size(), 7168);
    }

    #[test]
    fn footprint_from_layer_count_widths_dtype_and_ladder() {
        // An 8B-class decode step: 32 layers, hidden width 4096 and intermediate width 14336 in
        // bf16 (8192 and 28672 bytes per token). At bucket 64 the slots are 524288 and 1835008
        // bytes (both already 256-aligned), so one layer is 2359296 bytes and 32 layers are
        // 75497472 — the number that answers "does this fit in the memory budget".
        let roles = [
            Dtype::Bf16.width_bytes(4096),
            Dtype::Bf16.width_bytes(14336),
        ];
        let arena = CaptureArena::new(32, &roles, &[1, 8, 64]);

        assert_eq!(arena.total_size(), 75_497_472);
        assert_eq!(arena.total_size(), arena.bucket_footprint(BucketIdx(2)));
    }

    #[test]
    fn total_size_does_not_grow_when_smaller_buckets_are_added() {
        let largest_alone = CaptureArena::new(2, &[100, 300], &[8]);
        let with_smaller = CaptureArena::new(2, &[100, 300], &[2, 8, 4]);

        assert_eq!(with_smaller.total_size(), largest_alone.total_size());
    }

    /// Every slot of `bucket` as a half-open byte range.
    fn slot_ranges(arena: &CaptureArena, bucket: BucketIdx) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        for layer in 0..2 {
            for role in 0..2 {
                let start = arena.offset(bucket, LayerIdx(layer), TensorRole(role));
                ranges.push((start, start + arena.slot_size(bucket, TensorRole(role))));
            }
        }
        ranges
    }

    #[test]
    fn slots_within_a_bucket_do_not_overlap() {
        let arena = two_role_arena();

        for bucket in [BucketIdx(0), BucketIdx(1)] {
            let mut ranges = slot_ranges(&arena, bucket);
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                let ((_, first_end), (second_start, _)) = (pair[0], pair[1]);
                assert!(
                    second_start >= first_end,
                    "slots overlap under bucket {}: one ends at {first_end}, the next starts at {second_start}",
                    bucket.0
                );
            }
        }
    }

    #[test]
    fn buckets_share_their_first_address_and_diverge_after() {
        let arena = two_role_arena();

        assert_eq!(
            arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(0)),
            arena.offset(BucketIdx(1), LayerIdx(0), TensorRole(0)),
        );
        assert_ne!(
            arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(1)),
            arena.offset(BucketIdx(1), LayerIdx(0), TensorRole(1)),
        );
    }

    #[test]
    fn ladder_order_does_not_change_a_bucket_addressing() {
        let sorted = CaptureArena::new(2, &[100, 300], &[2, 8]);
        let reversed = CaptureArena::new(2, &[100, 300], &[8, 2]);

        // The entry value, not its position, determines the layout: bucket 0 of `reversed`
        // addresses exactly like bucket 1 of `sorted`, and the totals agree.
        assert_eq!(
            reversed.offset(BucketIdx(0), LayerIdx(1), TensorRole(1)),
            sorted.offset(BucketIdx(1), LayerIdx(1), TensorRole(1)),
        );
        assert_eq!(reversed.total_size(), sorted.total_size());
    }

    #[test]
    fn empty_ladder_needs_no_memory() {
        let arena = CaptureArena::new(2, &[100, 300], &[]);

        assert_eq!(arena.total_size(), 0);
    }

    #[test]
    #[should_panic(expected = "layer index 2 out of range")]
    fn offset_rejects_out_of_range_layer() {
        two_role_arena().offset(BucketIdx(0), LayerIdx(2), TensorRole(0));
    }

    #[test]
    #[should_panic(expected = "role index 2 out of range")]
    fn offset_rejects_out_of_range_role() {
        two_role_arena().offset(BucketIdx(0), LayerIdx(0), TensorRole(2));
    }
}
