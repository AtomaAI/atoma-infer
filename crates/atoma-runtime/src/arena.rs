//! The capture arena: engine-owned addresses for every captured step's activations.
//!
//! One arena is shared by every bucket — buckets never replay concurrently, so the activation
//! footprint is the largest bucket's, not the sum across the bucket ladder. A slot's address is a
//! pure function of (bucket, layer, role): every bucket bumps from the same base, so address
//! stability across replays is guaranteed by construction and needs no pinning, weak references,
//! or write-back copies.
//!
//! Addresses are exposed only through [`CaptureArena::offset`] — never an open-coded formula at a
//! call site — so the placement behind the lookup can change without touching any caller. The
//! arena has no model knowledge: roles enter as caller-declared per-token widths and lifetimes,
//! indexed positionally, and the bucket ladder as uninterpreted entries.
//!
//! The arena holds only what is sized per bucket: activations and bridge buffers. Buffers sized
//! once at the ladder maximum — per-step inputs, outputs, and kernel workspaces — are owned by
//! the [`GraphEntry`](crate::graph_entry::GraphEntry) whose graph baked their addresses, never by
//! the arena.

use thiserror::Error;

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

/// Half-open range `[first_use, last_use)` of a layer's op order in which a role's slot holds
/// live data.
///
/// Indices are frame-relative and signed: an index below zero or beyond the op order names an op
/// of an adjacent layer. That is how a residual stream entering a layer — written by the previous
/// layer's last op — declares that its slot is already live at layer entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    /// Op index of the first write into the slot.
    pub first_use: isize,
    /// One past the op index of the last read of the slot.
    pub last_use: isize,
}

/// One role's declaration: everything the arena knows about a tensor role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoleDeclaration {
    /// Per-token width in bytes.
    pub width_bytes: usize,
    /// When the role's slot holds live data, indexing the op order declared in
    /// [`RoleTable::ops_per_layer`].
    pub lifetime: Lifetime,
}

/// The caller-declared role table: a linear op order per layer and one declaration per role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleTable {
    /// Length of one layer's linear op order — the coordinate system every lifetime indexes.
    pub ops_per_layer: usize,
    /// One declaration per role, indexed by [`TensorRole`].
    pub roles: Vec<RoleDeclaration>,
}

/// How slots are placed behind the [`CaptureArena::offset`] lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaLayout {
    /// Greedy earliest-fit over declared lifetimes: a role takes the lowest offset whose bytes
    /// are free for its whole lifetime. Two roles whose lifetimes are disjoint may share bytes,
    /// so activation memory stops scaling with layer count.
    Greedy,
    /// One slot per layer per role, no sharing: the reference layout that reuse layouts are
    /// checked against.
    NoReuse,
}

impl Default for ArenaLayout {
    /// Greedy earliest-fit is the layout to build unless a run is checking against the
    /// reference layout.
    fn default() -> Self {
        ArenaLayout::Greedy
    }
}

/// Rejected arena constructions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ArenaError {
    #[error(
        "role {role} declares an empty or inverted lifetime [{first_use}, {last_use}): first-use \
         must come before last-use in the layer op order"
    )]
    EmptyOrInvertedLifetime {
        role: usize,
        first_use: isize,
        last_use: isize,
    },
}

/// Address layout for every captured step's activations. See the module docs for the invariants.
///
/// The whole offset table is computed once at construction; [`CaptureArena::offset`] is a table
/// lookup, so how slots are placed is a construction-time concern invisible at every call site.
#[derive(Debug, Clone)]
pub struct CaptureArena {
    num_layers: usize,
    role_table: RoleTable,
    /// Tokens per bucket, indexed by [`BucketIdx`]. Uninterpreted: never sorted, deduplicated,
    /// or assumed monotonic.
    bucket_ladder: Vec<usize>,
    /// Slot offsets in bytes, indexed `[bucket][layer * num_roles + role]`.
    offsets: Vec<Vec<usize>>,
    /// Bytes each bucket addresses, indexed by [`BucketIdx`].
    bucket_extents: Vec<usize>,
}

impl CaptureArena {
    /// Builds the arena layout from `num_layers` layers, the caller-declared role table, and the
    /// bucket ladder in tokens per bucket, placing slots per `layout`.
    ///
    /// Rejects a role table containing an empty or inverted lifetime, naming the offending role.
    pub fn new(
        num_layers: usize,
        role_table: RoleTable,
        bucket_ladder: &[usize],
        layout: ArenaLayout,
    ) -> Result<Self, ArenaError> {
        for (role, declaration) in role_table.roles.iter().enumerate() {
            let Lifetime {
                first_use,
                last_use,
            } = declaration.lifetime;
            if first_use >= last_use {
                return Err(ArenaError::EmptyOrInvertedLifetime {
                    role,
                    first_use,
                    last_use,
                });
            }
        }
        let mut arena = Self {
            num_layers,
            role_table,
            bucket_ladder: bucket_ladder.to_vec(),
            offsets: Vec::new(),
            bucket_extents: Vec::new(),
        };
        for bucket in 0..arena.bucket_ladder.len() {
            let bucket = BucketIdx(bucket);
            let (table, extent) = match layout {
                ArenaLayout::Greedy => arena.place_greedy(bucket),
                ArenaLayout::NoReuse => arena.place_no_reuse(bucket),
            };
            arena.offsets.push(table);
            arena.bucket_extents.push(extent);
        }
        Ok(arena)
    }

    /// The global lifetime of one slot: the declared frame-relative lifetime shifted by its
    /// layer's position in the step-wide op timeline.
    fn slot_lifetime(&self, slot: usize) -> (isize, isize) {
        let (layer, role) = (slot / self.num_roles(), slot % self.num_roles());
        let Lifetime {
            first_use,
            last_use,
        } = self.role_table.roles[role].lifetime;
        let base = isize::try_from(layer * self.role_table.ops_per_layer)
            .expect("op timeline fits in isize");
        (base + first_use, base + last_use)
    }

    /// Greedy earliest-fit over declared lifetimes: slots are processed in order of first use,
    /// and each takes the lowest offset whose bytes are free for its whole lifetime. Two slots
    /// whose lifetimes overlap never share bytes; two whose lifetimes are disjoint may.
    fn place_greedy(&self, bucket: BucketIdx) -> (Vec<usize>, usize) {
        let num_slots = self.num_layers * self.num_roles();
        let mut order: Vec<usize> = (0..num_slots).collect();
        order.sort_by_key(|&slot| (self.slot_lifetime(slot).0, slot));

        let mut table = vec![0; num_slots];
        // Placed slots as (live_from, live_until, offset, size).
        let mut placed: Vec<(isize, isize, usize, usize)> = Vec::new();
        let mut extent = 0;
        for slot in order {
            let (live_from, live_until) = self.slot_lifetime(slot);
            let size = self.slot_size(bucket, TensorRole(slot % self.num_roles()));
            let mut busy: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(from, until, _, _)| from < live_until && live_from < until)
                .map(|&(_, _, offset, size)| (offset, offset + size))
                .collect();
            busy.sort_unstable();
            // Busy ranges may overlap each other (two placed slots share bytes when their
            // lifetimes are disjoint), so the scan keeps the running maximum end.
            let mut offset = 0;
            for (busy_from, busy_until) in busy {
                if offset + size <= busy_from {
                    break;
                }
                offset = offset.max(busy_until);
            }
            table[slot] = offset;
            placed.push((live_from, live_until, offset, size));
            extent = extent.max(offset + size);
        }
        (table, extent)
    }

    /// One slot per layer per role, ordered by declaration: the layout with no sharing.
    fn place_no_reuse(&self, bucket: BucketIdx) -> (Vec<usize>, usize) {
        let per_layer = self.layer_extent(bucket);
        let mut table = Vec::with_capacity(self.num_layers * self.num_roles());
        for layer in 0..self.num_layers {
            let mut within_layer = 0;
            for role in 0..self.num_roles() {
                table.push(layer * per_layer + within_layer);
                within_layer += self.slot_size(bucket, TensorRole(role));
            }
        }
        (table, self.num_layers * per_layer)
    }

    /// Byte offset of the slot holding `role`'s activation for `layer` under `bucket`.
    pub fn offset(&self, bucket: BucketIdx, layer: LayerIdx, role: TensorRole) -> usize {
        assert!(
            bucket.0 < self.bucket_ladder.len(),
            "bucket index {} out of range: the bucket ladder has {} buckets",
            bucket.0,
            self.bucket_ladder.len()
        );
        assert!(
            layer.0 < self.num_layers,
            "layer index {} out of range: the arena was built for {} layers",
            layer.0,
            self.num_layers
        );
        assert!(
            role.0 < self.num_roles(),
            "role index {} out of range: {} roles were declared",
            role.0,
            self.num_roles()
        );
        self.offsets[bucket.0][layer.0 * self.num_roles() + role.0]
    }

    /// Size in bytes of the slot holding `role`'s activation under `bucket`.
    pub fn slot_size(&self, bucket: BucketIdx, role: TensorRole) -> usize {
        assert!(
            bucket.0 < self.bucket_ladder.len(),
            "bucket index {} out of range: the bucket ladder has {} buckets",
            bucket.0,
            self.bucket_ladder.len()
        );
        let tokens = self.bucket_ladder[bucket.0];
        (self.role_table.roles[role.0].width_bytes * tokens).next_multiple_of(SLOT_ALIGN)
    }

    /// Total bytes `bucket` addresses: every layer's slots for every role.
    pub fn bucket_footprint(&self, bucket: BucketIdx) -> usize {
        assert!(
            bucket.0 < self.bucket_ladder.len(),
            "bucket index {} out of range: the bucket ladder has {} buckets",
            bucket.0,
            self.bucket_ladder.len()
        );
        self.bucket_extents[bucket.0]
    }

    /// Bytes of device memory backing the arena: the largest bucket's footprint, since every
    /// bucket bumps from the same base and buckets never replay concurrently.
    pub fn total_size(&self) -> usize {
        self.bucket_extents.iter().copied().max().unwrap_or(0)
    }

    fn layer_extent(&self, bucket: BucketIdx) -> usize {
        (0..self.num_roles())
            .map(|r| self.slot_size(bucket, TensorRole(r)))
            .sum()
    }

    fn num_roles(&self) -> usize {
        self.role_table.roles.len()
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

    /// Declares `width_bytes` with a minimal valid lifetime; no-reuse placement ignores
    /// lifetimes, so tests that only exercise addressing declare the simplest one.
    fn role(width_bytes: usize) -> RoleDeclaration {
        RoleDeclaration {
            width_bytes,
            lifetime: Lifetime {
                first_use: 0,
                last_use: 1,
            },
        }
    }

    fn no_reuse(num_layers: usize, widths: &[usize], ladder: &[usize]) -> CaptureArena {
        let roles = widths.iter().map(|&w| role(w)).collect();
        CaptureArena::new(
            num_layers,
            RoleTable {
                ops_per_layer: 1,
                roles,
            },
            ladder,
            ArenaLayout::NoReuse,
        )
        .expect("test role tables declare valid lifetimes")
    }

    /// Two roles at 100 and 300 bytes per token; slot sizes below are hand-computed:
    /// bucket 0 (2 tokens): 200 -> 256, 600 -> 768 (layer extent 1024);
    /// bucket 1 (8 tokens): 800 -> 1024, 2400 -> 2560 (layer extent 3584).
    fn two_role_arena() -> CaptureArena {
        no_reuse(2, &[100, 300], &[2, 8])
    }

    #[test]
    fn empty_lifetime_is_rejected_naming_the_role() {
        let roles = vec![
            role(100),
            RoleDeclaration {
                width_bytes: 300,
                lifetime: Lifetime {
                    first_use: 5,
                    last_use: 5,
                },
            },
        ];
        let err = CaptureArena::new(
            2,
            RoleTable {
                ops_per_layer: 8,
                roles,
            },
            &[2],
            ArenaLayout::NoReuse,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ArenaError::EmptyOrInvertedLifetime {
                role: 1,
                first_use: 5,
                last_use: 5
            }
        );
        assert!(err.to_string().contains("role 1"), "got: {err}");
    }

    #[test]
    fn inverted_lifetime_is_rejected_naming_the_role() {
        let roles = vec![RoleDeclaration {
            width_bytes: 100,
            lifetime: Lifetime {
                first_use: 7,
                last_use: 3,
            },
        }];
        let err = CaptureArena::new(
            1,
            RoleTable {
                ops_per_layer: 8,
                roles,
            },
            &[2],
            ArenaLayout::NoReuse,
        )
        .unwrap_err();

        assert!(err.to_string().contains("role 0"), "got: {err}");
        assert!(err.to_string().contains("[7, 3)"), "got: {err}");
    }

    #[test]
    fn residual_lifetime_entering_from_the_previous_layer_is_valid() {
        // A residual stream is written by the previous layer's last op: first_use is -1 in the
        // consuming layer's frame. That is a declaration the arena must accept.
        let roles = vec![RoleDeclaration {
            width_bytes: 100,
            lifetime: Lifetime {
                first_use: -1,
                last_use: 7,
            },
        }];
        let arena = CaptureArena::new(
            2,
            RoleTable {
                ops_per_layer: 13,
                roles,
            },
            &[2],
            ArenaLayout::NoReuse,
        );

        assert!(arena.is_ok());
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
        let arena = no_reuse(32, &roles, &[1, 8, 64]);

        assert_eq!(arena.total_size(), 75_497_472);
        assert_eq!(arena.total_size(), arena.bucket_footprint(BucketIdx(2)));
    }

    #[test]
    fn total_size_does_not_grow_when_smaller_buckets_are_added() {
        let largest_alone = no_reuse(2, &[100, 300], &[8]);
        let with_smaller = no_reuse(2, &[100, 300], &[2, 8, 4]);

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
        let sorted = no_reuse(2, &[100, 300], &[2, 8]);
        let reversed = no_reuse(2, &[100, 300], &[8, 2]);

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
        let arena = no_reuse(2, &[100, 300], &[]);

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

    #[test]
    #[should_panic(expected = "bucket index 2 out of range")]
    fn offset_rejects_out_of_range_bucket() {
        two_role_arena().offset(BucketIdx(2), LayerIdx(0), TensorRole(0));
    }

    fn declared(width_bytes: usize, first_use: isize, last_use: isize) -> RoleDeclaration {
        RoleDeclaration {
            width_bytes,
            lifetime: Lifetime {
                first_use,
                last_use,
            },
        }
    }

    fn greedy(
        num_layers: usize,
        ops_per_layer: usize,
        roles: Vec<RoleDeclaration>,
        ladder: &[usize],
    ) -> CaptureArena {
        CaptureArena::new(
            num_layers,
            RoleTable {
                ops_per_layer,
                roles,
            },
            ladder,
            ArenaLayout::Greedy,
        )
        .expect("test role tables declare valid lifetimes")
    }

    /// Every (layer, role) slot of `bucket` with its global lifetime and byte range.
    fn live_ranges(
        arena: &CaptureArena,
        bucket: BucketIdx,
    ) -> Vec<((isize, isize), (usize, usize))> {
        let mut ranges = Vec::new();
        for layer in 0..arena.num_layers {
            for role in 0..arena.num_roles() {
                let live = arena.slot_lifetime(layer * arena.num_roles() + role);
                let start = arena.offset(bucket, LayerIdx(layer), TensorRole(role));
                let size = arena.slot_size(bucket, TensorRole(role));
                ranges.push((live, (start, start + size)));
            }
        }
        ranges
    }

    /// The greedy invariant: two slots whose global lifetimes overlap never share bytes.
    fn assert_live_slots_never_share_bytes(arena: &CaptureArena, bucket: BucketIdx) {
        let ranges = live_ranges(arena, bucket);
        for (i, &((a_from, a_until), (a_start, a_end))) in ranges.iter().enumerate() {
            for &((b_from, b_until), (b_start, b_end)) in &ranges[i + 1..] {
                let live_together = a_from < b_until && b_from < a_until;
                let share_bytes = a_start < b_end && b_start < a_end;
                assert!(
                    !(live_together && share_bytes),
                    "slots live [{a_from}, {a_until}) at bytes [{a_start}, {a_end}) and \
                     [{b_from}, {b_until}) at [{b_start}, {b_end}) overlap under bucket {}",
                    bucket.0
                );
            }
        }
    }

    #[test]
    fn greedy_shares_an_offset_between_disjoint_lifetimes() {
        let arena = greedy(1, 4, vec![declared(100, 0, 2), declared(100, 2, 4)], &[2]);

        assert_eq!(
            arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(0)),
            arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(1)),
        );
        assert_eq!(arena.bucket_footprint(BucketIdx(0)), 256);
    }

    #[test]
    fn greedy_never_overlaps_two_live_roles() {
        let arena = greedy(1, 4, vec![declared(100, 0, 3), declared(100, 1, 4)], &[2]);

        assert_live_slots_never_share_bytes(&arena, BucketIdx(0));
        assert_eq!(arena.bucket_footprint(BucketIdx(0)), 512);
    }

    #[test]
    fn greedy_reuse_stops_scaling_with_layer_count() {
        // One role live for a single op: every layer's slot is dead before the next layer's is
        // live, so the whole ladder of layers shares one slot and the footprint is flat.
        let shallow = greedy(4, 1, vec![declared(100, 0, 1)], &[2]);
        let deep = greedy(32, 1, vec![declared(100, 0, 1)], &[2]);

        assert_eq!(shallow.total_size(), deep.total_size());
        assert_eq!(deep.total_size(), 256);
    }

    #[test]
    fn greedy_layers_chain_through_a_residual_role() {
        // A residual role live from the previous layer's last op (-1) through the whole next
        // frame overlaps its neighbouring layers' instances, so consecutive layers must not
        // share its slot — but layers two apart may, giving a ping-pong at constant footprint.
        let roles = vec![declared(100, -1, 3), declared(100, 0, 1)];
        let arena = greedy(8, 3, roles, &[2]);

        assert_live_slots_never_share_bytes(&arena, BucketIdx(0));
        let first = arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(0));
        let second = arena.offset(BucketIdx(0), LayerIdx(1), TensorRole(0));
        assert_ne!(first, second);
        // Layer 2's residual is free to reuse layer 0's slot: the footprint stays flat at depth.
        assert_eq!(
            arena.offset(BucketIdx(0), LayerIdx(2), TensorRole(0)),
            first
        );
    }

    #[test]
    fn bridge_role_spanning_a_break_point_is_never_overlapped_while_live() {
        // One layer, op order 0..12, break point between ops 5 and 6: the bridge is written in
        // the first segment (op 3) and read in the second (through op 8), so its lifetime spans
        // the break. Every role live anywhere in [3, 9) must leave its bytes alone; the role
        // live entirely after it may reuse them.
        let bridge = declared(100, 3, 9);
        let segment_a = declared(100, 0, 4);
        let segment_a_tail = declared(100, 4, 6);
        let segment_b_head = declared(100, 6, 9);
        let after_bridge = declared(100, 9, 12);
        let arena = greedy(
            1,
            12,
            vec![
                bridge,
                segment_a,
                segment_a_tail,
                segment_b_head,
                after_bridge,
            ],
            &[2],
        );

        assert_live_slots_never_share_bytes(&arena, BucketIdx(0));
        let bridge_offset = arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(0));
        for live_across_break in [1, 2, 3] {
            assert_ne!(
                arena.offset(BucketIdx(0), LayerIdx(0), TensorRole(live_across_break)),
                bridge_offset,
                "role {live_across_break} is live while the bridge holds data across the break"
            );
        }
        // Reuse still happens around the bridge: the whole table fits in two slots.
        assert_eq!(arena.bucket_footprint(BucketIdx(0)), 512);
    }

    #[test]
    fn greedy_total_size_is_the_largest_bucket_extent() {
        let roles = vec![declared(100, 0, 2), declared(300, 1, 3)];
        let arena = greedy(2, 3, roles, &[2, 8, 4]);

        let largest = (0..3)
            .map(|b| arena.bucket_footprint(BucketIdx(b)))
            .max()
            .expect("three buckets");
        assert_eq!(arena.total_size(), largest);
        assert_eq!(arena.total_size(), arena.bucket_footprint(BucketIdx(1)));
    }

    #[test]
    fn greedy_ladder_order_does_not_change_a_bucket_addressing() {
        let roles = vec![declared(100, -1, 2), declared(300, 0, 3)];
        let sorted = greedy(2, 3, roles.clone(), &[2, 8]);
        let reversed = greedy(2, 3, roles, &[8, 2]);

        for layer in 0..2 {
            for role in 0..2 {
                assert_eq!(
                    reversed.offset(BucketIdx(0), LayerIdx(layer), TensorRole(role)),
                    sorted.offset(BucketIdx(1), LayerIdx(layer), TensorRole(role)),
                );
            }
        }
        assert_eq!(reversed.total_size(), sorted.total_size());
    }

    mod greedy_properties {
        use proptest::prelude::*;

        use super::*;

        prop_compose! {
            fn role_declarations()(
                width_bytes in 1usize..4096,
                first_use in -16isize..16,
                span in 1isize..24,
            ) -> RoleDeclaration {
                declared(width_bytes, first_use, first_use + span)
            }
        }

        proptest! {
            #[test]
            fn overlapping_lifetimes_never_share_bytes(
                num_layers in 1usize..6,
                ops_per_layer in 1usize..16,
                roles in prop::collection::vec(role_declarations(), 1..12),
                ladder in prop::collection::vec(1usize..64, 1..4),
            ) {
                let arena = greedy(num_layers, ops_per_layer, roles, &ladder);
                for bucket in 0..ladder.len() {
                    let bucket = BucketIdx(bucket);
                    assert_live_slots_never_share_bytes(&arena, bucket);
                    for (_, (_, end)) in live_ranges(&arena, bucket) {
                        prop_assert!(end <= arena.bucket_footprint(bucket));
                    }
                }
            }
        }
    }
}
