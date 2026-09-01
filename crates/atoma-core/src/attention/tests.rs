//! The contract driven end to end by a fake backend, on a machine with no GPU.

use crate::attention::fake::{FakeBackend, FakeError, FakeRecorder};
use crate::attention::{
    AttentionBackend, CaptureContract, Captured, DeclarerKind, ModelDeclaration, PlanInput,
    PreparedPlan, SupportLevel, Workspace, WorkspaceRequirement,
};
use crate::dispatch::{
    BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, EagerReason, GraphKey, LiveBatch,
};
use crate::test_support::{requests, site, tokens};

/// Entries the fake backends here plan over.
const METADATA_ENTRIES: usize = 64;

/// The key of a uniform single-token decode batch padded to `bucket`.
fn key(bucket: usize) -> GraphKey {
    GraphKey::from_padded_batch(tokens(bucket), requests(bucket), true)
}

fn backend(name: &str, support_level: SupportLevel) -> FakeBackend {
    FakeBackend::new(name, support_level, METADATA_ENTRIES)
}

/// A workspace holding what `backend` needs at `bucket`.
fn workspace(backend: &FakeBackend, bucket: usize) -> Workspace<Captured, Vec<u8>> {
    let bytes = backend.workspace_bytes(key(bucket));
    Workspace::new(vec![0; bytes], bytes)
}

#[test]
fn a_backend_declares_its_level_and_the_sites_it_cannot_capture() {
    let declaration = backend("fake", SupportLevel::UniformSingleTokenDecode)
        .cannot_capture(site(0, 4))
        .rank_coupled(site(7, 2))
        .declaration();

    assert_eq!(declaration.name(), "fake");
    assert_eq!(
        declaration.support_level(),
        SupportLevel::UniformSingleTokenDecode
    );
    assert_eq!(declaration.break_points().sites(), [site(0, 4), site(7, 2)]);
    assert!(declaration
        .break_points()
        .iter()
        .all(|point| point.declarer().kind() == DeclarerKind::Backend));
}

#[test]
fn several_backends_settle_the_graph_mode_at_the_weakest() {
    let contract = CaptureContract::resolve(
        &[
            backend("fake-any-batch", SupportLevel::Always).declaration(),
            backend("fake-decode-only", SupportLevel::UniformSingleTokenDecode).declaration(),
        ],
        &ModelDeclaration::new("fake-model"),
    );

    assert_eq!(
        contract.graph_mode().support_level(),
        SupportLevel::UniformSingleTokenDecode
    );
}

#[test]
fn preparation_replans_the_same_addresses_for_every_batch_shape() {
    let mut backend = backend("fake", SupportLevel::Always);

    let small = backend
        .prepare(PlanInput {
            key: key(2),
            sequence_lens: &[16, 32],
        })
        .expect("two entries fit");
    let planned_small = backend.metadata().to_vec();
    let large = backend
        .prepare(PlanInput {
            key: key(8),
            sequence_lens: &[64, 64, 64, 64, 64, 64, 64, 64],
        })
        .expect("eight entries fit");

    assert_eq!(
        small.metadata_addresses(),
        large.metadata_addresses(),
        "a captured graph baked these addresses; only their contents may change"
    );
    assert_ne!(
        planned_small,
        backend.metadata(),
        "the second batch is scheduled differently from the first"
    );
    assert_eq!(
        &backend.metadata()[..8],
        &[4, 4, 4, 4, 4, 4, 4, 4],
        "each 64-token entry takes four 16-token tiles"
    );
}

#[test]
fn preparation_over_what_was_allocated_is_refused_with_both_numbers() {
    let mut backend = FakeBackend::new("fake", SupportLevel::Always, 4);

    let refused = backend
        .prepare(PlanInput {
            key: key(8),
            sequence_lens: &[1; 8],
        })
        .expect_err("eight entries do not fit metadata for four");

    assert_eq!(
        refused,
        FakeError::BatchOverMetadata {
            entries: 8,
            allocated: 4,
        }
    );
}

#[test]
fn recording_launches_the_shapes_the_key_fixed() {
    let mut backend = backend("fake", SupportLevel::Always);
    let mut recorder = FakeRecorder::default();
    let mut workspace = workspace(&backend, 8);

    let plan = backend
        .prepare(PlanInput {
            key: key(8),
            // Three live entries in a bucket of eight: the padding dummies plan like the rest,
            // and nothing the recording enqueues depends on which entries were live.
            sequence_lens: &[48, 16, 32, 1, 1, 1, 1, 1],
        })
        .expect("eight entries fit");
    backend
        .record(&plan, &mut workspace, &mut recorder)
        .expect("the workspace covers the call");
    backend
        .record(&plan, &mut workspace, &mut recorder)
        .expect("recording again enqueues the same work");

    let launches = recorder.launches();
    assert_eq!(launches.len(), 4);
    assert_eq!(launches[..2], launches[2..]);
    assert!(
        launches.iter().all(|launch| launch.threads == 8),
        "every launch is shaped by the padded bucket, not by the live batch"
    );
    assert!(
        launches
            .iter()
            .all(|launch| launch.metadata == plan.metadata_addresses()[0]),
        "every launch reads the metadata at the address preparation re-plans"
    );
}

#[test]
fn a_larger_bucket_records_larger_launches() {
    let mut backend = backend("fake", SupportLevel::Always);
    let mut recorder = FakeRecorder::default();
    let mut workspace = workspace(&backend, 16);

    for bucket in [8, 16] {
        let plan = backend
            .prepare(PlanInput {
                key: key(bucket),
                sequence_lens: &vec![32; bucket],
            })
            .expect("the batch fits");
        backend
            .record(&plan, &mut workspace, &mut recorder)
            .expect("the workspace is sized for the larger bucket");
    }

    let threads: Vec<usize> = recorder
        .launches()
        .iter()
        .map(|launch| launch.threads)
        .collect();
    assert_eq!(threads, [8, 8, 16, 16]);
}

#[test]
fn recording_writes_only_the_workspace_the_caller_allocated() {
    let mut backend = backend("fake", SupportLevel::Always);
    let mut recorder = FakeRecorder::default();
    let mut workspace = workspace(&backend, 8);
    let allocated = workspace.buffer().as_ptr();
    let capacity = workspace.buffer().capacity();

    let plan = backend
        .prepare(PlanInput {
            key: key(8),
            sequence_lens: &[32; 8],
        })
        .expect("eight entries fit");
    backend
        .record(&plan, &mut workspace, &mut recorder)
        .expect("the workspace covers the call");

    assert_eq!(
        workspace.buffer().as_ptr(),
        allocated,
        "a recording that reallocated would leave the graph pointing at freed bytes"
    );
    assert_eq!(workspace.buffer().capacity(), capacity);
}

#[test]
fn recording_refuses_a_workspace_smaller_than_the_call_needs() {
    let mut backend = backend("fake", SupportLevel::Always);
    let mut recorder = FakeRecorder::default();
    // Allocated for a bucket of eight, handed to a call recorded for sixty-four.
    let mut workspace = workspace(&backend, 8);

    let plan = backend
        .prepare(PlanInput {
            key: key(64),
            sequence_lens: &[32; 64],
        })
        .expect("sixty-four entries fit");
    let refused = backend
        .record(&plan, &mut workspace, &mut recorder)
        .expect_err("the workspace is too small");

    assert_eq!(
        refused,
        FakeError::WorkspaceTooSmall {
            needed: plan.workspace_bytes(),
            handed: workspace.bytes(),
        }
    );
    assert!(
        recorder.launches().is_empty(),
        "a refused recording enqueues nothing"
    );
}

#[test]
fn a_workspace_sized_at_the_largest_bucket_covers_every_smaller_call() {
    let mut backend = backend("fake", SupportLevel::Always);
    let largest = workspace(&backend, 64);

    for bucket in [1, 8, 64] {
        let plan = backend
            .prepare(PlanInput {
                key: key(bucket),
                sequence_lens: &vec![32; bucket],
            })
            .expect("the batch fits");
        assert!(
            largest.covers(&plan),
            "a workspace for the largest bucket must cover a call at bucket {bucket}"
        );
    }
}

/// The dispatcher built from what `backends` and `model` settled, over a small bucket ladder.
fn dispatcher(backends: &[FakeBackend], model: &ModelDeclaration) -> Dispatcher {
    let declarations: Vec<_> = backends.iter().map(FakeBackend::declaration).collect();
    Dispatcher::new(
        &DispatchConfig {
            bucket_ladder: BucketLadder::new(vec![1, 2, 4, 8]).expect("nonzero buckets"),
            captured_max_requests: requests(8),
        },
        &CaptureContract::resolve(&declarations, model),
    )
}

fn decode_batch(count: usize) -> LiveBatch {
    LiveBatch {
        token_count: tokens(count),
        request_count: requests(count),
        uniform_decode: true,
    }
}

#[test]
fn capability_and_policy_break_points_reach_the_dispatcher_as_their_union() {
    let attention = backend("fake-attention", SupportLevel::Always).cannot_capture(site(0, 3));
    let collective = backend("fake-collective", SupportLevel::Always).rank_coupled(site(5, 0));
    let model = ModelDeclaration::new("fake-model").eager_at(site(2, 1));

    let mut dispatcher = dispatcher(&[attention, collective], &model);

    assert_eq!(
        dispatcher.break_points().sites(),
        [site(0, 3), site(2, 1), site(5, 0)]
    );
    assert!(
        matches!(
            dispatcher.dispatch(decode_batch(4)),
            DispatchDecision::SegmentedReplay(_)
        ),
        "a pass with break points standing over it replays as segments"
    );
}

#[test]
fn nothing_breaks_at_the_attention_op_unless_a_declarer_names_it() {
    // The fake records an attention launch in every step, and no declarer names its site, so the
    // pass captures whole: no op is a break point by virtue of what it computes.
    let mut backend = backend("fake", SupportLevel::Always);
    let mut recorder = FakeRecorder::default();
    let mut workspace = workspace(&backend, 4);
    let plan = backend
        .prepare(PlanInput {
            key: key(4),
            sequence_lens: &[16; 4],
        })
        .expect("four entries fit");
    backend
        .record(&plan, &mut workspace, &mut recorder)
        .expect("the workspace covers the call");
    assert!(recorder
        .launches()
        .iter()
        .any(|launch| launch.name == "decode_attention"));

    let mut dispatcher = dispatcher(&[backend], &ModelDeclaration::new("fake-model"));

    assert!(dispatcher.break_points().is_empty());
    assert!(matches!(
        dispatcher.dispatch(decode_batch(4)),
        DispatchDecision::FullReplay(_)
    ));
}

#[test]
fn a_backend_that_captures_nothing_sends_every_batch_eager() {
    let mut dispatcher = dispatcher(
        &[
            backend("fake-any-batch", SupportLevel::Always),
            backend("fake-uncapturable", SupportLevel::Never),
        ],
        &ModelDeclaration::new("fake-model"),
    );

    assert_eq!(
        dispatcher.dispatch(decode_batch(4)),
        DispatchDecision::Eager(EagerReason::SupportLevelInsufficient {
            support_level: SupportLevel::Never,
            required: SupportLevel::UniformSingleTokenDecode,
            token_count: tokens(4),
            request_count: requests(4),
        }),
        "one backend that captures nothing settles the graph mode for all of them"
    );
}
