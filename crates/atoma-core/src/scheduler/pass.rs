//! The body of one scheduling pass.
//!
//! Running requests are budgeted first, in batch order, so a step's tokens go to work already
//! under way before anything new is admitted. A running request the pool cannot grow preempts the
//! most recently admitted request — itself, when it is the newest — and retries. Admission never
//! preempts: it claims each candidate's cached prefix and stops at the first request the budget or
//! the pool cannot serve, putting that one back untouched.
//!
//! The pass neither sweeps nor finishes requests. Requests whose clients are gone are retired
//! before it runs, and requests that reach a stop criterion are retired once their results are
//! applied; what the pass returns is the entries the step computes and the requests it displaced.
//!
//! The scheduler's state reaches the pass as [`Parts`], borrowed field by field, with the settings
//! admission reads beside it as an [`AdmissionWindow`].

use tracing::debug;

use crate::request::{Priority, RequestPhase, Sequence};
use crate::scheduler::admission::AdmissionWindow;
use crate::scheduler::kv::{Kv, PoolExhausted};
use crate::scheduler::{AdmissionPolicy, Entry, Parts, Scheduled};
use crate::types::{RequestSlot, SequenceIndex};

/// Runs the pass: budgets running requests in batch order, then admits until the window, the
/// budget or the pool refuses, returning what the step computes and whom it displaced.
pub(crate) fn schedule(mut parts: Parts<'_>, window: AdmissionWindow) -> Scheduled {
    let mut entries = Vec::with_capacity(parts.budget.max_requests().get());
    let preempted = schedule_running(&mut parts, &mut entries);
    admit(&mut parts, window, &mut entries);
    Scheduled {
        step: parts.step,
        entries,
        preempted,
    }
}

/// Budgets every running request in batch order, returning the requests preempted on the
/// way. A request the budget cannot serve this step stays running with no entry; one the
/// pool cannot grow preempts the most recently admitted request — itself, if it is the
/// newest — and retries.
fn schedule_running(parts: &mut Parts<'_>, entries: &mut Vec<Entry>) -> Vec<RequestSlot> {
    let mut preempted = Vec::new();
    let mut position = 0;
    while position < parts.running.len() {
        let slot = parts.running[position];
        match budget_running(parts, slot, entries, &mut preempted) {
            // The queue shrank under this position, so the next request is at it already.
            Budgeted::DisplacedItself => {}
            Budgeted::Scheduled => position += 1,
            Budgeted::BudgetSpent => return preempted,
        }
    }
    preempted
}

/// Gives every sequence of the running request at `slot` an entry, growing its blocks and
/// spending the budget for each.
fn budget_running(
    parts: &mut Parts<'_>,
    slot: RequestSlot,
    entries: &mut Vec<Entry>,
    preempted: &mut Vec<RequestSlot>,
) -> Budgeted {
    let sequence_count = parts
        .requests
        .get(slot)
        .expect("running slots are live")
        .sequences()
        .len();
    for index in 0..sequence_count {
        let sequence = &parts
            .requests
            .get(slot)
            .expect("running slots are live")
            .sequences()[index];
        let Some(query_len) = parts.budget.offer(sequence.remaining()) else {
            return Budgeted::BudgetSpent;
        };
        let context_len = sequence.computed();
        let total = sequence.total();
        let sequence_len = context_len + query_len.get();
        if grow(parts, slot, index, sequence_len, preempted) == Grown::DisplacedItself {
            return Budgeted::DisplacedItself;
        }
        parts.budget.spend(query_len);
        entries.push(Entry {
            slot,
            sequence: SequenceIndex::new(u16::try_from(index).expect("sequence indices fit u16")),
            context_len,
            query_len,
            samples: sequence_len == total,
        });
    }
    Budgeted::Scheduled
}

/// Grows sequence `index` of `slot` to cover `sequence_len` tokens, preempting the most
/// recently admitted request as often as it takes and recording each victim in `preempted`.
/// The victim is this request itself when it is the newest one running.
fn grow(
    parts: &mut Parts<'_>,
    slot: RequestSlot,
    index: usize,
    sequence_len: usize,
    preempted: &mut Vec<RequestSlot>,
) -> Grown {
    loop {
        let sequence = &mut parts
            .requests
            .get_mut(slot)
            .expect("running slots are live")
            .sequences_mut()[index];
        if parts.kv.ensure_blocks(sequence, sequence_len) != Err(PoolExhausted) {
            return Grown::Fits;
        }
        let victim = preempt_newest(parts);
        preempted.push(victim);
        if victim == slot {
            return Grown::DisplacedItself;
        }
    }
}

/// Preemption: the most recently admitted running request releases its KV and computes from
/// its first token again when it re-enters Running. Returns the request displaced.
fn preempt_newest(parts: &mut Parts<'_>) -> RequestSlot {
    let slot = parts
        .running
        .pop()
        .expect("preemption has a running request to displace");
    let request = parts
        .requests
        .get_mut(slot)
        .expect("running slots are live");
    let RequestPhase::Running(running) = request.phase() else {
        unreachable!("the running queue holds only Running requests")
    };
    request.set_phase(RequestPhase::Preempted(running.preempt(parts.step)));
    for sequence in request.sequences_mut() {
        parts.kv.release(sequence);
        sequence.forget_computed();
    }
    parts.preempted.push(slot);
    debug!(request = request.id().get(), "request preempted");
    slot
}

/// Admission: examines up to the window of candidates — the preempted stack top first, last
/// in first out, then the waiting request the policy picks from the window — claims each
/// one's cached prefix, and stops at the first request the budget or the pool cannot serve.
/// Admission never preempts.
fn admit(parts: &mut Parts<'_>, window: AdmissionWindow, entries: &mut Vec<Entry>) {
    for _ in 0..window.size.get() {
        let Some((slot, from)) = next_candidate(parts, window) else {
            return;
        };
        if admit_candidate(parts, slot, from, entries) == Admitted::No {
            return;
        }
    }
}

/// Admits one candidate: it claims its cached prefix, takes an entry out of the budget and
/// leaves the queue it came from. A candidate the budget or the pool cannot serve goes back
/// untouched, and its refusal ends this pass's admission.
fn admit_candidate(
    parts: &mut Parts<'_>,
    slot: RequestSlot,
    from: Candidate,
    entries: &mut Vec<Entry>,
) -> Admitted {
    let request = parts.requests.get_mut(slot).expect("candidates are live");
    let sequence = &mut request.sequences_mut()[0];
    let total = sequence.total();
    parts.kv.claim_prefix(sequence);
    let context_len = sequence.computed();
    let Some(query_len) = parts.budget.offer(sequence.remaining()) else {
        return give_back(&mut parts.kv, sequence);
    };
    if parts
        .kv
        .ensure_blocks(sequence, context_len + query_len.get())
        == Err(PoolExhausted)
    {
        return give_back(&mut parts.kv, sequence);
    }
    let admitted = match (from, request.phase()) {
        (Candidate::Preempted, RequestPhase::Preempted(phase)) => {
            parts.preempted.pop();
            phase.admit(parts.step)
        }
        (Candidate::Waiting { position }, RequestPhase::Waiting(phase)) => {
            parts.waiting.remove(position);
            phase.admit(parts.step)
        }
        (
            Candidate::Preempted | Candidate::Waiting { .. },
            RequestPhase::Waiting(_)
            | RequestPhase::Running(_)
            | RequestPhase::Preempted(_)
            | RequestPhase::Finished(_)
            | RequestPhase::Padding,
        ) => unreachable!("each admission queue holds only its own phase"),
    };
    request.set_phase(RequestPhase::Running(admitted));
    parts.running.push(slot);
    parts.budget.spend(query_len);
    entries.push(Entry {
        slot,
        sequence: SequenceIndex::new(0),
        context_len,
        query_len,
        samples: context_len + query_len.get() == total,
    });
    Admitted::Yes
}

/// Puts a refused candidate back as it was: what its claim pinned or leased goes back to the
/// pool, and the sequence forgets what the claim counted as computed.
fn give_back(kv: &mut Kv<'_>, sequence: &mut Sequence) -> Admitted {
    kv.release(sequence);
    sequence.forget_computed();
    Admitted::No
}

/// The next admission candidate: the preempted stack top, or — while admission is open —
/// the waiting request the policy picks from the window. A closed admission still offers the
/// preempted stack, since those are running requests on their way to finishing. Candidates
/// whose clients are gone are gone already: the pass sweeps them before it spends anything.
fn next_candidate(
    parts: &mut Parts<'_>,
    window: AdmissionWindow,
) -> Option<(RequestSlot, Candidate)> {
    if let Some(&slot) = parts.preempted.last() {
        return Some((slot, Candidate::Preempted));
    }
    if !window.open {
        return None;
    }
    let position = match window.policy {
        AdmissionPolicy::Fcfs => 0,
        AdmissionPolicy::LongestPrefixMatch => longest_prefix_position(parts, window)?,
        AdmissionPolicy::Priority => priority_position(parts, window)?,
    };
    parts
        .waiting
        .get(position)
        .map(|&slot| (slot, Candidate::Waiting { position }))
}

/// The position in the waiting queue, within the window, of the request with the most
/// cached prefix blocks; ties go to the earliest arrival. A selection, never a sort.
fn longest_prefix_position(parts: &mut Parts<'_>, window: AdmissionWindow) -> Option<usize> {
    let block_size = parts.kv.block_size;
    let mut best: Option<(usize, usize)> = None;
    for (position, &slot) in parts.waiting.iter().take(window.size.get()).enumerate() {
        let sequence = &parts
            .requests
            .get(slot)
            .expect("waiting slots are live")
            .sequences()[0];
        let hits = parts.kv.index.lookup(sequence.hashable_prefix(block_size));
        if best.is_none_or(|(_, best_hits)| hits > best_hits) {
            best = Some((position, hits));
        }
    }
    best.map(|(position, _)| position)
}

/// The position in the waiting queue, within the window, of the highest-priority request; ties
/// go to the earliest arrival. A selection, never a sort.
fn priority_position(parts: &Parts<'_>, window: AdmissionWindow) -> Option<usize> {
    let mut best: Option<(usize, Priority)> = None;
    for (position, &slot) in parts.waiting.iter().take(window.size.get()).enumerate() {
        let priority = parts
            .requests
            .get(slot)
            .expect("waiting slots are live")
            .priority();
        if best.is_none_or(|(_, best_priority)| priority > best_priority) {
            best = Some((position, priority));
        }
    }
    best.map(|(position, _)| position)
}

/// Whether a candidate was admitted, or put back for the pool or the budget to allow later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admitted {
    Yes,
    No,
}

/// How a running request's turn at the budget ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Budgeted {
    /// Every sequence has an entry this step.
    Scheduled,
    /// The pool could not grow the request and it was the newest running one, so it displaced
    /// itself and is no longer in the running queue.
    DisplacedItself,
    /// The budget has nothing left for another entry; the pass is over.
    BudgetSpent,
}

/// How growing a sequence's blocks ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Grown {
    /// The sequence holds blocks for every token the step computes.
    Fits,
    /// The request needing the room was the newest running one, so it was its own victim.
    DisplacedItself,
}

/// Which admission queue a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Candidate {
    Preempted,
    Waiting { position: usize },
}
