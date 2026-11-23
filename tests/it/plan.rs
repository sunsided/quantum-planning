//! The plan-validity example end to end, and its query counts against a classical scan.

use quantum_planning::grover::{search_known, search_unknown};
use quantum_planning::{Oracle, PhaseOracle, SplitMix64, StateVector};

use crate::Spread;

/// Accepts 4-bit action masks that run exactly two actions, but not actions 0 and 1
/// together.
struct ValidPlan;

impl Oracle<u8> for ValidPlan {
    fn accepts(&self, plan: &u8) -> bool {
        plan.count_ones() == 2 && plan & 0b0011 != 0b0011
    }
}

/// Maps basis index `i` of a 4-qubit register to the plan with action mask `i`.
fn plan(index: usize) -> Option<u8> {
    u8::try_from(index).ok().filter(|plan| *plan < 16)
}

/// Scans `candidates` classically and returns the first accepted one together with the
/// number of oracle queries spent, the baseline a quantum search is measured against.
fn first_marked<C, O: Oracle<C>>(oracle: &O, candidates: &[C]) -> (Option<usize>, usize) {
    let mut queries = 0;
    for (index, candidate) in candidates.iter().enumerate() {
        queries += 1;
        if oracle.accepts(candidate) {
            return (Some(index), queries);
        }
    }

    (None, queries)
}

/// The oracle marks exactly the valid plans.
#[test]
fn marks_the_valid_plans() {
    let marked = (0..16_u8).filter(|plan| ValidPlan.accepts(plan)).count();

    assert_eq!(marked, 5);
    assert!(ValidPlan.accepts(&0b1010));
    assert!(!ValidPlan.accepts(&0b0011));
    assert!(!ValidPlan.accepts(&0b0111));
}

/// A classical scan stops at the first marked candidate and counts every query.
#[test]
fn classical_scan_counts_queries() {
    let plans: [u8; 4] = [0b0000, 0b0011, 0b0101, 0b0110];

    assert_eq!(first_marked(&ValidPlan, &plans), (Some(2), 3));
    assert_eq!(first_marked(&ValidPlan, &plans[..2]), (None, 2));
}

/// Both Grover searches find valid plans; one known-count run costs two queries and
/// succeeds with `sin²(3θ) ≈ 0.96`, `sin²θ = 5/16`.
#[test]
fn grover_finds_valid_plans_end_to_end() {
    let mut state = StateVector::new(4).unwrap();
    let mut known_hits = 0;

    for seed in 0..100 {
        let mut rng = SplitMix64::new(seed);

        let mut oracle = PhaseOracle::new(ValidPlan, plan);
        if let Some(index) = search_known(&mut state, &mut oracle, 5, &mut rng) {
            assert!(ValidPlan.accepts(&plan(index).unwrap()));
            known_hits += 1;
        }

        assert_eq!(oracle.queries(), 2);

        let mut oracle = PhaseOracle::new(ValidPlan, plan);
        let index = search_unknown(&mut state, &mut oracle, 40, &mut rng)
            .expect("five of sixteen plans are valid");
        assert!(ValidPlan.accepts(&plan(index).unwrap()));
    }

    assert!(known_hits >= 90, "{known_hits} of 100 known-count runs");
}

/// With one marked candidate among 1024, BBHT needs far fewer queries on average than the
/// classical scan, which needs about `N/2`.
#[test]
fn grover_beats_the_classical_scan() {
    let states = 1 << 10;
    let mut classical = 0;
    let mut quantum = 0;
    let mut state = StateVector::new(10).unwrap();

    for seed in 0..64 {
        let spread = Spread { states, count: 1 };
        let shuffled: Vec<usize> = (0..states)
            .map(|index| (index + seed * 16) % states)
            .collect();
        classical += first_marked(&spread, &shuffled).1;

        let mut oracle = PhaseOracle::new(&spread, Some);
        let mut rng = SplitMix64::new(seed as u64);
        search_unknown(&mut state, &mut oracle, 300, &mut rng).expect("one index is marked");
        quantum += oracle.queries();
    }

    println!(
        "mean queries over 64 runs: classical {}, BBHT {}",
        classical / 64,
        quantum / 64
    );
    assert!(classical / 64 > 400 && quantum / 64 < 100);
}
