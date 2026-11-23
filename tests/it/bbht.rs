//! BBHT exponential search: correctness and the `O(√(N/M))` query scaling.

use quantum_planning::grover::search_unknown;
use quantum_planning::{Oracle, PhaseOracle, SplitMix64, StateVector};

use crate::{Spread, to_f64};

/// Seeded runs per configuration.
const RUNS: usize = 200;

/// Returns the query budget `9·(⌊√N⌋ + 1)` for `states` basis states.
fn budget(states: usize) -> usize {
    9 * (states.isqrt() + 1)
}

/// Runs BBHT once with `seed` and returns the found index and the queries spent.
fn run(qubits: usize, marked: usize, seed: u64) -> (Option<usize>, usize) {
    let mut state = StateVector::new(qubits).unwrap();
    let spread = Spread {
        states: state.states(),
        count: marked,
    };
    let mut oracle = PhaseOracle::new(&spread, Some);
    let mut rng = SplitMix64::new(seed);

    let budget = budget(state.states());
    let found = search_unknown(&mut state, &mut oracle, budget, &mut rng);

    if let Some(index) = found {
        assert!(spread.accepts(&index), "BBHT returned an unmarked index");
    }

    (found, oracle.queries())
}

/// BBHT finds a marked index whenever one exists, on every seeded run.
#[test]
fn finds_a_marked_candidate_when_one_exists() {
    for qubits in 1..=10 {
        for marked in [1, 2, 3] {
            if marked >= 1 << qubits {
                continue;
            }

            for seed in 0..RUNS as u64 {
                let (found, _) = run(qubits, marked, seed);
                assert!(found.is_some(), "n={qubits} M={marked} seed={seed}");
            }
        }
    }
}

/// Without a marked index, BBHT stops at the budget and never goes past it.
#[test]
fn stops_at_the_budget_without_a_solution() {
    for seed in 0..20 {
        let (found, queries) = run(8, 0, seed);

        assert_eq!(found, None);
        assert_eq!(queries, budget(256));
    }
}

/// The mean query count over seeded runs, divided by `√(N/M)`, stays in a fixed band.
///
/// The measured ratios are 1.37 to 1.68 for these seeds; the band leaves margin on both sides.
#[test]
fn mean_queries_scale_as_sqrt_n_over_m() {
    for (qubits, marked) in [(6, 1), (8, 1), (10, 1), (10, 4), (10, 16), (10, 64)] {
        let total: usize = (0..RUNS as u64)
            .map(|seed| run(qubits, marked, seed).1)
            .sum();
        let mean = to_f64(total) / to_f64(RUNS);
        let ratio = mean / (to_f64(1 << qubits) / to_f64(marked)).sqrt();

        println!("n={qubits} M={marked}: mean {mean:.1} queries, ratio {ratio:.2}");
        assert!(
            (1.0..2.5).contains(&ratio),
            "n={qubits} M={marked}: {ratio}"
        );
    }
}
