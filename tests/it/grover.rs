//! Grover search with a known marked count, checked against the closed form.

use quantum_planning::grover::{amplify, optimal_iterations};
use quantum_planning::{Oracle, PhaseOracle, StateVector};

use crate::{Spread, to_f64};

/// Returns the closed-form success probability `sin²((2k+1)θ)` with `sin²θ = marked/states`.
fn closed_form(states: usize, marked: usize, iterations: usize) -> f64 {
    let theta = (to_f64(marked) / to_f64(states)).sqrt().asin();

    (to_f64(2 * iterations + 1) * theta).sin().powi(2)
}

/// Returns the marked counts to test for `states`: few, some, a quarter, half, all but one.
fn marked_counts(states: usize) -> Vec<usize> {
    let mut counts = vec![1, 2, 3, 5, states / 4, states / 2, states - 1];
    counts.retain(|&count| count >= 1 && count < states);
    counts.sort_unstable();
    counts.dedup();

    counts
}

/// After `k` iterations, the success probability is `sin²((2k+1)θ)` within `1e-9`, for every
/// `n ≤ 10`, several marked counts and `k` up to past the optimum.
#[test]
fn success_probability_matches_the_closed_form() {
    for qubits in 1..=10 {
        let mut state = StateVector::new(qubits).unwrap();
        let states = state.states();

        for marked in marked_counts(states) {
            let spread = Spread {
                states,
                count: marked,
            };
            let mut oracle = PhaseOracle::new(&spread, Some);

            for iterations in 0..=optimal_iterations(states, marked) + 3 {
                amplify(&mut state, &mut oracle, iterations);

                let success: f64 = state
                    .probabilities()
                    .enumerate()
                    .filter(|(index, _)| spread.accepts(index))
                    .map(|(_, probability)| probability)
                    .sum();
                let expected = closed_form(states, marked, iterations);

                assert!(
                    (success - expected).abs() < 1e-9,
                    "n={qubits} M={marked} k={iterations}: {success} != {expected}"
                );
            }
        }
    }
}

/// Every application of the phase oracle and every classical check counts as one query.
#[test]
fn counts_one_query_per_iteration() {
    let mut state = StateVector::new(4).unwrap();
    let mut oracle = PhaseOracle::new(
        Spread {
            states: 16,
            count: 1,
        },
        Some,
    );

    amplify(&mut state, &mut oracle, 3);
    assert_eq!(oracle.queries(), 3);

    assert!(
        oracle.check(1),
        "index 1 is the marked one: 37 + 11 = 48 = 0 mod 16"
    );
    assert!(!oracle.check(0));
    assert_eq!(oracle.queries(), 5);
}
