//! Exponential Grover search for an unknown number of marked states (BBHT).

use crate::{Oracle, PhaseOracle, SplitMix64, StateVector};

use super::{amplify, ratio};

/// The growth factor of the iteration bound per round. BBHT needs `1 < λ < 4/3`.
const LAMBDA: f64 = 6.0 / 5.0;

/// Runs Grover search without a known marked count and returns a verified index.
///
/// This is the exponential search of Boyer, Brassard, Høyer and Tapp. Each round picks `j`
/// uniformly below the bound `m`, applies `j` iterations, measures and checks the result.
/// After a failed round, `m` grows by `6/5` up to `√N`. A round costs `j + 1` queries. With
/// `M` of `N` states marked, the expected cost is `O(√(N/M))`.
///
/// The search stops when the queries of `oracle` reach `max_queries`, and never goes past
/// it. The error is one-sided: a returned index is always marked, but `None` can also mean
/// that the budget was too small. With no marked state, the result is `None` after
/// `max_queries` queries. A budget of `9·(⌊√N⌋ + 1)` makes a miss rare when a solution exists.
pub fn search_unknown<O, F, C>(
    state: &mut StateVector,
    oracle: &mut PhaseOracle<O, F, C>,
    max_queries: usize,
    rng: &mut SplitMix64,
) -> Option<usize>
where
    O: Oracle<C>,
    F: Fn(usize) -> Option<C>,
{
    let ceiling = libm::sqrt(ratio(state.states(), 1));
    let mut bound = 1.0;

    while oracle.queries() < max_queries {
        let remaining = max_queries - oracle.queries();
        let iterations = pick_iterations(bound, rng).min(remaining - 1);

        amplify(state, oracle, iterations);

        let index = state.sample(rng);

        if oracle.check(index) {
            return Some(index);
        }

        bound = (bound * LAMBDA).min(ceiling);
    }

    None
}

/// Picks an iteration count uniformly from the integers below `bound`.
fn pick_iterations(bound: f64, rng: &mut SplitMix64) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "1 ≤ bound ≤ √N ≤ 2^12, so the ceiling is a small positive integer"
    )]
    let choices = libm::ceil(bound) as usize;

    rng.below(choices)
}
