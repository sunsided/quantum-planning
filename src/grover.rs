//! Grover search: amplitude amplification of the states that a [`PhaseOracle`] marks.
//!
//! With `M` of `N = 2^n` states marked and `sin²θ = M/N`, `k` Grover iterations from the
//! even superposition give the success probability `sin²((2k+1)θ)`. [`search_known`] uses
//! the best `k` for a known `M`. [`search_unknown`] uses the exponential search of Boyer,
//! Brassard, Høyer and Tapp (BBHT) when `M` is not known.
//!
//! ```mermaid
//! graph LR
//!     prepare["H on all qubits"] --> oracle["phase oracle: one query"]
//!     oracle --> diffuse["diffusion: H, X, MCZ, X, H"]
//!     diffuse -->|"k times"| oracle
//!     diffuse --> measure["measure, then check: one query"]
//! ```

mod bbht;

pub use bbht::search_unknown;

use core::f64::consts::FRAC_PI_4;

use crate::{Oracle, PhaseOracle, SplitMix64, StateVector};

/// Returns the iteration count that maximizes `sin²((2k+1)θ)` for `marked` of `states`.
///
/// The value is `⌊π/(4θ)⌋` with `sin²θ = marked/states`. It is `0` when no state or every
/// state is marked, because iterations cannot help there.
#[must_use]
pub fn optimal_iterations(states: usize, marked: usize) -> usize {
    if marked == 0 || marked >= states {
        return 0;
    }

    let theta = libm::asin(libm::sqrt(ratio(marked, states)));

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0 < θ ≤ π/2, so π/(4θ) is finite, at least 1/2 and below 2^12 for 2^24 states"
    )]
    let iterations = libm::floor(FRAC_PI_4 / theta) as usize;

    iterations
}

/// Prepares the even superposition in `state` and applies `iterations` Grover iterations.
///
/// Each iteration costs one query of `oracle`. The `state` buffer is reused.
pub fn amplify<O, F, C>(
    state: &mut StateVector,
    oracle: &mut PhaseOracle<O, F, C>,
    iterations: usize,
) where
    O: Oracle<C>,
    F: Fn(usize) -> Option<C>,
{
    state.reset();
    hadamard_all(state);

    for _ in 0..iterations {
        oracle.apply(state);
        diffuse(state);
    }
}

/// Runs Grover search with a known count of `marked` states and returns a verified index.
///
/// The search runs [`optimal_iterations`] iterations, measures once and checks the result
/// classically. It costs that many queries plus one. One run succeeds with the probability
/// `sin²((2k+1)θ)`, at least `1 − M/N`. It returns `None` when the measured index
/// is not marked. Run it again on `None`, or use [`search_unknown`].
pub fn search_known<O, F, C>(
    state: &mut StateVector,
    oracle: &mut PhaseOracle<O, F, C>,
    marked: usize,
    rng: &mut SplitMix64,
) -> Option<usize>
where
    O: Oracle<C>,
    F: Fn(usize) -> Option<C>,
{
    amplify(state, oracle, optimal_iterations(state.states(), marked));

    let index = state.sample(rng);

    oracle.check(index).then_some(index)
}

/// Reflects `state` about the even superposition, up to a global phase of `−1`.
fn diffuse(state: &mut StateVector) {
    hadamard_all(state);
    x_all(state);
    state.mcz(0..state.qubits());
    x_all(state);
    hadamard_all(state);
}

/// Applies `H` to every qubit of `state`.
fn hadamard_all(state: &mut StateVector) {
    for qubit in 0..state.qubits() {
        state.h(qubit);
    }
}

/// Applies `X` to every qubit of `state`.
fn x_all(state: &mut StateVector) {
    for qubit in 0..state.qubits() {
        state.x(qubit);
    }
}

/// Returns `part / whole` as `f64`.
fn ratio(part: usize, whole: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "state counts are at most 2^24, exact in f64"
    )]
    let ratio = part as f64 / whole as f64;

    ratio
}

/// Unit tests for the iteration count.
#[cfg(test)]
mod tests {
    use super::optimal_iterations;

    /// Edge cases give no iterations; one marked state of 1024 gives `⌊π/4·√1024⌋ = 25`.
    #[test]
    fn counts_the_optimal_iterations() {
        assert_eq!(optimal_iterations(16, 0), 0);
        assert_eq!(optimal_iterations(16, 16), 0);
        assert_eq!(optimal_iterations(4, 1), 1);
        assert_eq!(optimal_iterations(16, 5), 1);
        assert_eq!(optimal_iterations(1024, 1), 25);
    }
}
