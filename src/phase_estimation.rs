//! Exact outcome distribution of quantum phase estimation, shared by the walk algorithms.
//!
//! Phase estimation with a `K`-outcome control register on a unitary `U` and a start state
//! `|ψ⟩` measures `y` with probability `‖(1/K)·Σₖ e^(−2πi·y·k/K)·Uᵏ|ψ⟩‖²`, for `k < K`. An
//! eigenvector with eigenvalue `e^(2πi·φ)` gives `y ≈ K·φ`. The helpers here compute that
//! sum directly from `Uᵏ|ψ⟩`, which is the exact distribution of the textbook circuit
//! (control register in even superposition, controlled powers, inverse Fourier transform).
//! The circuit applies `U` a total of `K − 1` times, so that is the query cost.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::TAU;

use crate::Complex;

/// Returns the probability of each outcome `y < outcomes` of phase estimation of `step` on
/// `start`.
///
/// `step` applies `U` in place. It runs `outcomes − 1` times. The cost is
/// `outcomes² · start.len()` complex multiplications, so keep both small.
pub(crate) fn outcome_probabilities(
    start: &[Complex],
    outcomes: usize,
    mut step: impl FnMut(&mut [Complex]),
) -> Vec<f64> {
    let mut sums = vec![vec![Complex::ZERO; start.len()]; outcomes];
    let mut power = start.to_vec();

    for k in 0..outcomes {
        if k > 0 {
            step(&mut power);
        }

        for (y, sum) in sums.iter_mut().enumerate() {
            let twiddle = Complex::from_angle(-TAU * ratio((y * k) % outcomes, outcomes));

            accumulate(sum, &power, twiddle);
        }
    }

    sums.iter()
        .map(|sum| squared_norm(sum) / square(outcomes))
        .collect()
}

/// Returns the probability of outcome `0`: the weight of `start` on the eigenvalues near 1.
///
/// Same as entry `0` of [`outcome_probabilities`], at `outcomes · start.len()` cost.
pub(crate) fn zero_outcome_probability(
    start: &[Complex],
    outcomes: usize,
    mut step: impl FnMut(&mut [Complex]),
) -> f64 {
    let mut sum = start.to_vec();
    let mut power = start.to_vec();

    for _ in 1..outcomes {
        step(&mut power);
        accumulate(&mut sum, &power, Complex::ONE);
    }

    squared_norm(&sum) / square(outcomes)
}

/// Adds `factor · vector` to `sum`, entry by entry.
fn accumulate(sum: &mut [Complex], vector: &[Complex], factor: Complex) {
    for (total, amplitude) in sum.iter_mut().zip(vector) {
        *total = *total + factor * *amplitude;
    }
}

/// Returns `Σ |vᵢ|²`.
fn squared_norm(vector: &[Complex]) -> f64 {
    vector.iter().map(|amplitude| amplitude.norm_sqr()).sum()
}

/// Returns `count²` as `f64`.
fn square(count: usize) -> f64 {
    let count = ratio(count, 1);

    count * count
}

/// Returns `part / whole` as `f64`.
fn ratio(part: usize, whole: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "outcome counts and products below them stay far below 2^53"
    )]
    let ratio = part as f64 / whole as f64;

    ratio
}

/// Unit tests on a diagonal unitary with exactly representable phases.
#[cfg(test)]
mod tests {
    use super::{outcome_probabilities, ratio, zero_outcome_probability};
    use crate::Complex;
    use core::f64::consts::TAU;

    /// Multiplies entry `j` by `e^(2πi·phases[j]/16)`.
    fn diagonal(phases: &[usize]) -> impl FnMut(&mut [Complex]) + '_ {
        move |vector| {
            for (amplitude, &phase) in vector.iter_mut().zip(phases) {
                *amplitude = *amplitude * Complex::from_angle(TAU * ratio(phase, 16));
            }
        }
    }

    /// An eigenvector with phase `j/K` gives outcome `j` with certainty; a mix of two gives
    /// both outcomes with their weights.
    #[test]
    fn exact_phases_give_exact_outcomes() {
        let phases = [5, 12];
        let eigen = [Complex::ONE, Complex::ZERO];

        let probabilities = outcome_probabilities(&eigen, 16, diagonal(&phases));
        assert!((probabilities[5] - 1.0).abs() < 1e-12);

        let half = core::f64::consts::FRAC_1_SQRT_2;
        let mix = [Complex::new(half, 0.0), Complex::new(0.0, half)];
        let probabilities = outcome_probabilities(&mix, 16, diagonal(&phases));
        assert!((probabilities[5] - 0.5).abs() < 1e-12);
        assert!((probabilities[12] - 0.5).abs() < 1e-12);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// The zero outcome alone matches the full distribution.
    #[test]
    fn zero_outcome_matches_the_distribution() {
        let phases = [0, 3];
        let start = [Complex::new(0.6, 0.0), Complex::new(0.0, 0.8)];

        let full = outcome_probabilities(&start, 16, diagonal(&phases))[0];
        let zero = zero_outcome_probability(&start, 16, diagonal(&phases));

        assert!((full - 0.36).abs() < 1e-12 && (zero - full).abs() < 1e-12);
    }
}
