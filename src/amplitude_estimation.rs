//! Amplitude estimation of Monte Carlo values, such as the rollout values of MCTS.
//!
//! A rollout from a game state draws a random seed and returns a value in `[0, 1]`, for
//! example `1` for a win and `0` for a loss. MCTS estimates the mean value `a` from the
//! rollouts. Classical Monte Carlo with `S` rollouts has the error `σ/√S`. Canonical
//! amplitude estimation (Brassard, Høyer, Mosca and Tapp) reaches the error
//! `O(√(a(1−a))/K)` with about `2K` rollout calls, a quadratic speedup. The constants
//! favour Monte Carlo for small budgets: on the eight-seed example of the tests, amplitude
//! estimation has the lower median error from about 255 queries on, not at 63.
//!
//! The algorithm prepares, on `n` seed qubits and one value qubit,
//!
//! ```text
//! A|0⟩ = (1/√N) · Σᵢ |i⟩ ⊗ (√(1 − v(i)) |0⟩ + √v(i) |1⟩),   N = 2^n,
//! ```
//!
//! so the value qubit reads `1` with probability `a = (1/N) · Σᵢ v(i)`. The Grover operator
//! `Q = −A·S₀·A⁻¹·S₁` (`S₁` flips the sign where the value qubit is `1`, `S₀` where the
//! whole register is `|0⟩`) has the eigenvalues `e^(±2iθ)` with `sin²θ = a`. Phase
//! estimation of `Q` with `K = 2^m` outcomes measures `y`, and the estimate is
//! `sin²(π·y/K)`.
//!
//! ```mermaid
//! graph LR
//!     prepare["A: H on the seeds, value rotation"] --> qpe["phase estimation of Q, K outcomes"]
//!     qpe --> measure["measure y"]
//!     measure --> estimate["a ≈ sin²(π·y/K)"]
//! ```
//!
//! # Query count
//!
//! One query is one call of the rollout `A` or its inverse. Phase estimation applies `Q`
//! `K − 1` times and `Q` calls `A` and `A⁻¹` once each, so with the preparation the cost is
//! `2K − 1` queries. One classical rollout is one query.
//!
//! # Simulation
//!
//! The simulator holds the `n + 1` qubits of `A` and computes the phase-estimation outcome
//! distribution exactly, without simulating the `m` control qubits. The register limit
//! still counts them: `n + 1 + m` must not exceed [`MAX_QUBITS`].

use alloc::vec::Vec;
use core::f64::consts::PI;
use core::fmt;

use rand_core::Rng;

use crate::phase_estimation::outcome_probabilities;
use crate::rng::{below, pick};
use crate::{MAX_QUBITS, StateVector, TooManyQubits};

/// An estimate of the mean rollout value, with the queries it cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The estimated mean value, in `[0, 1]`.
    pub value: f64,
    /// The rollout calls spent; see the [module docs](self).
    pub queries: usize,
}

/// The error for an amplitude-estimation problem the simulator cannot run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EstimationError {
    /// A rollout value is not a finite number in `[0, 1]`.
    InvalidValue {
        /// The seed of the rollout.
        seed: usize,
        /// The value it returned.
        value: f64,
    },
    /// The seed qubits, the value qubit and the precision qubits exceed the simulator.
    TooManyQubits(TooManyQubits),
}

impl fmt::Display for EstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { seed, value } => {
                write!(f, "rollout {seed} returned {value}, outside [0, 1]")
            }
            Self::TooManyQubits(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for EstimationError {}

impl From<TooManyQubits> for EstimationError {
    fn from(error: TooManyQubits) -> Self {
        Self::TooManyQubits(error)
    }
}

/// Estimates the mean of `value` over the `2^seed_qubits` seeds with `2^precision_bits`
/// phase-estimation outcomes, and samples one outcome from `rng`.
///
/// With probability at least `8/π²`, the error is at most
/// `2π·√(a(1−a))/K + π²/K²` for `K = 2^precision_bits`. It costs `2K − 1` queries.
///
/// # Errors
///
/// Returns [`EstimationError::InvalidValue`] when a value is not in `[0, 1]`, and
/// [`EstimationError::TooManyQubits`] when `seed_qubits + 1 + precision_bits` exceeds
/// [`MAX_QUBITS`].
pub fn estimate<R: Rng + ?Sized>(
    seed_qubits: usize,
    value: impl Fn(usize) -> f64,
    precision_bits: usize,
    rng: &mut R,
) -> Result<Estimate, EstimationError> {
    let probabilities = outcomes(seed_qubits, value, precision_bits)?;
    let outcome = pick(rng, probabilities.iter().copied());

    Ok(Estimate {
        value: outcome_value(outcome, probabilities.len()),
        queries: 2 * probabilities.len() - 1,
    })
}

/// Returns the probability of each phase-estimation outcome `y < 2^precision_bits`.
///
/// Outcome `y` stands for the estimate [`outcome_value`]`(y, 2^precision_bits)`.
///
/// # Errors
///
/// As [`estimate`].
pub fn outcomes(
    seed_qubits: usize,
    value: impl Fn(usize) -> f64,
    precision_bits: usize,
) -> Result<Vec<f64>, EstimationError> {
    let total = seed_qubits + 1 + precision_bits;

    if total > MAX_QUBITS {
        return Err(TooManyQubits { requested: total }.into());
    }

    let angles = rotation_angles(seed_qubits, value)?;
    let mut register = StateVector::new(seed_qubits + 1)?;
    prepare(&mut register, &angles);

    let start = register.amplitudes().to_vec();

    Ok(outcome_probabilities(
        &start,
        1 << precision_bits,
        |vector| {
            register.amplitudes_mut().copy_from_slice(vector);
            grover_operator(&mut register, &angles);
            vector.copy_from_slice(register.amplitudes());
        },
    ))
}

/// Returns the estimate `sin²(π·y/K)` of outcome `y` out of `outcomes`.
#[must_use]
pub fn outcome_value(outcome: usize, outcomes: usize) -> f64 {
    let sine = libm::sin(PI * ratio(outcome, outcomes));

    sine * sine
}

/// Estimates the mean of `value` classically from `samples` rollouts on uniform seeds.
///
/// Each rollout is one query. The error is about `σ/√samples`, where `σ` is the standard
/// deviation of the values.
///
/// # Errors
///
/// Returns [`EstimationError::InvalidValue`] when a drawn value is not in `[0, 1]`.
///
/// # Panics
///
/// Panics when `samples` is `0`, or when `seed_qubits` is not below `usize::BITS`.
pub fn classical_estimate<R: Rng + ?Sized>(
    seed_qubits: usize,
    value: impl Fn(usize) -> f64,
    samples: usize,
    rng: &mut R,
) -> Result<Estimate, EstimationError> {
    assert!(
        samples > 0,
        "a Monte Carlo estimate needs at least one sample"
    );

    let mut total = 0.0;

    for _ in 0..samples {
        let seed = below(rng, 1 << seed_qubits);
        total += checked_value(seed, value(seed))?;
    }

    Ok(Estimate {
        value: total / ratio(samples, 1),
        queries: samples,
    })
}

/// Returns the rotation angle `θᵢ = asin(√v(i))` of each seed.
///
/// # Errors
///
/// Returns [`EstimationError::InvalidValue`] for the first value outside `[0, 1]`.
fn rotation_angles(
    seed_qubits: usize,
    value: impl Fn(usize) -> f64,
) -> Result<Vec<f64>, EstimationError> {
    (0..1 << seed_qubits)
        .map(|seed| Ok(libm::asin(libm::sqrt(checked_value(seed, value(seed))?))))
        .collect()
}

/// Returns `value` when it is a number in `[0, 1]`.
///
/// # Errors
///
/// Returns [`EstimationError::InvalidValue`] otherwise, NaN included.
fn checked_value(seed: usize, value: f64) -> Result<f64, EstimationError> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(EstimationError::InvalidValue { seed, value })
    }
}

/// Applies `A` to `|0…0⟩` in `register`.
fn prepare(register: &mut StateVector, angles: &[f64]) {
    register.reset();
    rollout(register, angles);
}

/// Applies `A`: `H` on the seeds, then the value rotations.
fn rollout(register: &mut StateVector, angles: &[f64]) {
    hadamard_seeds(register);
    rotate_values(register, angles, 1.0);
}

/// Applies `A⁻¹`: the inverse value rotations, then `H` on the seeds.
fn unroll(register: &mut StateVector, angles: &[f64]) {
    rotate_values(register, angles, -1.0);
    hadamard_seeds(register);
}

/// Applies `Q = −A·S₀·A⁻¹·S₁` to `register`.
fn grover_operator(register: &mut StateVector, angles: &[f64]) {
    let value_bit = 1 << (register.qubits() - 1);

    for (index, amplitude) in register.amplitudes_mut().iter_mut().enumerate() {
        if index & value_bit != 0 {
            *amplitude = -*amplitude;
        }
    }

    unroll(register, angles);
    register.negate(0);
    rollout(register, angles);

    for amplitude in register.amplitudes_mut() {
        *amplitude = -*amplitude;
    }
}

/// Applies `H` to the seed qubits, all but the highest.
fn hadamard_seeds(register: &mut StateVector) {
    for qubit in 0..register.qubits() - 1 {
        register.h(qubit);
    }
}

/// Rotates the value qubit by `RY(2·direction·θᵢ)` where the seeds read `i`; `direction`
/// is `1` for `A` and `−1` for `A⁻¹`.
fn rotate_values(register: &mut StateVector, angles: &[f64], direction: f64) {
    let value_bit = angles.len();
    let amplitudes = register.amplitudes_mut();

    for (seed, &angle) in angles.iter().enumerate() {
        let (sin, cos) = (libm::sin(direction * angle), libm::cos(direction * angle));
        let zero = amplitudes[seed];
        let one = amplitudes[seed | value_bit];

        amplitudes[seed] = zero.scale(cos) - one.scale(sin);
        amplitudes[seed | value_bit] = zero.scale(sin) + one.scale(cos);
    }
}

/// Returns `part / whole` as `f64`.
fn ratio(part: usize, whole: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "outcome and sample counts stay far below 2^53"
    )]
    let ratio = part as f64 / whole as f64;

    ratio
}

/// Unit tests for the state preparation and the Grover operator.
#[cfg(test)]
mod tests {
    use super::{grover_operator, prepare, rotation_angles};
    use crate::StateVector;

    /// `A|0⟩` puts the mean value on the value qubit, and `Q` keeps the norm.
    #[test]
    fn prepares_the_mean_and_keeps_the_norm() {
        let values = [0.0, 0.25, 1.0, 0.5];
        let angles = rotation_angles(2, |seed| values[seed]).unwrap();
        let mut register = StateVector::new(3).unwrap();
        prepare(&mut register, &angles);

        let good: f64 = register.probabilities().skip(4).sum();
        assert!((good - 0.4375).abs() < 1e-12);

        grover_operator(&mut register, &angles);
        assert!((register.probabilities().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
