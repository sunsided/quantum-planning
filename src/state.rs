//! A state-vector simulator: `2^n` complex amplitudes and the gates the algorithms use.
//!
//! # Qubit order
//!
//! The order is little-endian. Qubit `q` is bit `q` of the basis-state index, so qubit 0 is
//! the least significant bit. On three qubits, `X` on qubit 0 moves `|000⟩` (index 0) to
//! index `0b001 = 1`, and `X` on qubit 2 moves it to index `0b100 = 4`. A
//! [`PhaseOracle`](crate::PhaseOracle) gives basis index `i` to its index map, so candidate
//! `i` is the basis state whose bits spell `i`.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::FRAC_1_SQRT_2;
use core::fmt;

use rand_core::Rng;

use crate::Complex;
use crate::rng::unit_f64;

/// The largest register the simulator accepts: `2^24` amplitudes, 256 MiB of state.
pub const MAX_QUBITS: usize = 24;

/// The error for a register with more than [`MAX_QUBITS`] qubits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooManyQubits {
    /// The number of qubits that the caller asked for.
    pub requested: usize,
}

impl fmt::Display for TooManyQubits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} qubits requested, but the simulator holds at most {MAX_QUBITS}",
            self.requested
        )
    }
}

impl core::error::Error for TooManyQubits {}

/// The state of an `n`-qubit register as `2^n` complex amplitudes.
///
/// Gates change the state in place. Reuse one `StateVector` across runs: [`reset`](Self::reset)
/// does not allocate. See the [module docs](self) for the qubit order.
#[derive(Clone, Debug)]
pub struct StateVector {
    /// The number of qubits `n`. At most [`MAX_QUBITS`].
    qubits: usize,
    /// The amplitudes, indexed by basis state. Always `2^n` entries with norm 1 up to rounding.
    amplitudes: Vec<Complex>,
}

impl StateVector {
    /// Makes an `n`-qubit register in the state `|0…0⟩`.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when `qubits` is more than [`MAX_QUBITS`].
    pub fn new(qubits: usize) -> Result<Self, TooManyQubits> {
        if qubits > MAX_QUBITS {
            return Err(TooManyQubits { requested: qubits });
        }

        let mut amplitudes = vec![Complex::ZERO; 1 << qubits];
        amplitudes[0] = Complex::ONE;

        Ok(Self { qubits, amplitudes })
    }

    /// Returns the number of qubits.
    #[must_use]
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Returns the number of basis states, `2^n`.
    #[must_use]
    pub fn states(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns the amplitudes, indexed by basis state.
    #[must_use]
    pub fn amplitudes(&self) -> &[Complex] {
        &self.amplitudes
    }

    /// Sets the register back to `|0…0⟩` without a new allocation.
    pub fn reset(&mut self) {
        self.amplitudes.fill(Complex::ZERO);
        self.amplitudes[0] = Complex::ONE;
    }

    /// Applies the Hadamard gate to `qubit`.
    ///
    /// # Panics
    ///
    /// Panics when `qubit` is not less than [`qubits`](Self::qubits).
    pub fn h(&mut self, qubit: usize) {
        let bit = self.bit(qubit);

        for zero in (0..self.states()).filter(|index| index & bit == 0) {
            let a = self.amplitudes[zero];
            let b = self.amplitudes[zero | bit];

            self.amplitudes[zero] = (a + b).scale(FRAC_1_SQRT_2);
            self.amplitudes[zero | bit] = (a - b).scale(FRAC_1_SQRT_2);
        }
    }

    /// Applies the Pauli-X (NOT) gate to `qubit`.
    ///
    /// # Panics
    ///
    /// Panics when `qubit` is not less than [`qubits`](Self::qubits).
    pub fn x(&mut self, qubit: usize) {
        let bit = self.bit(qubit);

        for zero in (0..self.states()).filter(|index| index & bit == 0) {
            self.amplitudes.swap(zero, zero | bit);
        }
    }

    /// Applies the Pauli-Z gate to `qubit`: the `|1⟩` component changes sign.
    ///
    /// # Panics
    ///
    /// Panics when `qubit` is not less than [`qubits`](Self::qubits).
    pub fn z(&mut self, qubit: usize) {
        self.mcz([qubit]);
    }

    /// Applies the phase gate `diag(1, e^(i·angle))` to `qubit`, with `angle` in radians.
    ///
    /// # Panics
    ///
    /// Panics when `qubit` is not less than [`qubits`](Self::qubits).
    pub fn phase(&mut self, qubit: usize, angle: f64) {
        let bit = self.bit(qubit);
        let factor = Complex::from_angle(angle);

        for (_, amplitude) in self.ones_mut(bit) {
            *amplitude = *amplitude * factor;
        }
    }

    /// Applies the multi-controlled Z gate on `qubits`: a basis state changes sign when all
    /// of these qubits are `1`.
    ///
    /// The gate is symmetric, so no qubit is special as the target. One qubit gives `Z`. An
    /// empty set changes the sign of every state, a global phase without physical effect.
    ///
    /// # Panics
    ///
    /// Panics when a qubit is not less than [`qubits`](Self::qubits).
    pub fn mcz(&mut self, qubits: impl IntoIterator<Item = usize>) {
        let mask = qubits
            .into_iter()
            .fold(0, |mask, qubit| mask | self.bit(qubit));

        for (_, amplitude) in self.ones_mut(mask) {
            *amplitude = -*amplitude;
        }
    }

    /// Changes the sign of the amplitude of the basis state `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not less than [`states`](Self::states).
    pub(crate) fn negate(&mut self, index: usize) {
        self.amplitudes[index] = -self.amplitudes[index];
    }

    /// Returns the measurement probability of each basis state, in index order.
    pub fn probabilities(&self) -> impl Iterator<Item = f64> + '_ {
        self.amplitudes.iter().map(|amplitude| amplitude.norm_sqr())
    }

    /// Measures all qubits once and returns the basis-state index, without a state collapse.
    ///
    /// The draw comes from `rng`, so a fixed seed gives a reproducible result. When rounding
    /// leaves the probabilities a little below 1 in total, the last index takes the rest.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let draw = unit_f64(rng);
        let mut cumulative = 0.0;

        for (index, probability) in self.probabilities().enumerate() {
            cumulative += probability;

            if draw < cumulative {
                return index;
            }
        }

        self.states() - 1
    }

    /// Returns the index mask of `qubit`.
    ///
    /// # Panics
    ///
    /// Panics when `qubit` is not less than [`qubits`](Self::qubits).
    fn bit(&self, qubit: usize) -> usize {
        assert!(
            qubit < self.qubits,
            "qubit {qubit} is outside the {}-qubit register",
            self.qubits
        );

        1 << qubit
    }

    /// Returns the amplitudes of the basis states that have all bits of `mask` set.
    fn ones_mut(&mut self, mask: usize) -> impl Iterator<Item = (usize, &mut Complex)> {
        self.amplitudes
            .iter_mut()
            .enumerate()
            .filter(move |(index, _)| index & mask == mask)
    }
}

/// Unit tests for the gates and the qubit order.
#[cfg(test)]
mod tests {
    use super::{MAX_QUBITS, StateVector, TooManyQubits};
    use crate::SplitMix64;
    use alloc::vec::Vec;

    /// Returns the probabilities of `state` as a vector.
    fn probabilities(state: &StateVector) -> Vec<f64> {
        state.probabilities().collect()
    }

    /// `X` on qubit `q` sets bit `q` of the basis index: the order is little-endian.
    #[test]
    fn orders_qubits_little_endian() {
        let mut state = StateVector::new(3).unwrap();
        state.x(0);
        assert_eq!(probabilities(&state)[0b001], 1.0);

        state.reset();
        state.x(2);
        assert_eq!(probabilities(&state)[0b100], 1.0);

        state.x(0);
        assert_eq!(probabilities(&state)[0b101], 1.0);
    }

    /// `H` gives an even superposition, and a second `H` undoes it.
    #[test]
    fn hadamard_is_its_own_inverse() {
        let mut state = StateVector::new(2).unwrap();
        state.h(1);

        let even = probabilities(&state);
        assert!((even[0b00] - 0.5).abs() < 1e-12 && (even[0b10] - 0.5).abs() < 1e-12);

        state.h(1);
        assert!((probabilities(&state)[0] - 1.0).abs() < 1e-12);
    }

    /// `Z`, the phase gate and `MCZ` change only the phases of the states they select.
    #[test]
    fn phase_gates_select_the_right_states() {
        let mut state = StateVector::new(2).unwrap();
        state.h(0);
        state.h(1);
        state.mcz([0, 1]);

        let signs: Vec<f64> = state.amplitudes().iter().map(|a| a.re.signum()).collect();
        assert_eq!(signs, [1.0, 1.0, 1.0, -1.0]);

        state.z(0);
        state.phase(1, core::f64::consts::PI);

        let signs: Vec<f64> = state.amplitudes().iter().map(|a| a.re.signum()).collect();
        assert_eq!(signs, [1.0, -1.0, -1.0, -1.0]);
    }

    /// Seeded samples follow the probabilities and repeat for the same seed.
    #[test]
    fn samples_follow_the_probabilities() {
        let mut state = StateVector::new(1).unwrap();
        state.phase(0, 1.0);
        state.h(0);

        let mut rng = SplitMix64::new(42);
        let ones = (0..10_000).filter(|_| state.sample(&mut rng) == 1).count();
        assert!((4_800..5_200).contains(&ones), "{ones} ones");

        let first: Vec<usize> = {
            let mut rng = SplitMix64::new(1);
            (0..32).map(|_| state.sample(&mut rng)).collect()
        };
        let second: Vec<usize> = {
            let mut rng = SplitMix64::new(1);
            (0..32).map(|_| state.sample(&mut rng)).collect()
        };
        assert_eq!(first, second);
    }

    /// A register above the limit is an error, not an allocation.
    #[test]
    fn rejects_too_many_qubits() {
        let requested = MAX_QUBITS + 1;

        let error = StateVector::new(requested).unwrap_err();

        assert_eq!(error, TooManyQubits { requested });
        assert_eq!(
            alloc::format!("{error}"),
            "25 qubits requested, but the simulator holds at most 24"
        );
    }
}
