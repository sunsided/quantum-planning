//! QAOA (Farhi, Goldstone and Gutmann) for QUBO-encoded planning problems from
//! [`planning_qubo`].
//!
//! A QUBO model gives each assignment `x ∈ {0, 1}ⁿ` an energy `E(x)`; low energy means a
//! good plan. QAOA with `p` layers prepares
//!
//! ```text
//! |γ, β⟩ = U_B(β_p)·U_C(γ_p) ⋯ U_B(β_1)·U_C(γ_1) · H^⊗n |0⟩,
//! U_C(γ) = e^(−iγ·E),   U_B(β) = e^(−iβ·Σ X_q),
//! ```
//!
//! tunes `γ` and `β` classically to lower the expected energy `⟨γ, β|E|γ, β⟩`, and then
//! samples assignments. Variable `q` is qubit `q`, so basis index `i` is the assignment
//! whose bits spell `i` (see the [simulator](crate::StateVector) for the order).
//!
//! ```mermaid
//! graph LR
//!     qubo["Qubo: offset, linear, pairs"] --> energies["E(x) per basis state"]
//!     energies --> layers["p layers of U_C(γ), U_B(β)"]
//!     layers --> expect["expected energy"]
//!     expect -->|"optimize γ, β"| layers
//!     layers --> sample["sample shots, keep the lowest energy"]
//!     layers -->|circuit| qasm["OpenQASM 3: rz, cx, rx"]
//! ```
//!
//! # Circuit
//!
//! With `xᵢ = (1 − Zᵢ)/2`, the energy is `c + Σ hᵢ·Zᵢ + Σ Jᵢⱼ·Zᵢ·Zⱼ`, so `U_C(γ)` is
//! `rz(2γ·hᵢ)` on each qubit and `cx`, `rz(2γ·Jᵢⱼ)`, `cx` on each pair, up to the global
//! phase `e^(−iγc)`; `U_B(β)` is `rx(2β)` on each qubit. [`Qaoa::circuit`] records that
//! for export through [`Circuit::qasm`].
//!
//! # Cost
//!
//! The classical loop counts [`Parameters::evaluations`], each one expectation value; on
//! hardware that is a batch of shots. Sampling counts shots. The exhaustive baseline
//! evaluates all `2ⁿ` energies.

use alloc::vec::Vec;

use planning_qubo::{Coefficient, Qubo};
use rand_core::Rng;

use crate::rng::unit_f64;
use crate::{Circuit, Complex, MAX_QUBITS, StateVector, TooManyQubits};

/// A QAOA instance: the energy of every assignment and the Ising terms for the circuit.
#[derive(Clone, Debug)]
pub struct Qaoa {
    /// The number of variables and qubits.
    qubits: usize,
    /// The energy of each basis state, `E(i)`.
    energies: Vec<f64>,
    /// The Ising field `hᵢ` of each qubit.
    fields: Vec<f64>,
    /// The Ising couplings `(i, j, Jᵢⱼ)` with `i < j`.
    couplings: Vec<(usize, usize, f64)>,
}

/// QAOA angles for `p` layers, with the expected energy they reach.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameters {
    /// The phase angles `γ₁ … γ_p`.
    pub gammas: Vec<f64>,
    /// The mixer angles `β₁ … β_p`.
    pub betas: Vec<f64>,
    /// The expected energy `⟨γ, β|E|γ, β⟩`.
    pub expectation: f64,
    /// The expectation values computed to find the angles.
    pub evaluations: usize,
}

/// A sampled assignment with its energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solution {
    /// The basis index: bit `q` is the value of variable `q`.
    pub index: usize,
    /// The energy of the assignment.
    pub energy: f64,
    /// The shots or energy evaluations spent.
    pub queries: usize,
}

impl Qaoa {
    /// Makes the QAOA instance of `qubo`, with energies in `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the model has more than [`MAX_QUBITS`] variables.
    pub fn new<T: Coefficient>(qubo: &Qubo<T>) -> Result<Self, TooManyQubits> {
        let qubits = qubo.num_variables();

        if qubits > MAX_QUBITS {
            return Err(TooManyQubits { requested: qubits });
        }

        let linear: Vec<f64> = (0..qubits).map(|i| qubo.linear(i).to_f64()).collect();
        let pairs: Vec<(usize, usize, f64)> = qubo
            .quadratic_terms()
            .map(|(i, j, b)| (i, j, b.to_f64()))
            .collect();

        let energies = (0..1 << qubits)
            .map(|index| energy(qubo.offset().to_f64(), &linear, &pairs, index))
            .collect();
        let (fields, couplings) = ising_terms(&linear, &pairs);

        Ok(Self {
            qubits,
            energies,
            fields,
            couplings,
        })
    }

    /// Returns the number of variables and qubits.
    #[must_use]
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Returns the energy of basis state `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below `2^qubits`.
    #[must_use]
    pub fn energy(&self, index: usize) -> f64 {
        self.energies[index]
    }

    /// Prepares `|γ, β⟩` in `state`.
    ///
    /// # Panics
    ///
    /// Panics when `gammas` and `betas` differ in length, or when `state` has a different
    /// qubit count.
    pub fn prepare(&self, gammas: &[f64], betas: &[f64], state: &mut StateVector) {
        assert_eq!(gammas.len(), betas.len(), "one β per γ");
        assert_eq!(state.qubits(), self.qubits, "one qubit per variable");

        state.reset();

        for qubit in 0..self.qubits {
            state.h(qubit);
        }

        for (&gamma, &beta) in gammas.iter().zip(betas) {
            self.separate_phases(state, gamma);

            for qubit in 0..self.qubits {
                state.rx(qubit, 2.0 * beta);
            }
        }
    }

    /// Returns the expected energy of `|γ, β⟩`.
    ///
    /// # Panics
    ///
    /// As [`prepare`](Self::prepare).
    #[must_use]
    pub fn expectation(&self, gammas: &[f64], betas: &[f64]) -> f64 {
        let mut state = StateVector::new(self.qubits).expect("new checked the qubit count");
        self.prepare(gammas, betas, &mut state);

        state
            .probabilities()
            .zip(&self.energies)
            .map(|(probability, energy)| probability * energy)
            .sum()
    }

    /// Records the circuit of `|γ, β⟩` with `rz`, `cx` and `rx`, measuring every qubit.
    ///
    /// It prepares the state of [`prepare`](Self::prepare) up to a global phase.
    ///
    /// # Panics
    ///
    /// Panics when `gammas` and `betas` differ in length.
    #[must_use]
    pub fn circuit(&self, gammas: &[f64], betas: &[f64]) -> Circuit {
        assert_eq!(gammas.len(), betas.len(), "one β per γ");

        let mut circuit = Circuit::new(self.qubits).expect("new checked the qubit count");

        for qubit in 0..self.qubits {
            circuit.h(qubit);
        }

        for (&gamma, &beta) in gammas.iter().zip(betas) {
            self.record_phases(&mut circuit, gamma);

            for qubit in 0..self.qubits {
                circuit.rx(qubit, 2.0 * beta);
            }
        }

        circuit.measure_all();

        circuit
    }

    /// Tunes `layers` pairs of angles from `restarts` random starts by coordinate descent
    /// and returns the angles with the lowest expected energy.
    ///
    /// Each start draws `γ` from `[0, π/s]`, with `s` the energy spread, and `β` from
    /// `[0, π/2]`. The descent tries a step up and down on each angle, halves the step when
    /// nothing improves, and stops below a step of `10⁻³` of the range.
    ///
    /// # Panics
    ///
    /// Panics when `restarts` is `0`.
    pub fn optimize<R: Rng + ?Sized>(
        &self,
        layers: usize,
        restarts: usize,
        rng: &mut R,
    ) -> Parameters {
        assert!(restarts > 0, "the optimizer needs at least one start");

        let ranges = self.angle_ranges();
        let mut best: Option<Parameters> = None;
        let mut evaluations = 0;

        for _ in 0..restarts {
            let candidate = self.descend(layers, ranges, rng);
            evaluations += candidate.evaluations;

            if best
                .as_ref()
                .is_none_or(|best| candidate.expectation < best.expectation)
            {
                best = Some(candidate);
            }
        }

        let mut best = best.expect("at least one restart ran");
        best.evaluations = evaluations;

        best
    }

    /// Samples `shots` assignments from `|γ, β⟩` and returns the one with the lowest
    /// energy.
    ///
    /// # Panics
    ///
    /// As [`prepare`](Self::prepare), and when `shots` is `0`.
    pub fn sample_best<R: Rng + ?Sized>(
        &self,
        parameters: &Parameters,
        shots: usize,
        rng: &mut R,
    ) -> Solution {
        assert!(shots > 0, "sampling needs at least one shot");

        let mut state = StateVector::new(self.qubits).expect("new checked the qubit count");
        self.prepare(&parameters.gammas, &parameters.betas, &mut state);

        let index = (0..shots)
            .map(|_| state.sample(rng))
            .min_by(|&a, &b| self.energies[a].total_cmp(&self.energies[b]))
            .expect("at least one shot");

        Solution {
            index,
            energy: self.energies[index],
            queries: shots,
        }
    }

    /// Returns the assignment of lowest energy by evaluating all `2ⁿ` energies, the
    /// classical baseline.
    #[must_use]
    pub fn exhaustive_minimum(&self) -> Solution {
        let index = (0..self.energies.len())
            .min_by(|&a, &b| self.energies[a].total_cmp(&self.energies[b]))
            // A register always has at least the basis state 0.
            .unwrap_or(0);

        Solution {
            index,
            energy: self.energies[index],
            queries: self.energies.len(),
        }
    }

    /// Applies `U_C(γ) = e^(−iγ·E)` to `state`, one phase per basis state.
    fn separate_phases(&self, state: &mut StateVector, gamma: f64) {
        for (amplitude, &energy) in state.amplitudes_mut().iter_mut().zip(&self.energies) {
            *amplitude = *amplitude * Complex::from_angle(-gamma * energy);
        }
    }

    /// Records `U_C(γ)` from the Ising terms, without the global phase.
    fn record_phases(&self, circuit: &mut Circuit, gamma: f64) {
        for (qubit, &field) in self.fields.iter().enumerate() {
            if field != 0.0 {
                circuit.rz(qubit, 2.0 * gamma * field);
            }
        }

        for &(i, j, coupling) in &self.couplings {
            circuit.cx(i, j);
            circuit.rz(j, 2.0 * gamma * coupling);
            circuit.cx(i, j);
        }
    }

    /// Returns the ranges of the `γ` and `β` draws: `π` over the energy spread, and `π/2`.
    fn angle_ranges(&self) -> (f64, f64) {
        let (low, high) = self
            .energies
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &e| {
                (low.min(e), high.max(e))
            });
        let spread = if high > low { high - low } else { 1.0 };

        (core::f64::consts::PI / spread, core::f64::consts::FRAC_PI_2)
    }

    /// Runs one coordinate descent from a random start within `ranges`.
    fn descend<R: Rng + ?Sized>(
        &self,
        layers: usize,
        (gamma_range, beta_range): (f64, f64),
        rng: &mut R,
    ) -> Parameters {
        let mut gammas: Vec<f64> = (0..layers).map(|_| gamma_range * unit_f64(rng)).collect();
        let mut betas: Vec<f64> = (0..layers).map(|_| beta_range * unit_f64(rng)).collect();
        let mut expectation = self.expectation(&gammas, &betas);
        let mut evaluations = 1;
        let mut step = 0.25;

        while step > 1e-3 {
            let mut improved = false;

            for coordinate in 0..2 * layers {
                for direction in [1.0, -1.0] {
                    let (angle, range) = if coordinate < layers {
                        (&mut gammas[coordinate], gamma_range)
                    } else {
                        (&mut betas[coordinate - layers], beta_range)
                    };
                    let old = *angle;
                    *angle += direction * step * range;

                    let trial = self.expectation(&gammas, &betas);
                    evaluations += 1;

                    if trial < expectation {
                        expectation = trial;
                        improved = true;
                    } else if coordinate < layers {
                        gammas[coordinate] = old;
                    } else {
                        betas[coordinate - layers] = old;
                    }
                }
            }

            if !improved {
                step /= 2.0;
            }
        }

        Parameters {
            gammas,
            betas,
            expectation,
            evaluations,
        }
    }
}

/// Returns the energy of basis state `index`: bit `q` is variable `q`.
fn energy(offset: f64, linear: &[f64], pairs: &[(usize, usize, f64)], index: usize) -> f64 {
    let bit = |q: usize| index >> q & 1 == 1;

    let linear: f64 = linear
        .iter()
        .enumerate()
        .filter(|&(q, _)| bit(q))
        .map(|(_, a)| a)
        .sum();
    let pairs: f64 = pairs
        .iter()
        .filter(|&&(i, j, _)| bit(i) && bit(j))
        .map(|&(_, _, b)| b)
        .sum();

    offset + linear + pairs
}

/// Returns the Ising fields and couplings of the QUBO terms, with `xᵢ = (1 − Zᵢ)/2`.
///
/// `a·xᵢ` gives `hᵢ −= a/2`; `b·xᵢ·xⱼ` gives `hᵢ −= b/4`, `hⱼ −= b/4` and `Jᵢⱼ = b/4`.
fn ising_terms(
    linear: &[f64],
    pairs: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let mut fields: Vec<f64> = linear.iter().map(|a| -a / 2.0).collect();
    let mut couplings = Vec::with_capacity(pairs.len());

    for &(i, j, b) in pairs {
        fields[i] -= b / 4.0;
        fields[j] -= b / 4.0;
        couplings.push((i, j, b / 4.0));
    }

    (fields, couplings)
}
