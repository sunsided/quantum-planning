//! # Quantum planning
//!
//! Quantum algorithms for planning and search, on simulators: Grover search, quantum backtracking, amplitude estimation, quantum walks on game trees and QAOA.
//!
//! ## Scope
//!
//! - A state-vector simulator ([`StateVector`]) with `H`, `X`, `Z`, phase, `RX`, `RZ`, `CX`
//!   and multi-controlled `Z` gates, measurement probabilities and seeded sampling over any
//!   [`rand_core::Rng`]
//! - Recorded circuits ([`Circuit`]) that run on the simulator and export as OpenQASM 3 for
//!   gate-based hardware, with multi-controlled `Z` lowered to `stdgates.inc`, depth and
//!   two-qubit gate counts, and hardware counts mapped back to candidates ([`circuit`])
//! - Grover search over candidate plans, with phase oracles ([`PhaseOracle`]) built from
//!   validity checks, for a known ([`grover::search_known`]) and an unknown
//!   ([`grover::search_unknown`], BBHT) number of solutions, and as an exportable circuit
//!   ([`grover::circuit`])
//! - Quantum backtracking (Montanaro) as a quantum walk on search trees (planned)
//! - Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts (planned)
//! - Quantum-walk evaluation of game trees (NAND trees) (planned)
//! - QAOA circuits for QUBO-encoded planning problems (planned)
//! - Query-count comparisons against the classical algorithms
//!
//! ## Query count
//!
//! The algorithms count oracle queries, not simulator work: one application of a
//! [`PhaseOracle`] is one query, and one classical check of a measured candidate is one
//! query. Compare these counts with the queries of a classical scan.
//!
//! The crate is `no_std` and needs `alloc` for the amplitudes.
//!
//! Repository: <https://github.com/sunsided/quantum-planning>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod circuit;
mod complex;
pub mod grover;
mod phase_oracle;
mod rng;
mod state;

pub use circuit::{Circuit, Gate};
pub use complex::Complex;
pub use phase_oracle::PhaseOracle;
pub use rng::SplitMix64;
pub use state::{MAX_QUBITS, StateVector, TooManyQubits};

/// A predicate that marks valid candidates, the oracle of a quantum search.
///
/// The predicate must be pure: the same candidate always gets the same answer, since the
/// algorithms count oracle queries.
pub trait Oracle<Candidate> {
    /// Whether `candidate` is a solution.
    fn accepts(&self, candidate: &Candidate) -> bool;
}

/// A reference to an oracle is an oracle, so a [`PhaseOracle`] can borrow one.
impl<Candidate, O: Oracle<Candidate> + ?Sized> Oracle<Candidate> for &O {
    fn accepts(&self, candidate: &Candidate) -> bool {
        (**self).accepts(candidate)
    }
}

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
