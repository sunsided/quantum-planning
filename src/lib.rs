//! # Quantum planning
//!
//! Quantum algorithms for planning and search, on simulators: Grover search, quantum backtracking, amplitude estimation, quantum walks on game trees and QAOA.
//!
//! ## Scope
//!
//! - Grover search over candidate plans, with oracles built from validity checks
//! - Quantum backtracking (Montanaro) as a quantum walk on search trees
//! - Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts
//! - Quantum-walk evaluation of game trees (NAND trees)
//! - QAOA circuits for QUBO-encoded planning problems
//! - A simulator interface, with query-count comparisons against the classical algorithms
//!
//! Repository: <https://github.com/sunsided/quantum-planning>

#![no_std]
#![forbid(unsafe_code)]

/// A predicate that marks valid candidates, the oracle of a quantum search.
pub trait Oracle<Candidate> {
    /// Whether `candidate` is a solution.
    fn accepts(&self, candidate: &Candidate) -> bool;
}

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
