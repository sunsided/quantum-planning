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

/// Unit tests for the `Oracle` trait, through a minimal plan-validity oracle.
#[cfg(test)]
mod tests {
    use super::Oracle;

    /// Accepts 4-bit action masks that run exactly two actions, but not actions 0 and 1
    /// together.
    struct ValidPlan;

    impl Oracle<u8> for ValidPlan {
        fn accepts(&self, plan: &u8) -> bool {
            plan.count_ones() == 2 && plan & 0b0011 != 0b0011
        }
    }

    /// Scans `candidates` classically and returns the first accepted one together with the
    /// number of oracle queries spent, the baseline a quantum search is measured against.
    fn first_marked<C, O: Oracle<C>>(oracle: &O, candidates: &[C]) -> (Option<usize>, usize) {
        let mut queries = 0;
        for (index, candidate) in candidates.iter().enumerate() {
            queries += 1;
            if oracle.accepts(candidate) {
                return (Some(index), queries);
            }
        }

        (None, queries)
    }

    /// The oracle marks exactly the valid plans.
    #[test]
    fn marks_the_valid_plans() {
        let marked = (0..16_u8).filter(|plan| ValidPlan.accepts(plan)).count();

        assert_eq!(marked, 5);
        assert!(ValidPlan.accepts(&0b1010));
        assert!(!ValidPlan.accepts(&0b0011));
        assert!(!ValidPlan.accepts(&0b0111));
    }

    /// A classical scan stops at the first marked candidate and counts every query.
    #[test]
    fn classical_scan_counts_queries() {
        let plans: [u8; 4] = [0b0000, 0b0011, 0b0101, 0b0110];

        assert_eq!(first_marked(&ValidPlan, &plans), (Some(2), 3));
        assert_eq!(first_marked(&ValidPlan, &plans[..2]), (None, 2));
    }
}
