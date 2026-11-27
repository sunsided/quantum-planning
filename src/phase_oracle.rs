//! The bridge from a classical [`Oracle`] to a phase oracle on a [`StateVector`].

use core::marker::PhantomData;

use crate::{Oracle, StateVector};

/// A phase oracle built from an [`Oracle`] and a map from basis index to candidate.
///
/// It changes the sign of each basis state whose candidate the oracle accepts. The map
/// returns `None` for an index without a candidate, for example when the candidate count is
/// not a power of two; such an index is never marked.
///
/// # Query count
///
/// One [`apply`](Self::apply) is one quantum query, and one [`check`](Self::check) is one
/// classical query, as in the query model. The simulator calls [`Oracle::accepts`] once per
/// basis state inside one `apply`; that cost is a property of the simulation, not of the
/// algorithm, so it does not count.
#[derive(Clone, Debug)]
pub struct PhaseOracle<O, F, C> {
    /// The classical predicate that marks the solutions.
    oracle: O,
    /// The map from basis index to candidate.
    candidate: F,
    /// The quantum and classical queries since construction.
    queries: usize,
    /// Ties the candidate type to the oracle and the map.
    marker: PhantomData<fn() -> C>,
}

impl<O, F, C> PhaseOracle<O, F, C>
where
    O: Oracle<C>,
    F: Fn(usize) -> Option<C>,
{
    /// Makes a phase oracle from `oracle` and the index map `candidate`.
    pub fn new(oracle: O, candidate: F) -> Self {
        Self {
            oracle,
            candidate,
            queries: 0,
            marker: PhantomData,
        }
    }

    /// Returns the quantum and classical queries since construction.
    #[must_use]
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Changes the sign of every marked basis state of `state`. This counts as one query.
    pub fn apply(&mut self, state: &mut StateVector) {
        self.queries += 1;

        for index in 0..state.states() {
            if self.marks(index) {
                state.negate(index);
            }
        }
    }

    /// Checks classically whether basis index `index` is marked. This counts as one query.
    pub fn check(&mut self, index: usize) -> bool {
        self.queries += 1;

        self.marks(index)
    }

    /// Returns the basis indices below `states` that the oracle marks, in index order.
    ///
    /// This evaluates the predicate once per index, classically, and counts no query. It
    /// compiles the oracle to a marked set for circuit export
    /// ([`grover::circuit`](crate::grover::circuit)); it is not part of a search. For an
    /// `n`-qubit register, pass `states = 2^n`.
    pub fn marked(&self, states: usize) -> impl Iterator<Item = usize> + '_ {
        (0..states).filter(|&index| self.marks(index))
    }

    /// Returns whether `index` maps to a candidate that the oracle accepts. No query counts.
    fn marks(&self, index: usize) -> bool {
        (self.candidate)(index).is_some_and(|candidate| self.oracle.accepts(&candidate))
    }
}
