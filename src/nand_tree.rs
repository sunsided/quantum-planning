//! Game-tree evaluation by a quantum walk on the NAND tree (Farhi, Goldstone and Gutmann;
//! Childs, Cleve, Jordan and Yonge-Mallo).
//!
//! In a two-player game where the player to move wins when some move leaves the opponent
//! in a lost position, the value of a node is the NAND of its children:
//! `win(v) = ¬(win(c₁) ∧ … ∧ win(c_d))`. The leaves are the inputs. Classical evaluation of
//! a balanced binary tree with `N` leaves needs `Θ(N^0.753)` leaf queries even when
//! randomized; a quantum walk needs time `O(√N)`.
//!
//! # The walk
//!
//! The graph is the tree with one extra leaf below each input that is `true`, and one extra
//! node `r'` above the root, joined to it with the weight `w`. Its weighted adjacency
//! matrix `H` is the Hamiltonian. `H` has an eigenvector of eigenvalue `0` with weight on
//! `|r'⟩` exactly when the root is `true`: at energy 0, the ratio of a node's amplitude to its
//! parent's amplitude is `0` or `∞`, and it follows the NAND rule from the leaves up. When
//! the root is `false`, the spectrum seen from `|r'⟩` has a gap around 0 of order `1/√N`.
//! Phase estimation of `e^(iHτ)` from `|r'⟩` with `K` outcomes tells the two apart in
//! outcome `0`.
//!
//! ```mermaid
//! graph LR
//!     inputs["leaf inputs"] --> walkgraph["tree, extra leaf per true input, r' above the root"]
//!     walkgraph --> evolve["e^(iHτ): continuous-time walk"]
//!     evolve --> qpe["phase estimation from r', K outcomes"]
//!     qpe -->|"outcome 0"| value["root is true"]
//! ```
//!
//! # Cost
//!
//! The cost of a continuous-time walk is its evolution time: the inputs sit in `H`, so
//! time plays the part of queries. One run of phase estimation evolves for `(K − 1)·τ`,
//! between `17·√N` and `34·√N` for binary trees (`K` rounds up to a power of two);
//! [`Evaluation::time`] reports it. One classical leaf read is one query. The constants put
//! the crossover with the `N` reads of a worst-case classical evaluation between about 300
//! and 1200 leaves, past what the tests simulate.
//!
//! # Simulation
//!
//! The simulator applies `e^(iHτ)` by its Taylor series to machine precision (`‖Hτ‖ ≤ π`).

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

use rand_core::Rng;

use crate::phase_estimation::zero_outcome_probability;
use crate::rng::unit_f64;
use crate::{Complex, MAX_QUBITS, TooManyQubits};

/// A balanced NAND tree: every inner node has `fan_out` children, and all leaves are at
/// `depth`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NandTree {
    /// The children of every inner node, at least 2.
    fan_out: usize,
    /// The depth of the leaves below the root.
    depth: usize,
}

/// The result of a quantum evaluation: the root value and the evolution time it cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    /// Whether more than a fifth of the runs measured outcome `0`: the root is `true`.
    pub value: bool,
    /// The total evolution time of all runs.
    pub time: f64,
}

/// The result of a classical evaluation: the root value and the leaves read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassicalEvaluation {
    /// The value of the root.
    pub value: bool,
    /// The leaf queries.
    pub queries: usize,
}

impl NandTree {
    /// Makes the balanced tree with `fan_out` children per inner node and leaves at `depth`.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the walk graph can have more than `2^MAX_QUBITS` nodes:
    /// the tree, one extra leaf per input (all inputs `true`) and `r'`. A graph node holds its
    /// edges and amplitude, so the limit costs more memory than a [`MAX_QUBITS`] state vector.
    ///
    /// # Panics
    ///
    /// Panics when `fan_out` is less than 2.
    pub fn new(fan_out: usize, depth: usize) -> Result<Self, TooManyQubits> {
        assert!(
            fan_out >= 2,
            "a NAND tree needs at least two children per node"
        );

        let too_many = TooManyQubits {
            requested: MAX_QUBITS + 1,
        };

        if walk_nodes(fan_out, depth).is_none_or(|nodes| nodes > 1 << MAX_QUBITS) {
            return Err(too_many);
        }

        Ok(Self { fan_out, depth })
    }

    /// Returns the number of leaves, `fan_out^depth`.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.fan_out.pow(self.exponent())
    }

    /// Returns the number of tree nodes, leaves included.
    #[must_use]
    pub fn nodes(&self) -> usize {
        (0..=self.exponent())
            .map(|level| self.fan_out.pow(level))
            .sum()
    }

    /// Returns the depth as the exponent of `fan_out`.
    fn exponent(&self) -> u32 {
        u32::try_from(self.depth).expect("new bounds the depth")
    }

    /// Returns the index of the first leaf; nodes are in heap order from the root `0`.
    fn first_leaf(&self) -> usize {
        self.nodes() - self.leaves()
    }

    /// Returns the children of inner node `node` in heap order.
    fn children(&self, node: usize) -> impl Iterator<Item = usize> + use<> {
        let first = self.fan_out * node + 1;

        first..first + self.fan_out
    }
}

/// Returns the most nodes the walk graph of the balanced tree can have: every tree node, one
/// extra leaf per input and `r'`; `None` on overflow.
fn walk_nodes(fan_out: usize, depth: usize) -> Option<usize> {
    let mut level = 1_usize;
    let mut nodes = 1_usize;

    for _ in 0..depth {
        level = level.checked_mul(fan_out)?;
        nodes = nodes.checked_add(level)?;
    }

    nodes.checked_add(level)?.checked_add(1)
}

/// Returns the number `K` of phase-estimation outcomes for `tree`: the next power of two of
/// `16·√N`.
///
/// Over every input of the small trees of the tests and random inputs up to depth 6, the
/// constant 16 keeps the acceptance of a `false` root at most `0.05` and that of a `true`
/// root at least `0.35`; the constant 1 leaves the two overlapping.
#[must_use]
pub fn phase_outcomes(tree: &NandTree) -> usize {
    let scale = libm::ceil(16.0 * libm::sqrt(ratio(tree.leaves())));

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the leaf count is at most 2^22, so the value is positive and below 2^16"
    )]
    let scale = scale as usize;

    scale.next_power_of_two()
}

/// Returns the probability that one phase-estimation run from `|r'⟩` measures outcome `0`.
///
/// It is at least about `0.35` when the root is `true` and at most about `0.05` when it is
/// `false` (see [`phase_outcomes`]).
///
/// # Panics
///
/// Panics when `inputs` does not have one value per leaf.
#[must_use]
pub fn acceptance_probability(tree: &NandTree, inputs: &[bool]) -> f64 {
    Walk::new(tree, inputs).acceptance()
}

/// Evaluates the root of `tree` on `inputs` with `runs` phase-estimation runs, sampled from
/// `rng`: the root is `true` when more than a fifth of the runs measure outcome `0`.
///
/// The threshold `1/5` lies between the acceptance probabilities of `true` and `false`
/// roots, so the error falls exponentially in `runs`.
///
/// # Panics
///
/// As [`acceptance_probability`], and when `runs` is `0`.
pub fn evaluate<R: Rng + ?Sized>(
    tree: &NandTree,
    inputs: &[bool],
    runs: usize,
    rng: &mut R,
) -> Evaluation {
    assert!(runs > 0, "an evaluation needs at least one run");

    let walk = Walk::new(tree, inputs);
    let acceptance = walk.acceptance();
    let accepted = (0..runs).filter(|_| unit_f64(rng) < acceptance).count();

    Evaluation {
        value: 5 * accepted > runs,
        time: ratio(runs * (walk.outcomes - 1)) * walk.tau,
    }
}

/// Evaluates the root of `tree` on `inputs` depth first, left to right, and skips the rest
/// of a node's children as soon as one is `false`.
///
/// # Panics
///
/// As [`acceptance_probability`].
#[must_use]
pub fn classical_evaluate(tree: &NandTree, inputs: &[bool]) -> ClassicalEvaluation {
    assert_eq!(inputs.len(), tree.leaves(), "one input per leaf");

    let mut queries = 0;
    let value = classical_value(tree, inputs, 0, &mut queries);

    ClassicalEvaluation { value, queries }
}

/// Returns the value of `node`, counting each leaf read in `queries`.
fn classical_value(tree: &NandTree, inputs: &[bool], node: usize, queries: &mut usize) -> bool {
    if node >= tree.first_leaf() {
        *queries += 1;

        return inputs[node - tree.first_leaf()];
    }

    !tree
        .children(node)
        .all(|child| classical_value(tree, inputs, child, queries))
}

/// The walk graph of one tree and input, with the evolution step.
struct Walk {
    /// The weighted neighbours of each graph node; `r'` is the last node.
    neighbours: Vec<Vec<(usize, f64)>>,
    /// The evolution time of one step, chosen so that `‖Hτ‖ ≤ π`.
    tau: f64,
    /// The number `K` of phase-estimation outcomes.
    outcomes: usize,
}

impl Walk {
    /// Builds the graph of `tree` on `inputs`.
    ///
    /// The weight of the edge from `r'` to the root is `N^(−1/4)`, which keeps the
    /// eigenvalue-0 weight on `|r'⟩` large when the root is `true`, while the coupling stays
    /// below the gap of order `N^(−1/2)` when it is `false`.
    ///
    /// # Panics
    ///
    /// Panics when `inputs` does not have one value per leaf.
    fn new(tree: &NandTree, inputs: &[bool]) -> Self {
        assert_eq!(inputs.len(), tree.leaves(), "one input per leaf");

        let mut neighbours = vec![Vec::new(); tree.nodes()];

        for node in 0..tree.first_leaf() {
            for child in tree.children(node) {
                connect(&mut neighbours, node, child, 1.0);
            }
        }

        for (leaf, _) in inputs.iter().enumerate().filter(|(_, input)| **input) {
            let extra = neighbours.len();
            neighbours.push(Vec::new());
            connect(&mut neighbours, tree.first_leaf() + leaf, extra, 1.0);
        }

        let r_prime = neighbours.len();
        neighbours.push(Vec::new());
        connect(
            &mut neighbours,
            0,
            r_prime,
            libm::pow(ratio(tree.leaves()), -0.25),
        );

        let degree = ratio(tree.fan_out + 1);

        Self {
            neighbours,
            tau: PI / degree,
            outcomes: phase_outcomes(tree),
        }
    }

    /// Returns the probability of outcome `0` of phase estimation from `|r'⟩`.
    fn acceptance(&self) -> f64 {
        let mut start = vec![Complex::ZERO; self.neighbours.len()];
        start[self.neighbours.len() - 1] = Complex::ONE;

        zero_outcome_probability(&start, self.outcomes, |state| self.evolve(state))
    }

    /// Applies `e^(iHτ)` to `state` by its Taylor series, until the terms vanish.
    fn evolve(&self, state: &mut [Complex]) {
        let mut term = state.to_vec();
        let mut order = 1.0;

        while term.iter().any(|amplitude| amplitude.norm_sqr() > 1e-36) {
            term = self.times_i_h(&term, self.tau / order);

            for (total, amplitude) in state.iter_mut().zip(&term) {
                *total = *total + *amplitude;
            }

            order += 1.0;
        }
    }

    /// Returns `i·scale·H·vector`.
    fn times_i_h(&self, vector: &[Complex], scale: f64) -> Vec<Complex> {
        let i_scale = Complex::new(0.0, scale);

        self.neighbours
            .iter()
            .map(|edges| {
                edges.iter().fold(Complex::ZERO, |sum, &(other, weight)| {
                    sum + i_scale * vector[other].scale(weight)
                })
            })
            .collect()
    }
}

/// Adds the undirected edge `a`–`b` with `weight`.
fn connect(neighbours: &mut [Vec<(usize, f64)>], a: usize, b: usize, weight: f64) {
    neighbours[a].push((b, weight));
    neighbours[b].push((a, weight));
}

/// Converts a count to `f64`.
fn ratio(count: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "node and step counts are at most 2^24, exact in f64"
    )]
    let value = count as f64;

    value
}

/// Unit tests for the evolution step.
#[cfg(test)]
mod tests {
    use super::{NandTree, Walk};
    use crate::Complex;
    use alloc::vec;

    /// The Taylor evolution keeps the norm.
    #[test]
    fn evolution_is_unitary() {
        let tree = NandTree::new(2, 2).unwrap();
        let walk = Walk::new(&tree, &[true, false, true, true]);
        let mut state = vec![Complex::ZERO; walk.neighbours.len()];
        state[0] = Complex::ONE;
        state[3] = Complex::new(0.0, 1.0);

        for _ in 0..50 {
            walk.evolve(&mut state);
        }

        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 2.0).abs() < 1e-9, "{norm}");
    }
}
