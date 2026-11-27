//! Quantum backtracking (Montanaro): a quantum walk on the backtracking tree.
//!
//! Classical backtracking explores a tree of partial solutions: a predicate accepts a node
//! as a solution, rejects it, or lets it branch into children. Depth-first search needs up
//! to `T` predicate queries on a tree of `T` nodes. Montanaro's walk detects a solution in
//! `O(√(T·n))` walk steps, where `n` bounds the depth, and finds one in `O(√(T·n)·n)` up to
//! logarithmic factors.
//!
//! # The walk
//!
//! Nodes at even depth form the set `A` (the root `r` included), nodes at odd depth the set
//! `B`. For an unsolved node `x`, the star state on `x` and its children is
//!
//! ```text
//! |ψ_x⟩ ∝ |x⟩ + Σ_children |y⟩               (x ≠ r)
//! |ψ_r⟩ ∝ |r⟩ + √n · Σ_children |y⟩
//! ```
//!
//! and `D_x = I − 2|ψ_x⟩⟨ψ_x|` on the span of the star. For a solution, `D_x = I`. The
//! walk step is `R_B·R_A`, with `R_A` the product of `D_x` over `A` and `R_B` that over `B`
//! (identity on the root). The stars within one set are disjoint, so each product is one
//! reflection per node. A solution in the tree gives an eigenvalue-1 eigenvector with weight
//! at least `1/2` on `|r⟩`; without one, `|r⟩` lies mostly on eigenphases of size
//! `Ω(1/√(T·n))`. Phase estimation from `|r⟩` with [`phase_outcomes`] outcomes sees the
//! difference in outcome `0`.
//!
//! ```mermaid
//! graph LR
//!     tree["SearchTree: root, children, is_solution"] --> materialize["nodes, parents, depths"]
//!     materialize --> walk["walk step R_B·R_A"]
//!     walk --> qpe["phase estimation from the root"]
//!     qpe -->|"outcome 0"| detect["a solution exists"]
//!     detect -->|"descend into a child that detects"| find["find a solution"]
//! ```
//!
//! # Query count
//!
//! One walk step is one query: it evaluates the predicate on the nodes of the stars it
//! reflects, as one quantum query. One classical check of a node is one query. A detection
//! run costs `K − 1` walk steps for `K` = [`phase_outcomes`].
//!
//! # Simulation
//!
//! The simulator lists the tree once to build the walk; that listing is not counted, as it
//! stands in for the quantum memory of the walk. It uses the exact node count `T` and depth
//! `n` as the bounds that Montanaro's algorithm takes as input.

use alloc::vec;
use alloc::vec::Vec;

use rand_core::Rng;

use crate::phase_estimation::zero_outcome_probability;
use crate::rng::unit_f64;
use crate::{Complex, MAX_QUBITS, TooManyQubits};

/// A backtracking tree: a root, the children of a node and the solution predicate.
///
/// `children` of a rejected node or of a solution are ignored by the walk: a solution ends
/// its branch. The predicate must be pure, as for [`Oracle`](crate::Oracle).
pub trait SearchTree {
    /// A node of the tree: a partial solution.
    type Node: Clone;

    /// Returns the root, the empty partial solution.
    fn root(&self) -> Self::Node;

    /// Returns the children of `node`, empty when the predicate rejects it.
    fn children(&self, node: &Self::Node) -> impl IntoIterator<Item = Self::Node>;

    /// Returns whether `node` is a solution.
    fn is_solution(&self, node: &Self::Node) -> bool;
}

/// The result of a detection: whether a solution was seen and what it cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detection {
    /// Whether the majority of the runs measured outcome `0`.
    pub found: bool,
    /// The walk steps of all runs.
    pub queries: usize,
}

/// The result of a search: a solution, if found, and what it cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Search<N> {
    /// A verified solution, or `None`.
    pub solution: Option<N>,
    /// The walk steps of all detections plus the classical checks.
    pub queries: usize,
}

/// Returns the number `K` of phase-estimation outcomes for `nodes` nodes and depth `depth`:
/// the next power of two of `2·√(T·max(n, 1))`.
///
/// Over every solution placement on the small trees of the tests, and on complete binary
/// trees without a solution up to depth 10, the constant 2 keeps the acceptance without a
/// solution at most `0.23`, below the `1/4` the analysis needs, while the constant 1 lets
/// it reach `0.60`.
#[must_use]
pub fn phase_outcomes(nodes: usize, depth: usize) -> usize {
    let scale = libm::ceil(2.0 * libm::sqrt(ratio(nodes * depth.max(1))));

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the node count is at most 2^24, so the value is positive and far below 2^32"
    )]
    let scale = scale as usize;

    scale.next_power_of_two()
}

/// Returns the probability that one detection run on `tree` measures outcome `0`.
///
/// It is at least `1/2` when the tree has a solution, and at most about `1/4` otherwise
/// (see [`phase_outcomes`]).
///
/// # Errors
///
/// Returns [`TooManyQubits`] when the tree needs more than [`MAX_QUBITS`] qubits for its
/// nodes.
pub fn acceptance_probability<T: SearchTree>(tree: &T) -> Result<f64, TooManyQubits> {
    Ok(Walk::new(tree, tree.root())?.acceptance())
}

/// Runs `runs` detection runs on `tree` and reports whether more than `3/8` of them saw a
/// solution.
///
/// The threshold lies between the acceptance probabilities with and without a solution, so
/// the error falls exponentially in `runs`.
///
/// # Errors
///
/// As [`acceptance_probability`].
pub fn detect<T: SearchTree, R: Rng + ?Sized>(
    tree: &T,
    runs: usize,
    rng: &mut R,
) -> Result<Detection, TooManyQubits> {
    detect_below(tree, tree.root(), runs, rng)
}

/// Finds a solution of `tree` by detection: from the root, descend into the first child
/// whose subtree detects a solution, until the node is a solution.
///
/// Each node on the way costs one classical check, and each tested child one [`detect`]
/// with `runs` runs. The result is `None` when no child detects a solution; the error is
/// one-sided, so a returned node is always a solution.
///
/// # Errors
///
/// As [`acceptance_probability`].
pub fn find<T: SearchTree, R: Rng + ?Sized>(
    tree: &T,
    runs: usize,
    rng: &mut R,
) -> Result<Search<T::Node>, TooManyQubits> {
    let mut node = tree.root();
    let mut queries = 0;

    loop {
        queries += 1;

        if tree.is_solution(&node) {
            return Ok(Search {
                solution: Some(node),
                queries,
            });
        }

        let mut next = None;

        for child in tree.children(&node) {
            let detection = detect_below(tree, child.clone(), runs, rng)?;
            queries += detection.queries;

            if detection.found {
                next = Some(child);
                break;
            }
        }

        let Some(child) = next else {
            return Ok(Search {
                solution: None,
                queries,
            });
        };

        node = child;
    }
}

/// Runs depth-first backtracking on `tree` and returns the first solution with the
/// predicate queries spent: one per visited node.
pub fn classical_backtrack<T: SearchTree>(tree: &T) -> Search<T::Node> {
    let mut stack = vec![tree.root()];
    let mut queries = 0;

    while let Some(node) = stack.pop() {
        queries += 1;

        if tree.is_solution(&node) {
            return Search {
                solution: Some(node),
                queries,
            };
        }

        let start = stack.len();
        stack.extend(tree.children(&node));
        stack[start..].reverse();
    }

    Search {
        solution: None,
        queries,
    }
}

/// Runs `runs` detection runs on the subtree below `root`.
///
/// # Errors
///
/// As [`acceptance_probability`].
fn detect_below<T: SearchTree, R: Rng + ?Sized>(
    tree: &T,
    root: T::Node,
    runs: usize,
    rng: &mut R,
) -> Result<Detection, TooManyQubits> {
    let walk = Walk::new(tree, root)?;
    let acceptance = walk.acceptance();
    let accepted = (0..runs).filter(|_| unit_f64(rng) < acceptance).count();

    Ok(Detection {
        found: 8 * accepted > 3 * runs,
        queries: runs * (walk.outcomes() - 1),
    })
}

/// The walk of one (sub)tree: the listed nodes and their stars.
struct Walk {
    /// The parent of each node; the root, node `0`, has none.
    parents: Vec<Option<usize>>,
    /// The depth of each node below the root.
    depths: Vec<usize>,
    /// The children of each unsolved node; empty for solutions.
    children: Vec<Vec<usize>>,
    /// Whether each node is a solution.
    solutions: Vec<bool>,
    /// The greatest depth, the bound `n`.
    depth: usize,
}

impl Walk {
    /// Lists the subtree of `tree` below `root` in breadth-first order.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the nodes do not fit in [`MAX_QUBITS`] qubits.
    fn new<T: SearchTree>(tree: &T, root: T::Node) -> Result<Self, TooManyQubits> {
        let mut nodes = vec![root];
        let mut walk = Self {
            parents: vec![None],
            depths: vec![0],
            children: Vec::new(),
            solutions: Vec::new(),
            depth: 0,
        };

        let mut next = 0;

        while next < nodes.len() {
            let solution = tree.is_solution(&nodes[next]);
            let children = if solution {
                Vec::new()
            } else {
                walk.add_children(tree, &mut nodes, next)?
            };

            walk.solutions.push(solution);
            walk.children.push(children);
            next += 1;
        }

        Ok(walk)
    }

    /// Appends the children of node `parent` to `nodes` and returns their indices.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the node count passes `2^MAX_QUBITS`.
    fn add_children<T: SearchTree>(
        &mut self,
        tree: &T,
        nodes: &mut Vec<T::Node>,
        parent: usize,
    ) -> Result<Vec<usize>, TooManyQubits> {
        let mut indices = Vec::new();
        let depth = self.depths[parent] + 1;

        let children: Vec<T::Node> = tree.children(&nodes[parent]).into_iter().collect();

        for child in children {
            if nodes.len() == 1 << MAX_QUBITS {
                return Err(TooManyQubits {
                    requested: MAX_QUBITS + 1,
                });
            }

            indices.push(nodes.len());
            nodes.push(child);
            self.parents.push(Some(parent));
            self.depths.push(depth);
            self.depth = self.depth.max(depth);
        }

        Ok(indices)
    }

    /// Returns the number of phase-estimation outcomes for this tree.
    fn outcomes(&self) -> usize {
        phase_outcomes(self.parents.len(), self.depth)
    }

    /// Returns the probability of outcome `0` of phase estimation from the root.
    fn acceptance(&self) -> f64 {
        let mut root = vec![Complex::ZERO; self.parents.len()];
        root[0] = Complex::ONE;

        zero_outcome_probability(&root, self.outcomes(), |state| self.step(state))
    }

    /// Applies the walk step `R_B·R_A` to `state`.
    fn step(&self, state: &mut [Complex]) {
        self.reflect(state, 0);
        self.reflect(state, 1);
    }

    /// Applies `D_x` for every node `x` whose depth has the parity `parity`.
    fn reflect(&self, state: &mut [Complex], parity: usize) {
        for node in (0..self.parents.len()).filter(|&node| self.depths[node] % 2 == parity) {
            if !self.solutions[node] {
                self.reflect_star(state, node);
            }
        }
    }

    /// Applies `D_x = I − 2|ψ_x⟩⟨ψ_x|` for the star of `node`.
    fn reflect_star(&self, state: &mut [Complex], node: usize) {
        let weight = if self.parents[node].is_none() {
            libm::sqrt(ratio(self.depth))
        } else {
            1.0
        };
        let children = &self.children[node];
        let norm = 1.0 + weight * weight * ratio(children.len());

        let overlap = children
            .iter()
            .fold(state[node], |sum, &child| sum + state[child].scale(weight));
        let factor = overlap.scale(2.0 / norm);

        state[node] = state[node] - factor;

        for &child in children {
            state[child] = state[child] - factor.scale(weight);
        }
    }
}

/// Converts a count to `f64`.
fn ratio(count: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "node counts are at most 2^24, exact in f64"
    )]
    let value = count as f64;

    value
}

/// Unit tests for the walk operator.
#[cfg(test)]
mod tests {
    use super::{SearchTree, Walk};
    use crate::Complex;
    use alloc::vec::Vec;

    /// A complete binary tree of depth 2 over node numbers in heap order, solution at `goal`.
    struct Heap {
        /// The solution node, if any.
        goal: Option<usize>,
    }

    impl SearchTree for Heap {
        type Node = usize;

        fn root(&self) -> usize {
            0
        }

        fn children(&self, node: &usize) -> impl IntoIterator<Item = usize> {
            if *node < 3 {
                Vec::from([2 * node + 1, 2 * node + 2])
            } else {
                Vec::new()
            }
        }

        fn is_solution(&self, node: &usize) -> bool {
            self.goal == Some(*node)
        }
    }

    /// Each reflection squares to the identity and the walk step keeps the norm.
    #[test]
    fn reflections_are_involutions() {
        let walk = Walk::new(&Heap { goal: Some(4) }, 0).unwrap();
        let start: Vec<Complex> = (0..7)
            .map(|i| Complex::new(f64::from(i) - 2.5, 0.5 * f64::from(i)))
            .collect();

        for parity in [0, 1] {
            let mut state = start.clone();
            walk.reflect(&mut state, parity);
            walk.reflect(&mut state, parity);

            assert!(
                state
                    .iter()
                    .zip(&start)
                    .all(|(a, b)| (*a - *b).norm_sqr() < 1e-20)
            );
        }

        let mut state = start.clone();
        walk.step(&mut state);
        let norm = |v: &[Complex]| v.iter().map(|a| a.norm_sqr()).sum::<f64>();

        assert!((norm(&state) - norm(&start)).abs() < 1e-9);
    }
}
