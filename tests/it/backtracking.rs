//! Quantum backtracking: detection over every solution placement on small trees, finding a
//! solution, and the query counts against classical backtracking.

use quantum_planning::SplitMix64;
use quantum_planning::backtracking::{
    SearchTree, acceptance_probability, classical_backtrack, detect, find,
};

/// A tree given by the children of each node, with the solutions in a bit mask.
struct Shape {
    /// The children of each node; node 0 is the root.
    children: Vec<Vec<usize>>,
    /// The solution nodes.
    solutions: Vec<usize>,
}

impl Shape {
    /// A complete tree with `fan_out` children per inner node and depth `depth`.
    fn complete(fan_out: usize, depth: u32) -> Self {
        let inner = (0..depth).map(|level| fan_out.pow(level)).sum::<usize>();
        let nodes = inner + fan_out.pow(depth);
        let children = (0..nodes)
            .map(|node| {
                if node < inner {
                    (1..=fan_out).map(|i| fan_out * node + i).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        Self {
            children,
            solutions: Vec::new(),
        }
    }

    /// An unbalanced tree: a path of four nodes with a fan of three at the end, and a
    /// sibling leaf next to every path node.
    fn unbalanced() -> Self {
        let children = vec![
            vec![1, 2],
            vec![3, 4],
            vec![],
            vec![5, 6],
            vec![],
            vec![7, 8, 9],
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        Self {
            children,
            solutions: Vec::new(),
        }
    }

    /// Returns the node count.
    fn nodes(&self) -> usize {
        self.children.len()
    }

    /// Returns the same shape with the solutions whose bits are set in `mask`.
    fn with(&self, mask: u64) -> Self {
        self.with_solutions(
            (0..self.nodes())
                .filter(|node| mask >> node & 1 == 1)
                .collect(),
        )
    }

    /// Returns the same shape with `solutions` as the solution nodes.
    fn with_solutions(&self, solutions: Vec<usize>) -> Self {
        Self {
            children: self.children.clone(),
            solutions,
        }
    }
}

impl SearchTree for Shape {
    type Node = usize;

    fn root(&self) -> usize {
        0
    }

    fn children(&self, node: &usize) -> impl IntoIterator<Item = usize> {
        self.children[*node].clone()
    }

    fn is_solution(&self, node: &usize) -> bool {
        self.solutions.contains(node)
    }
}

/// Returns the solution masks to test on `shape`: every subset for small trees, otherwise
/// none, each single node and each pair with the last leaf.
fn masks(shape: &Shape) -> Vec<u64> {
    let nodes = shape.nodes();

    if nodes <= 10 {
        return (0..1 << nodes).collect();
    }

    let last = 1 << (nodes - 1);
    let mut masks = vec![0];
    masks.extend((0..nodes).map(|node| 1 << node));
    masks.extend((0..nodes - 1).map(|node| 1 << node | last));

    masks
}

/// Returns the lowest acceptance with a solution and the highest without, over all masks.
fn margins(shape: &Shape) -> (f64, f64) {
    let mut yes = f64::INFINITY;
    let mut no = f64::NEG_INFINITY;

    for mask in masks(shape) {
        let acceptance = acceptance_probability(&shape.with(mask)).unwrap();

        // A solution below a solution is unreachable, so reachability decides the class.
        if classical_backtrack(&shape.with(mask)).solution.is_some() {
            yes = yes.min(acceptance);
        } else {
            no = no.max(acceptance);
        }
    }

    (yes, no)
}

/// Over every solution placement on small trees, a tree with a solution accepts with
/// probability at least 1/2, and a tree without one at most 1/4.
#[test]
fn separates_trees_with_and_without_solutions() {
    let shapes = [
        Shape::complete(2, 2),
        Shape::complete(3, 1),
        Shape::unbalanced(),
        Shape::complete(2, 3),
        Shape::complete(3, 2),
    ];

    for shape in &shapes {
        let (yes, no) = margins(shape);

        assert!(
            yes >= 0.5 && no <= 0.25,
            "T={}: {yes} vs {no}",
            shape.nodes()
        );
    }
}

/// Returns a complete binary tree of `depth` with its only solution at the last leaf, the
/// last node that depth-first backtracking visits.
fn worst_case(depth: u32) -> Shape {
    let shape = Shape::complete(2, depth);
    let last = shape.nodes() - 1;

    shape.with_solutions(vec![last])
}

/// Detection costs `O(√(T·n))` walk steps, and from depth 8 on fewer queries than the
/// classical backtracking that visits all `T` nodes.
#[test]
fn detection_beats_classical_backtracking() {
    for depth in [4, 6, 8, 10] {
        let tree = worst_case(depth);
        let classical = classical_backtrack(&tree);
        assert_eq!(classical.queries, tree.nodes());

        let detection = detect(&tree, 1, &mut SplitMix64::new(1)).unwrap();
        let scale = (crate::to_f64(tree.nodes()) * f64::from(depth)).sqrt();

        assert!(
            acceptance_probability(&tree).unwrap() >= 0.5,
            "depth {depth}"
        );
        assert!(
            crate::to_f64(detection.queries) <= 4.0 * scale,
            "depth {depth}"
        );

        if depth >= 8 {
            assert!(
                detection.queries < classical.queries,
                "depth {depth}: {} vs {}",
                detection.queries,
                classical.queries
            );
        }
    }
}

/// Past the tuned sizes, complete binary trees without a solution at depths 6, 8 and 10
/// still accept below the threshold 3/8.
#[test]
fn rejects_larger_trees_without_a_solution() {
    for depth in [6, 8, 10] {
        let acceptance = acceptance_probability(&Shape::complete(2, depth)).unwrap();
        assert!(acceptance < 0.375, "depth {depth}: {acceptance}");
    }
}

/// Detection with several runs decides correctly, and `find` returns the solution.
#[test]
fn detects_and_finds_the_solution() {
    let mut rng = SplitMix64::new(5);
    let tree = worst_case(4);

    assert!(detect(&tree, 15, &mut rng).unwrap().found);
    assert!(!detect(&tree.with(0), 15, &mut rng).unwrap().found);

    let search = find(&tree, 15, &mut rng).unwrap();
    assert_eq!(search.solution, Some(tree.nodes() - 1));
    assert!(search.queries > 0);
}

/// Without a solution, `find` returns `None` and never a node that is not a solution.
#[test]
fn find_is_one_sided() {
    let mut rng = SplitMix64::new(8);
    let tree = Shape::complete(3, 2);

    for _ in 0..20 {
        assert_eq!(find(&tree, 3, &mut rng).unwrap().solution, None);
    }

    let tree = tree.with(1 << 7);
    for _ in 0..20 {
        let solution = find(&tree, 3, &mut rng).unwrap().solution;
        assert!(solution.is_none_or(|node| node == 7));
    }
}
