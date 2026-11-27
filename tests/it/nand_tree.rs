//! NAND-tree evaluation: every input of small trees, and the cost against classical
//! evaluation.

use quantum_planning::SplitMix64;
use quantum_planning::nand_tree::{NandTree, acceptance_probability, classical_evaluate, evaluate};
use rand_core::Rng;

/// Returns the inputs of `tree` given by the bits of `mask`.
fn inputs(tree: &NandTree, mask: u64) -> Vec<bool> {
    (0..tree.leaves())
        .map(|leaf| mask >> leaf & 1 == 1)
        .collect()
}

/// Returns the lowest acceptance over `true` roots and the highest over `false` roots.
fn margins(tree: &NandTree, masks: impl Iterator<Item = u64>) -> (f64, f64) {
    let mut yes = f64::INFINITY;
    let mut no = f64::NEG_INFINITY;

    for mask in masks {
        let inputs = inputs(tree, mask);
        let acceptance = acceptance_probability(tree, &inputs);

        if classical_evaluate(tree, &inputs).value {
            yes = yes.min(acceptance);
        } else {
            no = no.max(acceptance);
        }
    }

    (yes, no)
}

/// Returns every input mask of `tree` up to 512 masks, otherwise 256 seeded random ones.
fn sample_masks(tree: &NandTree) -> Vec<u64> {
    let leaves = tree.leaves();

    if leaves <= 9 {
        return (0..1 << leaves).collect();
    }

    let mut rng = SplitMix64::new(17);
    let keep = if leaves >= 64 {
        u64::MAX
    } else {
        (1 << leaves) - 1
    };

    (0..256).map(|_| rng.next_u64() & keep).collect()
}

/// Over every input of the binary trees up to depth 3 and the ternary trees up to depth 2,
/// and over 256 random inputs of the binary tree of depth 4, `true` roots accept with
/// probability at least 0.36 and `false` roots at most 0.05.
#[test]
fn separates_true_and_false_roots() {
    for (fan_out, depth) in [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (2, 4)] {
        let tree = NandTree::new(fan_out, depth).unwrap();
        let (yes, no) = margins(&tree, sample_masks(&tree).into_iter());

        assert!(
            yes >= 0.36 && no <= 0.05,
            "d={fan_out} n={depth}: {yes} vs {no}"
        );
    }
}

/// Past the tuned sizes, at depths 5 and 6, random inputs and the inputs that make
/// left-to-right evaluation read every leaf stay on the right side of the threshold 1/5.
#[test]
fn keeps_the_margin_on_larger_trees() {
    for depth in [5_usize, 6] {
        let tree = NandTree::new(2, depth).unwrap();
        let mut rng = SplitMix64::new(23);
        let mut cases: Vec<Vec<bool>> = (0..12)
            .map(|_| {
                (0..tree.leaves())
                    .map(|_| rng.next_u64() & 1 == 1)
                    .collect()
            })
            .collect();
        cases.push(worst_inputs(2, depth, true));
        cases.push(worst_inputs(2, depth, false));

        for inputs in &cases {
            let value = classical_evaluate(&tree, inputs).value;
            let acceptance = acceptance_probability(&tree, inputs);
            assert_eq!(acceptance > 0.2, value, "n={depth}: {acceptance}");
        }
    }
}

/// With a few runs, the evaluation matches the classical value on every input of depth 2.
#[test]
fn evaluates_like_the_classical_tree() {
    let tree = NandTree::new(2, 2).unwrap();
    let mut rng = SplitMix64::new(4);

    for mask in 0..16 {
        let inputs = inputs(&tree, mask);

        assert_eq!(
            evaluate(&tree, &inputs, 25, &mut rng).value,
            classical_evaluate(&tree, &inputs).value,
            "inputs {mask:04b}"
        );
    }
}

/// Returns inputs on which left-to-right evaluation reads every leaf, for a node of
/// `depth` that must take `value`: a `false` node has only `true` children, a `true` node
/// has its single `false` child last.
fn worst_inputs(fan_out: usize, depth: usize, value: bool) -> Vec<bool> {
    if depth == 0 {
        return vec![value];
    }

    (0..fan_out)
        .flat_map(|child| {
            let child_value = !value || child + 1 < fan_out;

            worst_inputs(fan_out, depth - 1, child_value)
        })
        .collect()
}

/// The evolution time grows as `√N`, while the worst-case classical reads grow as `N`.
#[test]
fn time_grows_as_the_square_root_of_the_leaves() {
    for depth in [2, 4, 6] {
        let tree = NandTree::new(2, depth).unwrap();
        let leaves = tree.leaves();

        let worst = worst_inputs(2, depth, true);
        let classical = classical_evaluate(&tree, &worst);
        let quantum = evaluate(&tree, &worst, 1, &mut SplitMix64::new(2));
        let per_root = quantum.time / crate::to_f64(leaves).sqrt();

        assert_eq!((classical.value, classical.queries), (true, leaves));
        assert!((16.0..36.0).contains(&per_root), "n={depth}: {per_root}");
    }
}

/// Trees with fewer than two children per node or past the simulator are rejected.
#[test]
fn rejects_trees_above_the_limit() {
    assert!(NandTree::new(2, 23).is_err());
    assert!(NandTree::new(2, 200).is_err());
    assert_eq!(NandTree::new(3, 2).unwrap().nodes(), 13);
}
