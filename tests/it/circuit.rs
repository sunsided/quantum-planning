//! Recorded Grover circuits: round trip on the simulator, OpenQASM 3 golden files and the
//! mapping from hardware counts back to candidate indices.

use quantum_planning::circuit::{InvalidBitstring, bitstring_index, candidate_counts};
use quantum_planning::grover::{self, amplify, optimal_iterations};
use quantum_planning::{Circuit, Complex, PhaseOracle, StateVector};

use crate::Spread;

/// Returns whether two states agree amplitude by amplitude within `1e-9`.
fn close(a: &[Complex], b: &[Complex]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (*x - *y).norm_sqr() < 1e-18)
}

/// Simulating the recorded circuit gives the state of direct application within `1e-9`,
/// for `n ≤ 6`, several marked counts and iteration counts.
#[test]
fn recorded_grover_matches_direct_application() {
    for qubits in 1..=6 {
        let states = 1 << qubits;

        for count in [1, 2, 3].into_iter().filter(|&count| count < states) {
            let spread = Spread { states, count };
            let mut oracle = PhaseOracle::new(&spread, Some);
            let marked: Vec<usize> = oracle.marked(states).collect();
            assert_eq!(marked.len(), count);

            for iterations in 0..=optimal_iterations(states, count) + 1 {
                let mut direct = StateVector::new(qubits).unwrap();
                amplify(&mut direct, &mut oracle, iterations);

                let circuit = grover::circuit(qubits, &marked, iterations).unwrap();
                let mut recorded = StateVector::new(qubits).unwrap();
                circuit.apply(&mut recorded);

                assert!(
                    close(direct.amplitudes(), recorded.amplitudes()),
                    "n={qubits} M={count} k={iterations}"
                );
            }
        }
    }
}

/// The lowered circuit, ancillas included, gives the same state on the ancilla-zero
/// subspace and nothing elsewhere.
#[test]
fn lowered_grover_matches_on_the_ancilla_zero_subspace() {
    for qubits in 3..=6 {
        let circuit = grover::circuit(qubits, &[1, 5], 1).unwrap();
        let lowered = circuit.decompose().unwrap();
        assert_eq!(lowered.ancillas(), qubits - 3);

        let mut direct = StateVector::new(qubits).unwrap();
        circuit.apply(&mut direct);
        let mut wide = StateVector::new(lowered.qubits()).unwrap();
        lowered.apply(&mut wide);

        let (data, rest) = wide.amplitudes().split_at(direct.states());
        assert!(close(data, direct.amplitudes()), "n={qubits}");
        assert!(rest.iter().all(|amplitude| amplitude.norm_sqr() < 1e-18));
    }
}

/// The lowered circuits report the cost that decides what runs on noisy hardware.
#[test]
fn reports_depth_and_two_qubit_gates() {
    let costs: Vec<(usize, usize, usize)> = (2..=5)
        .map(|qubits| {
            let marked = [0];
            let iterations = optimal_iterations(1 << qubits, 1);
            let lowered = grover::circuit(qubits, &marked, iterations)
                .unwrap()
                .decompose()
                .unwrap();

            (
                lowered.ancillas(),
                lowered.two_qubit_gates(),
                lowered.depth(),
            )
        })
        .collect();

    // n=2: one cz per oracle and diffusion. n=3: one Toffoli (6 CX) each, twice per iteration.
    assert_eq!((costs[0].0, costs[0].1), (0, 2));
    assert_eq!(costs[1].1, 2 * 2 * 6);
    assert_eq!(costs[2].0, 1);
    assert!(
        costs.windows(2).all(|pair| pair[0].2 < pair[1].2),
        "{costs:?}"
    );
}

/// Two qubits, index 3 marked, one iteration: the emitted program is the golden file.
#[test]
fn emits_golden_qasm_for_two_qubits() {
    let qasm = grover::circuit(2, &[3], 1).unwrap().qasm().unwrap();

    assert_eq!(qasm, include_str!("golden/grover_2q.qasm"));
}

/// Three qubits, index 5 marked, two iterations: the Toffoli lowering is in the program.
#[test]
fn emits_golden_qasm_for_three_qubits() {
    let qasm = grover::circuit(3, &[5], 2).unwrap().qasm().unwrap();

    assert_eq!(qasm, include_str!("golden/grover_3q.qasm"));
}

/// Four qubits need one ancilla; the program declares it and measures only the data qubits.
#[test]
fn emits_golden_qasm_for_four_qubits() {
    let qasm = grover::circuit(4, &[9], 1).unwrap().qasm().unwrap();

    assert_eq!(qasm, include_str!("golden/grover_4q.qasm"));
}

/// Formats `index` over `qubits` bits with the highest qubit first, as hardware prints it.
fn hardware_bitstring(index: usize, qubits: usize) -> String {
    format!("{index:0qubits$b}")
}

/// `X` on qubit `q` measures as index `2^q`, which hardware prints with the `1` at position
/// `n − 1 − q` from the left; the mapping reads it back as `2^q`.
#[test]
fn maps_counts_back_in_qubit_order() {
    let qubits = 4;

    for qubit in 0..qubits {
        let mut circuit = Circuit::new(qubits).unwrap();
        circuit.x(qubit);
        circuit.measure_all();

        let mut state = StateVector::new(qubits).unwrap();
        circuit.apply(&mut state);
        let index = state.sample(&mut quantum_planning::SplitMix64::new(1));
        let bitstring = hardware_bitstring(index, qubits);

        assert_eq!(index, 1 << qubit);
        assert_eq!(bitstring.find('1'), Some(qubits - 1 - qubit));
        assert_eq!(bitstring_index(&bitstring, qubits), Ok(1 << qubit));
    }

    let counts = [("1001", 700), ("0000", 20), ("1001", 4), ("0110", 300)];
    assert_eq!(
        candidate_counts(counts, 4),
        Ok(vec![(0, 20), (6, 300), (9, 704)])
    );
}

/// Bitstrings of the wrong length or with other characters are errors.
#[test]
fn rejects_malformed_bitstrings() {
    assert_eq!(
        bitstring_index("101", 4),
        Err(InvalidBitstring::Length {
            expected: 4,
            found: 3
        })
    );
    assert_eq!(
        bitstring_index("1 01", 4),
        Err(InvalidBitstring::Character {
            position: 1,
            found: ' '
        })
    );
    assert!(candidate_counts([("2", 1)], 1).is_err());
}

/// A register above the simulator limit is an error for recording and for lowering.
#[test]
fn rejects_circuits_above_the_limit() {
    assert!(Circuit::new(quantum_planning::MAX_QUBITS + 1).is_err());

    let mut wide = grover::circuit(14, &[0], 0).unwrap();
    wide.mcz(0..14);

    assert!(wide.decompose().is_err(), "14 data qubits need 11 ancillas");
}

/// A repeated marked index counts once, also without iterations; an index past the
/// register panics before any gate is recorded.
#[test]
fn treats_marked_indices_as_a_set() {
    assert_eq!(
        grover::circuit(3, &[5, 5, 2], 1).unwrap(),
        grover::circuit(3, &[2, 5], 1).unwrap()
    );

    let result = std::panic::catch_unwind(|| grover::circuit(3, &[8], 0));
    assert!(result.is_err(), "index 8 needs four qubits");
}
