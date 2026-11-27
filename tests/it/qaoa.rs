//! QAOA on a QUBO-encoded assignment problem: the circuit against the simulator, the
//! optimized angles and the sampled plan against exhaustive search.

use planning_qubo::{Qubo, assignment};
use quantum_planning::qaoa::Qaoa;
use quantum_planning::{Complex, SplitMix64, StateVector};

/// Two tasks, two agents, one task per agent: the cheapest plan gives task 0 to agent 1
/// and task 1 to agent 0, at cost 3.
fn assignment_qaoa() -> (Qaoa, assignment::Encoding<i64>) {
    let costs = [vec![4, 1], vec![2, 3]];
    let encoding = assignment::encode(&costs, &[1, 1], 11).unwrap();

    (Qaoa::new(encoding.problem().qubo()).unwrap(), encoding)
}

/// Returns `|⟨a|b⟩|`, which is 1 when the states agree up to a global phase.
fn overlap(a: &[Complex], b: &[Complex]) -> f64 {
    let inner = a.iter().zip(b).fold(Complex::ZERO, |sum, (x, y)| {
        sum + Complex::new(x.re, -x.im) * *y
    });

    inner.norm_sqr().sqrt()
}

/// The energies are those of the QUBO model, bit `q` being variable `q`.
#[test]
fn energies_match_the_model() {
    let (qaoa, encoding) = assignment_qaoa();
    let qubo = encoding.problem().qubo();

    for index in 0..1 << qaoa.qubits() {
        let bits: Vec<bool> = (0..qaoa.qubits()).map(|q| index >> q & 1 == 1).collect();

        #[expect(clippy::cast_precision_loss, reason = "small test energies")]
        let expected = qubo.energy(&bits) as f64;

        assert!((qaoa.energy(index) - expected).abs() < 1e-9);
    }
}

/// The recorded `rz`/`cx`/`rx` circuit prepares the simulated state up to a global phase,
/// on a model with fields, couplings and an offset.
#[test]
fn circuit_matches_the_simulated_state() {
    // Variable 3 has no terms, so its field is zero and the circuit skips its rotation.
    let mut qubo = Qubo::<f64>::new(4);
    qubo.add_offset(0.7);
    qubo.add_linear(0, -1.5);
    qubo.add_linear(2, 0.4);
    qubo.add_quadratic(0, 1, 2.0);
    qubo.add_quadratic(1, 2, -0.9);
    let qaoa = Qaoa::new(&qubo).unwrap();

    let (gammas, betas) = ([0.31, 1.2], [0.7, 0.15]);
    let mut direct = StateVector::new(4).unwrap();
    qaoa.prepare(&gammas, &betas, &mut direct);

    let circuit = qaoa.circuit(&gammas, &betas);
    assert!(!circuit.qasm().unwrap().contains("rz(0) q[3]"));
    assert!(
        !circuit
            .gates()
            .iter()
            .any(|gate| matches!(gate, quantum_planning::Gate::Rz { qubit: 3, .. }))
    );
    let mut recorded = StateVector::new(4).unwrap();
    circuit.apply(&mut recorded);

    assert!((overlap(direct.amplitudes(), recorded.amplitudes()) - 1.0).abs() < 1e-9);

    let qasm = circuit.qasm().unwrap();
    assert!(qasm.contains("rz(") && qasm.contains("rx(") && qasm.contains("cx q[0], q[1];"));
}

/// Optimized angles lower the expected energy below the even superposition, more layers
/// lower it further, and sampling finds the exhaustive optimum.
#[test]
fn finds_the_cheapest_assignment() {
    let (qaoa, encoding) = assignment_qaoa();
    let mut rng = SplitMix64::new(11);

    let uniform = qaoa.expectation(&[0.0], &[0.0]);
    let one = qaoa.optimize(1, 3, &mut rng);
    let two = qaoa.optimize(2, 3, &mut rng);

    assert!(
        one.expectation < uniform,
        "{} vs {uniform}",
        one.expectation
    );
    assert!(two.expectation <= one.expectation + 1e-9);

    // The reported expectation belongs to the returned angles.
    for parameters in [&one, &two] {
        let recomputed = qaoa.expectation(&parameters.gammas, &parameters.betas);
        assert!((recomputed - parameters.expectation).abs() < 1e-12);
    }
    assert!(two.evaluations > 0);

    let exhaustive = qaoa.exhaustive_minimum();
    let sampled = qaoa.sample_best(&two, 32, &mut rng);
    assert_eq!(exhaustive.queries, 16);
    assert_eq!(sampled.index, exhaustive.index);

    let bits: Vec<bool> = (0..4).map(|q| sampled.index >> q & 1 == 1).collect();
    let decoded = encoding.decode(&bits);
    assert_eq!(decoded.agents, [Some(1), Some(0)]);
    assert_eq!(decoded.cost, 3);
}

/// A model with more variables than the simulator holds is an error.
#[test]
fn rejects_models_above_the_limit() {
    let qubo = Qubo::<i64>::new(quantum_planning::MAX_QUBITS + 1);

    assert!(Qaoa::new(&qubo).is_err());
}
