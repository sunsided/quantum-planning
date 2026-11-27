//! Lowering of multi-controlled Z gates to `stdgates.inc`: Toffoli V-chains, then `CX` and `T`.

use alloc::vec::Vec;
use core::f64::consts::FRAC_PI_4;

use crate::TooManyQubits;

use super::{Circuit, Gate};

/// Returns `circuit` with every `MCZ` on three or more qubits lowered; see
/// [`Circuit::decompose`].
///
/// # Errors
///
/// Returns [`TooManyQubits`] when the ancillas do not fit next to the qubits.
pub(super) fn decompose(circuit: &Circuit) -> Result<Circuit, TooManyQubits> {
    let ancillas = circuit
        .gates
        .iter()
        .map(|gate| match gate {
            Gate::Mcz(mask) => (mask.count_ones() as usize).saturating_sub(3),
            _ => 0,
        })
        .max()
        .unwrap_or(0)
        .max(circuit.ancillas);
    let mut lowered = Circuit::with_ancillas(circuit.data_qubits(), ancillas)?;

    // The existing ancillas keep their indices; the new ones go above the data qubits too.
    let first_ancilla = circuit.data_qubits();

    for gate in &circuit.gates {
        match *gate {
            Gate::Mcz(mask) if mask.count_ones() >= 3 => {
                lower_mcz(&mut lowered, mask, first_ancilla);
            }
            gate => lowered.push(gate),
        }
    }

    Ok(lowered)
}

/// Appends `MCZ` on the qubits of `mask` (at least three) as `H`, a multi-controlled X and
/// `H` on the highest qubit, with ancillas from `first_ancilla` upward.
fn lower_mcz(circuit: &mut Circuit, mask: usize, first_ancilla: usize) {
    let qubits: Vec<usize> = (0..usize::BITS as usize)
        .filter(|qubit| mask >> qubit & 1 == 1)
        .collect();
    let (&target, controls) = qubits
        .split_last()
        .expect("the mask has three or more bits");

    circuit.h(target);
    mcx(circuit, controls, target, first_ancilla);
    circuit.h(target);
}

/// Appends `X` on `target` controlled by all `controls` (at least two), as a V-chain of
/// Toffoli gates over `controls.len() − 2` ancillas that end in `|0⟩` again.
///
/// The chain computes `a₀ = c₀ ∧ c₁` and `aᵢ = cᵢ₊₁ ∧ aᵢ₋₁`, flips the target with the last
/// control and the last ancilla, and uncomputes the ancillas in reverse.
fn mcx(circuit: &mut Circuit, controls: &[usize], target: usize, first_ancilla: usize) {
    if let [a, b] = *controls {
        ccx(circuit, a, b, target);
        return;
    }

    let ancilla = |index: usize| first_ancilla + index;
    let chain: Vec<(usize, usize, usize)> =
        core::iter::once((controls[0], controls[1], ancilla(0)))
            .chain((2..controls.len() - 1).map(|i| (controls[i], ancilla(i - 2), ancilla(i - 1))))
            .collect();

    for &(a, b, c) in &chain {
        ccx(circuit, a, b, c);
    }

    ccx(
        circuit,
        controls[controls.len() - 1],
        ancilla(controls.len() - 3),
        target,
    );

    for &(a, b, c) in chain.iter().rev() {
        ccx(circuit, a, b, c);
    }
}

/// Appends the Toffoli gate (`X` on `c` where `a` and `b` are `1`) as six `CX` gates with
/// `H`, `T` and `T†`, exactly and without a global phase (Nielsen and Chuang, figure 4.9).
fn ccx(circuit: &mut Circuit, a: usize, b: usize, c: usize) {
    circuit.h(c);
    circuit.cx(b, c);
    circuit.phase(c, -FRAC_PI_4);
    circuit.cx(a, c);
    circuit.phase(c, FRAC_PI_4);
    circuit.cx(b, c);
    circuit.phase(c, -FRAC_PI_4);
    circuit.cx(a, c);
    circuit.phase(b, FRAC_PI_4);
    circuit.phase(c, FRAC_PI_4);
    circuit.h(c);
    circuit.cx(a, b);
    circuit.phase(a, FRAC_PI_4);
    circuit.phase(b, -FRAC_PI_4);
    circuit.cx(a, b);
}

/// Unit tests for the lowering, checked on the simulator.
#[cfg(test)]
mod tests {
    use crate::{Circuit, Complex, StateVector};

    /// Prepares an uneven superposition with distinct phases on the data qubits.
    fn prepare(circuit: &mut Circuit, data: usize) {
        for qubit in 0..data {
            circuit.h(qubit);
            circuit.phase(qubit, 0.3 + 0.7 * f64::from(u32::try_from(qubit).unwrap()));
            circuit.rx(qubit, 0.2);
        }
    }

    /// For `k = 3, 4, 5, 6` qubits, the lowered `MCZ` acts as `MCZ` on the ancilla-zero
    /// subspace and leaves no amplitude where an ancilla is `1`.
    #[test]
    fn lowered_mcz_matches_mcz() {
        for data in 3..=6 {
            let mut circuit = Circuit::new(data).unwrap();
            prepare(&mut circuit, data);
            circuit.mcz(0..data);

            let lowered = circuit.decompose().unwrap();
            assert_eq!(lowered.ancillas(), data - 3);
            assert!(
                lowered
                    .gates()
                    .iter()
                    .all(|gate| gate.mask().count_ones() <= 2)
            );

            let mut direct = StateVector::new(data).unwrap();
            circuit.apply(&mut direct);
            let mut wide = StateVector::new(lowered.qubits()).unwrap();
            lowered.apply(&mut wide);

            for (index, amplitude) in wide.amplitudes().iter().enumerate() {
                let expected = direct
                    .amplitudes()
                    .get(index)
                    .copied()
                    .unwrap_or(Complex::ZERO);

                assert!(
                    (*amplitude - expected).norm_sqr() < 1e-20,
                    "k={data} index {index}: {amplitude:?} != {expected:?}"
                );
            }
        }
    }

    /// One Toffoli costs six `CX`; a 5-qubit `MCZ` takes five Toffolis on two ancillas.
    #[test]
    fn counts_the_lowered_gates() {
        let mut circuit = Circuit::new(5).unwrap();
        circuit.mcz(0..5);

        let lowered = circuit.decompose().unwrap();

        assert_eq!(lowered.ancillas(), 2);
        assert_eq!(lowered.two_qubit_gates(), 5 * 6);
    }
}
