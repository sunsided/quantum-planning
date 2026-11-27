//! Recorded circuits: a gate list that the simulator can run and that exports as OpenQASM 3.
//!
//! A [`Circuit`] records the gates of an algorithm instead of applying them to a
//! [`StateVector`]. [`Circuit::apply`] runs the recording on the simulator, so a recorded
//! circuit and direct application give the same state. [`Circuit::qasm`] renders the
//! circuit as an OpenQASM 3 program for gate-based hardware.
//!
//! ```mermaid
//! graph LR
//!     record["Circuit: H, X, Z, phase, RX, RZ, CX, MCZ, measure"] -->|apply| simulate["StateVector"]
//!     record -->|decompose| lowered["stdgates only, ancillas for MCZ"]
//!     lowered -->|qasm| text["OpenQASM 3 program"]
//!     text -->|"hardware shots"| counts["bitstring counts"]
//!     counts -->|candidate_counts| indices["candidate indices"]
//! ```
//!
//! # Qubit and bit order
//!
//! Qubit `q` of a circuit is `q[q]` in the program and is measured into `c[q]`. The order is
//! the little-endian order of the [simulator](crate::StateVector): the basis index has bit
//! `q` set when qubit `q` is `1`. Hardware results usually print the classical register
//! with the highest bit first, so the bitstring `"011"` of three qubits reads
//! `c[2] = 0, c[1] = 1, c[0] = 1`, which is index `0b011 = 3`. [`bitstring_index`] and
//! [`candidate_counts`] read bitstrings that way.
//!
//! # Lowering
//!
//! OpenQASM 3's `stdgates.inc` has no multi-controlled Z on more than two qubits.
//! [`Circuit::decompose`] replaces each such gate on `k ≥ 3` qubits with `H`, Toffoli, `H` on
//! a V-chain of `k − 3` clean ancillas, and each Toffoli with six `CX` and seven `T`/`T†`.
//! The ancillas are extra qubits above the data qubits; they start and end in `|0⟩`. The
//! [`depth`](Circuit::depth) and [`two_qubit_gates`](Circuit::two_qubit_gates) of the
//! lowered circuit estimate what runs on noisy hardware, where Grover search is limited to
//! roughly 3 to 5 qubits.

mod counts;
mod decompose;
mod qasm;

pub use counts::{InvalidBitstring, bitstring_index, candidate_counts};

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::{MAX_QUBITS, StateVector, TooManyQubits};

/// One gate of a [`Circuit`], on qubit indices of that circuit.
///
/// Angles are in radians and follow the definitions of OpenQASM's `stdgates.inc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate {
    /// The Hadamard gate on a qubit.
    H(usize),
    /// The Pauli-X (NOT) gate on a qubit.
    X(usize),
    /// The Pauli-Z gate on a qubit.
    Z(usize),
    /// The phase gate `diag(1, e^(i·angle))`, `p` in OpenQASM (`t` for `π/4`, `tdg` for `−π/4`).
    Phase {
        /// The qubit.
        qubit: usize,
        /// The phase of `|1⟩`.
        angle: f64,
    },
    /// The rotation `e^(−i·angle·X/2)`.
    Rx {
        /// The qubit.
        qubit: usize,
        /// The rotation angle.
        angle: f64,
    },
    /// The rotation `e^(−i·angle·Z/2)`.
    Rz {
        /// The qubit.
        qubit: usize,
        /// The rotation angle.
        angle: f64,
    },
    /// The controlled NOT gate.
    Cx {
        /// The qubit that must be `1`.
        control: usize,
        /// The qubit that flips.
        target: usize,
    },
    /// The multi-controlled Z gate on the qubits whose bits are set in the mask.
    ///
    /// An empty mask is the global phase `−1`.
    Mcz(usize),
    /// A measurement of a data qubit into the classical bit of the same index.
    Measure(usize),
}

impl Gate {
    /// Returns the mask of the qubits that the gate acts on.
    ///
    /// # Panics
    ///
    /// Panics when a qubit index is not below `usize::BITS`; a [`Circuit`] never holds
    /// such a gate.
    #[must_use]
    pub fn mask(&self) -> usize {
        match *self {
            Self::H(qubit)
            | Self::X(qubit)
            | Self::Z(qubit)
            | Self::Phase { qubit, .. }
            | Self::Rx { qubit, .. }
            | Self::Rz { qubit, .. }
            | Self::Measure(qubit) => 1 << qubit,
            Self::Cx { control, target } => (1 << control) | (1 << target),
            Self::Mcz(mask) => mask,
        }
    }

    /// Returns the highest qubit index of the gate, or `None` for an empty `MCZ`.
    fn highest_qubit(&self) -> Option<usize> {
        match *self {
            Self::H(qubit)
            | Self::X(qubit)
            | Self::Z(qubit)
            | Self::Phase { qubit, .. }
            | Self::Rx { qubit, .. }
            | Self::Rz { qubit, .. }
            | Self::Measure(qubit) => Some(qubit),
            Self::Cx { control, target } => Some(control.max(target)),
            Self::Mcz(mask) => mask
                .checked_ilog2()
                .map(|bit| usize::try_from(bit).expect("a bit index fits in usize")),
        }
    }
}

/// A recorded circuit: a register size and a list of gates.
///
/// The register holds the data qubits and, after [`decompose`](Self::decompose), the
/// ancillas above them. See the [module docs](self) for the qubit order.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    /// The number of qubits, ancillas included. At most [`MAX_QUBITS`].
    qubits: usize,
    /// The number of ancillas: the highest qubits, which start and end in `|0⟩`.
    ancillas: usize,
    /// The gates in application order.
    gates: Vec<Gate>,
}

impl Circuit {
    /// Makes an empty circuit on `qubits` data qubits.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when `qubits` is more than [`MAX_QUBITS`], so that every
    /// circuit can run on the simulator.
    pub fn new(qubits: usize) -> Result<Self, TooManyQubits> {
        Self::with_ancillas(qubits, 0)
    }

    /// Makes an empty circuit on `data` data qubits and `ancillas` ancillas above them.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the total is more than [`MAX_QUBITS`].
    fn with_ancillas(data: usize, ancillas: usize) -> Result<Self, TooManyQubits> {
        let qubits = data + ancillas;

        if qubits > MAX_QUBITS {
            return Err(TooManyQubits { requested: qubits });
        }

        Ok(Self {
            qubits,
            ancillas,
            gates: Vec::new(),
        })
    }

    /// Returns the number of qubits, ancillas included.
    #[must_use]
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Returns the number of ancillas, the highest qubits of the register.
    #[must_use]
    pub fn ancillas(&self) -> usize {
        self.ancillas
    }

    /// Returns the number of data qubits, the qubits below the ancillas.
    #[must_use]
    pub fn data_qubits(&self) -> usize {
        self.qubits - self.ancillas
    }

    /// Returns the gates in application order.
    #[must_use]
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Appends `gate`.
    ///
    /// # Panics
    ///
    /// Panics when the gate acts on a qubit outside the register, when a `Cx` has the same
    /// control and target, or when a `Measure` names an ancilla.
    pub fn push(&mut self, gate: Gate) {
        assert!(
            gate.highest_qubit().is_none_or(|qubit| qubit < self.qubits),
            "{gate:?} acts outside the {}-qubit register",
            self.qubits
        );

        if let Gate::Cx { control, target } = gate {
            assert_ne!(control, target, "cx needs two distinct qubits");
        }

        if let Gate::Measure(qubit) = gate {
            assert!(qubit < self.data_qubits(), "qubit {qubit} is an ancilla");
        }

        self.gates.push(gate);
    }

    /// Appends the Hadamard gate on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn h(&mut self, qubit: usize) {
        self.push(Gate::H(qubit));
    }

    /// Appends the Pauli-X gate on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn x(&mut self, qubit: usize) {
        self.push(Gate::X(qubit));
    }

    /// Appends the Pauli-Z gate on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn z(&mut self, qubit: usize) {
        self.push(Gate::Z(qubit));
    }

    /// Appends the phase gate `diag(1, e^(i·angle))` on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn phase(&mut self, qubit: usize, angle: f64) {
        self.push(Gate::Phase { qubit, angle });
    }

    /// Appends the rotation `e^(−i·angle·X/2)` on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn rx(&mut self, qubit: usize, angle: f64) {
        self.push(Gate::Rx { qubit, angle });
    }

    /// Appends the rotation `e^(−i·angle·Z/2)` on `qubit`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn rz(&mut self, qubit: usize, angle: f64) {
        self.push(Gate::Rz { qubit, angle });
    }

    /// Appends the controlled NOT gate.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push).
    pub fn cx(&mut self, control: usize, target: usize) {
        self.push(Gate::Cx { control, target });
    }

    /// Appends the multi-controlled Z gate on `qubits`.
    ///
    /// # Panics
    ///
    /// As [`push`](Self::push), and when a qubit does not fit in the mask.
    pub fn mcz(&mut self, qubits: impl IntoIterator<Item = usize>) {
        let mask = qubits.into_iter().fold(0, |mask, qubit| {
            assert!(
                qubit < MAX_QUBITS,
                "qubit {qubit} is outside every register"
            );
            mask | 1 << qubit
        });

        self.push(Gate::Mcz(mask));
    }

    /// Appends a measurement of every data qubit.
    pub fn measure_all(&mut self) {
        for qubit in 0..self.data_qubits() {
            self.push(Gate::Measure(qubit));
        }
    }

    /// Runs the gates on `state`. Measurements are skipped, so `state` ends as the state
    /// right before measurement.
    ///
    /// # Panics
    ///
    /// Panics when `state` has a different qubit count than the circuit.
    pub fn apply(&self, state: &mut StateVector) {
        assert_eq!(
            state.qubits(),
            self.qubits,
            "the state must have as many qubits as the circuit"
        );

        for gate in &self.gates {
            apply_gate(state, *gate);
        }
    }

    /// Returns the depth: the number of layers when each gate runs as early as its qubits
    /// allow. Measurements do not count.
    #[must_use]
    pub fn depth(&self) -> usize {
        let mut layer = vec![0; self.qubits];

        for gate in self
            .gates
            .iter()
            .filter(|gate| !matches!(gate, Gate::Measure(_)))
        {
            let qubits = || (0..self.qubits).filter(|qubit| gate.mask() >> qubit & 1 == 1);
            let next = qubits().map(|qubit| layer[qubit]).max().unwrap_or(0) + 1;

            for qubit in qubits() {
                layer[qubit] = next;
            }
        }

        layer.into_iter().max().unwrap_or(0)
    }

    /// Returns the number of gates on exactly two qubits: `CX` and two-qubit `MCZ` (`cz`).
    ///
    /// Gates on three or more qubits do not count; [`decompose`](Self::decompose) first.
    #[must_use]
    pub fn two_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|gate| !matches!(gate, Gate::Measure(_)) && gate.mask().count_ones() == 2)
            .count()
    }

    /// Returns an equivalent circuit whose gates all exist in OpenQASM's `stdgates.inc`.
    ///
    /// Each `MCZ` on `k ≥ 3` qubits becomes Toffoli gates on a V-chain of `k − 3` ancillas,
    /// and each Toffoli six `CX` with `H`, `T` and `T†`. The other gates stay. On the
    /// ancilla-zero subspace, the result acts as `self`; see the [module docs](self).
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the data qubits and ancillas together exceed
    /// [`MAX_QUBITS`].
    pub fn decompose(&self) -> Result<Self, TooManyQubits> {
        decompose::decompose(self)
    }

    /// Renders the circuit as an OpenQASM 3 program over `stdgates.inc`, after
    /// [`decompose`](Self::decompose).
    ///
    /// The program declares one qubit register `q` (data qubits, then ancillas) and, when
    /// the circuit measures, one bit register `c` over the data qubits.
    ///
    /// # Errors
    ///
    /// Returns [`TooManyQubits`] when the lowering needs more than [`MAX_QUBITS`] qubits.
    pub fn qasm(&self) -> Result<String, TooManyQubits> {
        Ok(qasm::render(&self.decompose()?))
    }
}

/// Applies one gate to `state`; a measurement does nothing.
fn apply_gate(state: &mut StateVector, gate: Gate) {
    match gate {
        Gate::H(qubit) => state.h(qubit),
        Gate::X(qubit) => state.x(qubit),
        Gate::Z(qubit) => state.z(qubit),
        Gate::Phase { qubit, angle } => state.phase(qubit, angle),
        Gate::Rx { qubit, angle } => state.rx(qubit, angle),
        Gate::Rz { qubit, angle } => state.rz(qubit, angle),
        Gate::Cx { control, target } => state.cx(control, target),
        Gate::Mcz(mask) => state.mcz((0..state.qubits()).filter(|qubit| mask >> qubit & 1 == 1)),
        Gate::Measure(_) => {}
    }
}

/// The gates that the algorithms of this crate apply, on a simulator or into a recording.
///
/// The algorithms write their gate sequence once against this trait, so a recorded
/// [`Circuit`] and direct application on a [`StateVector`] cannot drift apart.
pub(crate) trait Gates {
    /// Returns the number of qubits the algorithm works on.
    fn width(&self) -> usize;

    /// Applies or records the Hadamard gate on `qubit`.
    fn h(&mut self, qubit: usize);

    /// Applies or records the Pauli-X gate on `qubit`.
    fn x(&mut self, qubit: usize);

    /// Applies or records the multi-controlled Z gate on the qubits of `mask`.
    fn mcz_mask(&mut self, mask: usize);
}

/// The simulator applies the gates.
impl Gates for StateVector {
    fn width(&self) -> usize {
        self.qubits()
    }

    fn h(&mut self, qubit: usize) {
        StateVector::h(self, qubit);
    }

    fn x(&mut self, qubit: usize) {
        StateVector::x(self, qubit);
    }

    fn mcz_mask(&mut self, mask: usize) {
        apply_gate(self, Gate::Mcz(mask));
    }
}

/// A circuit records the gates on its data qubits.
impl Gates for Circuit {
    fn width(&self) -> usize {
        self.data_qubits()
    }

    fn h(&mut self, qubit: usize) {
        Circuit::h(self, qubit);
    }

    fn x(&mut self, qubit: usize) {
        Circuit::x(self, qubit);
    }

    fn mcz_mask(&mut self, mask: usize) {
        self.push(Gate::Mcz(mask));
    }
}

/// Unit tests for recording, depth and gate counts.
#[cfg(test)]
mod tests {
    use super::{Circuit, Gate};

    /// Gates on disjoint qubits share a layer; measurements add no depth.
    #[test]
    fn counts_depth_and_two_qubit_gates() {
        let mut circuit = Circuit::new(3).unwrap();
        circuit.h(0);
        circuit.h(1);
        circuit.cx(0, 2);
        circuit.mcz([1, 2]);
        circuit.measure_all();

        assert_eq!(circuit.depth(), 3);
        assert_eq!(circuit.two_qubit_gates(), 2);
        assert_eq!(circuit.gates()[3], Gate::Mcz(0b110));
    }

    /// A gate outside the register is a bug in the caller.
    #[test]
    #[should_panic(expected = "acts outside")]
    fn rejects_gates_outside_the_register() {
        Circuit::new(2).unwrap().x(2);
    }

    /// An index past the width of a mask is rejected, not shifted out of range.
    #[test]
    #[should_panic(expected = "acts outside")]
    fn rejects_indices_past_the_mask_width() {
        Circuit::new(2).unwrap().x(64);
    }
}
