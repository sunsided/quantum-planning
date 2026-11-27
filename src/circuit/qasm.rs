//! Rendering of lowered circuits as OpenQASM 3 text over `stdgates.inc`.

use alloc::string::String;
use core::f64::consts::FRAC_PI_4;
use core::fmt::Write;

use super::{Circuit, Gate};

/// Renders `circuit`, whose `MCZ` gates act on at most two qubits, as an OpenQASM 3 program.
///
/// # Panics
///
/// Panics when an `MCZ` acts on three or more qubits; [`Circuit::qasm`] lowers those first.
pub(super) fn render(circuit: &Circuit) -> String {
    let mut out = String::new();

    // Writing to a `String` cannot fail.
    write_program(&mut out, circuit).expect("formatting into a String succeeds");

    out
}

/// Writes the header, the registers and one line per gate.
fn write_program(out: &mut String, circuit: &Circuit) -> core::fmt::Result {
    writeln!(out, "OPENQASM 3.0;")?;
    writeln!(out, "include \"stdgates.inc\";")?;
    writeln!(out, "qubit[{}] q;", circuit.qubits())?;

    if circuit
        .gates()
        .iter()
        .any(|gate| matches!(gate, Gate::Measure(_)))
    {
        writeln!(out, "bit[{}] c;", circuit.data_qubits())?;
    }

    for gate in circuit.gates() {
        write_gate(out, *gate)?;
    }

    Ok(())
}

/// Writes one gate as one statement.
#[expect(
    clippy::float_cmp,
    reason = "the lowering emits exactly ±FRAC_PI_4 for T and T†; any other angle stays p(θ)"
)]
fn write_gate(out: &mut String, gate: Gate) -> core::fmt::Result {
    match gate {
        Gate::H(qubit) => writeln!(out, "h q[{qubit}];"),
        Gate::X(qubit) => writeln!(out, "x q[{qubit}];"),
        Gate::Z(qubit) => writeln!(out, "z q[{qubit}];"),
        Gate::Phase { qubit, angle } if angle == FRAC_PI_4 => writeln!(out, "t q[{qubit}];"),
        Gate::Phase { qubit, angle } if angle == -FRAC_PI_4 => writeln!(out, "tdg q[{qubit}];"),
        Gate::Phase { qubit, angle } => writeln!(out, "p({angle}) q[{qubit}];"),
        Gate::Rx { qubit, angle } => writeln!(out, "rx({angle}) q[{qubit}];"),
        Gate::Rz { qubit, angle } => writeln!(out, "rz({angle}) q[{qubit}];"),
        Gate::Cx { control, target } => writeln!(out, "cx q[{control}], q[{target}];"),
        Gate::Mcz(mask) => write_mcz(out, mask),
        Gate::Measure(qubit) => writeln!(out, "c[{qubit}] = measure q[{qubit}];"),
    }
}

/// Writes an `MCZ` on at most two qubits: `gphase(π)`, `z` or `cz`.
///
/// # Panics
///
/// Panics when `mask` has three or more bits.
fn write_mcz(out: &mut String, mask: usize) -> core::fmt::Result {
    let low = mask.trailing_zeros();
    let high = usize::BITS - 1 - mask.leading_zeros();

    match mask.count_ones() {
        0 => writeln!(out, "gphase(pi);"),
        1 => writeln!(out, "z q[{low}];"),
        2 => writeln!(out, "cz q[{low}], q[{high}];"),
        _ => unreachable!("decompose lowers every MCZ to at most two qubits"),
    }
}
