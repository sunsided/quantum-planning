# quantum-planning

[![crates.io](https://img.shields.io/crates/v/quantum-planning.svg)](https://crates.io/crates/quantum-planning)
[![docs.rs](https://img.shields.io/docsrs/quantum-planning)](https://docs.rs/quantum-planning)
[![CI](https://github.com/sunsided/quantum-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/quantum-planning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/quantum-planning/graph/badge.svg)](https://codecov.io/gh/sunsided/quantum-planning)
[![MSRV](https://img.shields.io/crates/msrv/quantum-planning)](https://crates.io/crates/quantum-planning)
[![License](https://img.shields.io/crates/l/quantum-planning)](https://github.com/sunsided/quantum-planning#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Quantum algorithms for planning and search, on simulators: Grover search, quantum backtracking, amplitude estimation, quantum walks on game trees and QAOA.

## Who this is for

- **Researchers, students and teachers of quantum speedups for search**: exact Grover search
  with a known number of solutions and BBHT exponential search with an unknown number, on a
  state-vector simulator. The oracle is an ordinary validity check on a candidate, such as
  "is this plan valid", so the same predicate serves a classical and a quantum search.
  Montanaro's backtracking walk, amplitude estimation of rollout values, NAND-tree
  evaluation of game trees and QAOA on QUBO models run on the same simulator.
- **Honest query-count comparisons**: every phase-oracle application and every classical
  check of a measured candidate counts as one query. Put these counts next to a classical
  scan over the same oracle, and the `√N` against `N` gap shows on real instances, not
  only in the formula. The other algorithms report walk steps, rollout calls or evolution
  time next to their classical baselines, including where the classical side still wins.
- **Hardware experiments**: Grover and QAOA circuits export as OpenQASM 3 for IBM Quantum,
  Amazon Braket or Azure Quantum, with depth and two-qubit gate counts to judge what a noisy
  device can run, and measured counts map back to candidates.
- **`no_std` demos**: the crate needs only `alloc`, so it runs in WebAssembly, for example
  in a browser page that shows the amplitudes change from one Grover iteration to the next.

It is a simulator: its memory doubles with each qubit and stops at 24 qubits, so it finds no
solution faster than a classical search does. To solve plans, use
[heuristic-search](https://github.com/sunsided/heuristic-search) for state-space search,
[constraint-search](https://github.com/sunsided/constraint-search) for constraint problems
and [game-tree-search](https://github.com/sunsided/game-tree-search) for two-player games.

## Scope

- A state-vector simulator with `H`, `X`, `Z`, phase, `RX`, `RZ`, `CX` and multi-controlled
  `Z` gates, measurement probabilities and seeded sampling (little-endian: qubit `q` is bit
  `q` of the basis index)
- Recorded circuits that run on the simulator and export as OpenQASM 3 for gate-based
  hardware (IBM Quantum, Amazon Braket, Azure Quantum): multi-controlled `Z` lowered to
  `stdgates.inc` with ancillas, depth and two-qubit gate counts, and hardware counts
  mapped back to candidate indices
- Grover search over candidate plans, with phase oracles built from validity checks, for a
  known and an unknown (BBHT exponential search) number of solutions, and as an exportable
  circuit
- Quantum backtracking (Montanaro) as a quantum walk on search trees, with detection and
  search against classical depth-first backtracking
- Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts, against
  classical Monte Carlo at the same number of rollouts
- Quantum-walk evaluation of game trees (NAND trees) by phase estimation of a
  continuous-time walk
- QAOA circuits for QUBO-encoded planning problems, behind the optional `qubo` feature
- Query-count comparisons against the classical algorithms

The algorithms run on simulators and are measured in oracle and query counts against their classical counterparts, which limits instances to at most 24 qubits. QUBO models come from [planning-qubo](https://github.com/sunsided/planning-qubo), oracles from [constraint-search](https://github.com/sunsided/constraint-search) and game trees from [game-tree-search](https://github.com/sunsided/game-tree-search).

## Usage

```sh
cargo add quantum-planning
```

An oracle marks the candidates that solve a problem; Grover search finds a marked one in
about `π/4·√(N/M)` oracle queries, where `N` candidates hold `M` marked ones. Here, a plan
picks two of four actions, and actions 0 and 1 must not run together.

```rust
use quantum_planning::grover::{search_known, search_unknown};
use quantum_planning::{Oracle, PhaseOracle, SplitMix64, StateVector};

/// Accepts plans, given as 4-bit masks over the actions, that run exactly two actions and
/// not both action 0 and action 1.
struct ValidPlan;

impl Oracle<u8> for ValidPlan {
    fn accepts(&self, plan: &u8) -> bool {
        plan.count_ones() == 2 && plan & 0b0011 != 0b0011
    }
}

assert!(ValidPlan.accepts(&0b0101));
assert!(!ValidPlan.accepts(&0b0011));

// Six plans run two actions; only one of them runs both action 0 and action 1.
let marked = (0..16_u8).filter(|plan| ValidPlan.accepts(plan)).count();

assert_eq!(marked, 5);

// Basis index `i` of a 4-qubit register is the plan with action mask `i`.
let plan = |index: usize| u8::try_from(index).ok();
let mut state = StateVector::new(4).expect("4 qubits are below the limit");
let mut rng = SplitMix64::new(7);

// With 5 of 16 plans marked, one Grover iteration and one check find a valid plan with
// probability sin²(3θ) ≈ 0.96, where sin²θ = 5/16. `None` means: run it again.
let mut oracle = PhaseOracle::new(ValidPlan, plan);
if let Some(index) = search_known(&mut state, &mut oracle, 5, &mut rng) {
    assert!(ValidPlan.accepts(&plan(index).unwrap()));
}

assert_eq!(oracle.queries(), 2);

// Without the count, BBHT exponential search stops at the first verified plan or the budget.
let mut oracle = PhaseOracle::new(ValidPlan, plan);
let index = search_unknown(&mut state, &mut oracle, 40, &mut rng).expect("a valid plan exists");

assert!(ValidPlan.accepts(&plan(index).unwrap()));
```

The query counts count oracle calls, not simulator work: one phase-oracle application or one
classical check is one query. A classical scan needs about `N/(M+1)` queries; Grover search
needs about `π/4·√(N/M)`.

A recorded Grover circuit exports as an OpenQASM 3 program. Hardware prints measured
bitstrings with the highest qubit first; `candidate_counts` reads them back as indices.

```rust
use quantum_planning::circuit::candidate_counts;
use quantum_planning::grover;

// Three qubits, index 5 marked, two iterations: about 95% success.
let circuit = grover::circuit(3, &[5], 2).expect("3 qubits are below the limit");
let lowered = circuit.decompose().expect("no ancillas needed for 3 qubits");
let qasm = circuit.qasm().expect("the lowering fits");

assert!(qasm.starts_with("OPENQASM 3.0;\ninclude \"stdgates.inc\";"));
assert_eq!(lowered.two_qubit_gates(), 24);

// "101" is c[2] = 1, c[1] = 0, c[0] = 1: index 5.
let counts = candidate_counts([("101", 950), ("011", 50)], 3).expect("valid bitstrings");

assert_eq!(counts, [(3, 50), (5, 950)]);
```

The other algorithms live in their modules: `backtracking` (Montanaro's walk on a
`SearchTree`), `amplitude_estimation` (rollout values in `[0, 1]`), `nand_tree` (balanced
NAND trees) and, with the `qubo` feature, `qaoa` (on `planning_qubo::Qubo` models). Each
reports its cost in the unit its module documents, next to a classical baseline.

The crate is `no_std` with `alloc`. It depends on `libm` and on `rand_core`: sampling and the
searches take any `R: rand_core::Rng`, and `SplitMix64` is a small seeded one. The optional
`qubo` feature adds `planning-qubo`.

## Minimum supported Rust version

Rust 1.85 (edition 2024). The MSRV applies to the library; tests may need a newer
toolchain. It may be raised in a minor release when a dependency or a language feature
calls for it.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/quantum-planning/blob/main/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/quantum-planning/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/quantum-planning/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
