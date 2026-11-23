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
- **Honest query-count comparisons**: every phase-oracle application and every classical
  check of a measured candidate counts as one query. Put these counts next to a classical
  scan over the same oracle, and the `√N` against `N` gap shows on real instances, not
  only in the formula.
- **`no_std` demos**: the crate needs only `alloc`, so it runs in WebAssembly, for example
  in a browser page that shows the amplitudes change from one Grover iteration to the next.

It is a simulator: its memory doubles with each qubit and stops at 24 qubits, so it finds no
solution faster than a classical search does. To solve plans, use
[heuristic-search](https://github.com/sunsided/heuristic-search) for state-space search,
[constraint-search](https://github.com/sunsided/constraint-search) for constraint problems
and [game-tree-search](https://github.com/sunsided/game-tree-search) for two-player games.
Quantum backtracking, amplitude estimation, NAND-tree walks and QAOA on
[planning-qubo](https://github.com/sunsided/planning-qubo) models are planned, not built yet.

## Scope

- A state-vector simulator with `H`, `X`, `Z`, phase and multi-controlled `Z` gates,
  measurement probabilities and seeded sampling (little-endian: qubit `q` is bit `q` of the
  basis index)
- Grover search over candidate plans, with phase oracles built from validity checks, for a
  known and an unknown (BBHT exponential search) number of solutions
- Quantum backtracking (Montanaro) as a quantum walk on search trees (planned)
- Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts (planned)
- Quantum-walk evaluation of game trees (NAND trees) (planned)
- QAOA circuits for QUBO-encoded planning problems (planned)
- Query-count comparisons against the classical algorithms

The algorithms run on simulators and are measured in oracle and query counts against their classical counterparts, which limits instances to a few dozen qubits. QUBO models come from [planning-qubo](https://github.com/sunsided/planning-qubo), oracles from [constraint-search](https://github.com/sunsided/constraint-search) and game trees from [game-tree-search](https://github.com/sunsided/game-tree-search).

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

The crate is `no_std` with `alloc`. It depends on `libm` and on `rand_core`: sampling and the
searches take any `R: rand_core::Rng`, and `SplitMix64` is a small seeded one.

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
