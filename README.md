# quantum-planning

[![crates.io](https://img.shields.io/crates/v/quantum-planning.svg)](https://crates.io/crates/quantum-planning)
[![docs.rs](https://img.shields.io/docsrs/quantum-planning)](https://docs.rs/quantum-planning)
[![CI](https://github.com/sunsided/quantum-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/quantum-planning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/quantum-planning/graph/badge.svg)](https://codecov.io/gh/sunsided/quantum-planning)
[![MSRV](https://img.shields.io/crates/msrv/quantum-planning)](https://crates.io/crates/quantum-planning)
[![License](https://img.shields.io/crates/l/quantum-planning)](https://github.com/sunsided/quantum-planning#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Quantum algorithms for planning and search, on simulators: Grover search, quantum backtracking, amplitude estimation, quantum walks on game trees and QAOA.

## Scope

- Grover search over candidate plans, with oracles built from validity checks
- Quantum backtracking (Montanaro) as a quantum walk on search trees
- Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts
- Quantum-walk evaluation of game trees (NAND trees)
- QAOA circuits for QUBO-encoded planning problems
- A simulator interface, with query-count comparisons against the classical algorithms

The algorithms run on simulators and are measured in oracle and query counts against their classical counterparts, which limits instances to a few dozen qubits. QUBO models come from [planning-qubo](https://github.com/sunsided/planning-qubo), oracles from [constraint-search](https://github.com/sunsided/constraint-search) and game trees from [game-tree-search](https://github.com/sunsided/game-tree-search).

## Usage

```sh
cargo add quantum-planning
```

An oracle marks the candidates that solve a problem; Grover search finds a marked one in
about `π/4·√(N/M)` oracle queries, where `N` candidates hold `M` marked ones. Here, a plan
picks two of four actions, and actions 0 and 1 must not run together.

```rust
use quantum_planning::Oracle;

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

// With 5 of 16 plans marked, one Grover iteration already gives the best success chance.
let iterations = (core::f64::consts::FRAC_PI_4 * (16.0 / 5.0_f64).sqrt()).floor();

assert_eq!(iterations, 1.0);
```

The crate is `no_std` and has no dependencies.

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
