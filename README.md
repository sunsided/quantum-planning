# quantum-planning

Quantum algorithms for planning and search, on simulators: Grover search, quantum backtracking, amplitude estimation, quantum walks on game trees and QAOA.

## Scope

- Grover search over candidate plans, with oracles built from validity checks
- Quantum backtracking (Montanaro) as a quantum walk on search trees
- Amplitude estimation for Monte Carlo value estimates, as used in MCTS rollouts
- Quantum-walk evaluation of game trees (NAND trees)
- QAOA circuits for QUBO-encoded planning problems
- A simulator interface, with query-count comparisons against the classical algorithms

The algorithms run on simulators and are measured in oracle and query counts against their classical counterparts, which limits instances to a few dozen qubits. QUBO models come from [planning-qubo](https://github.com/sunsided/planning-qubo), oracles from [constraint-search](https://github.com/sunsided/constraint-search) and game trees from [game-tree-search](https://github.com/sunsided/game-tree-search).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/quantum-planning/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/quantum-planning/blob/main/LICENSE-MIT))

at your option.
