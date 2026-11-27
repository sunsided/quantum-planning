# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/sunsided/quantum-planning/compare/v0.1.0...v0.1.1) - 2026-11-27

### Added

- Add QAOA for QUBO models behind the qubo feature
- Add NAND-tree evaluation by a continuous-time quantum walk
- Add Montanaro quantum backtracking on search trees
- Add amplitude estimation of Monte Carlo rollout values
- Add OpenQASM 3 export of recorded circuits

### Changed

- Update the README audience for the algorithms added in #9
- Test error messages, the QAOA size limit and zero Ising fields
- Bound the NAND walk graph and reject zero-run detections

### Fixed

- Fix OpenQASM rendering of an empty multi-controlled Z

## [0.1.0](https://github.com/sunsided/quantum-planning/releases/tag/v0.1.0) - 2026-11-23

### Added

- Add a "Who this is for" section to the README
- Add state-vector simulator and Grover search over the Oracle trait
- Add dormant release automation with release-plz
- Add CONTRIBUTING.md
- Add git hooks and Copilot review instructions
- Add README badges, a usage example and package excludes
- Add REUSE licensing metadata
- Add supply-chain checks
- Add Codecov configuration
- Add Taskfile and CI workflows

### Changed

- Set the crate version to 0.1.0
- Take randomness through rand_core::Rng
- Pin the per-call BBHT budget and document short budgets
- Make the BBHT query budget count only the current call
- Test the Oracle trait through a minimal plan-validity oracle
- Tighten lints and forbid unsafe code
- Render mermaid diagrams on docs.rs and doctest the README
- Reserve the crate name with a 0.0.0 placeholder
