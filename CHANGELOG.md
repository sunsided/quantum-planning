# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
