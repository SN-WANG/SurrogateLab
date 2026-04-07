# SurrogateLab Project AGENTS

## Project Snapshot

- SurrogateLab is the benchmark and engineering-validation repository in the WSNet family.
- The repository has two main execution paths:
  - analytic benchmarks: `bench_main.py -> bench_config.py -> bench_funcs.py`
  - engineering validation: `abaqus_main.py -> abaqus_config.py -> AbaqusModel`
- The current engineering workflow is centered on two outputs only:
  - `weight`
  - `stress_skin`

## Canonical Sources of Truth

- Treat the current repository code as the source of truth when older notes, reports, or task documents disagree with it.
- `bench_funcs.py` is the canonical registry for the current analytic benchmark functions and their names.
- `bench_config.py` is the canonical place for current default analytic validation settings.
- `abaqus_config.py` is the canonical place for current default engineering validation settings.
- The current local `AbaqusModel` is only a mock, but its public call contract must remain compatible with the real remote workflow: instantiate the model and call `run(input_arr)`.

## WSNet Relationship

- WSNet is the reusable upstream for mature surrogate, sampling, optimization, and utility modules.
- Prefer syncing mature shared fixes through WSNet instead of quietly forking shared behavior inside SurrogateLab.
- SurrogateLab should own:
  - benchmark orchestration
  - engineering-case orchestration
  - benchmark-function registries
  - contract-facing reporting defaults
- Shared model implementations should stay stylistically aligned with WSNet even when they are copied locally.

## Metric Model

- The repository currently uses two evaluation metrics only:
  - `accuracy`
  - `r2`
- Accuracy is the primary metric and is defined as:
  - `accuracy = (1 - sum(abs(GT - Pred)) / (sum(abs(GT)) + eps)) * 100`
- `r2` is retained as auxiliary context, not the primary pass metric.

## Current Default Validation Targets

- Analytic ensemble defaults use a `10%` relative accuracy-gain target over the mean single-model baseline.
- Analytic multi-fidelity defaults currently use an `87%` minimum accuracy threshold.
- Analytic active-learning defaults use a `20%` relative accuracy-gain target.
- Engineering ensemble defaults use a `10%` relative accuracy-gain target.
- Engineering multi-fidelity defaults use a `90%` minimum accuracy threshold.
- Engineering active-learning defaults use a `20%` relative accuracy-gain target.
- Engineering optimization defaults use:
  - objective: `stress_skin`
  - constraint: `weight <= 0.31`

## Current Engineering Defaults

- Default engineering training counts are tuned to make the contract-style engineering checks practical under the current mock model:
  - `num_train = 12`
  - `num_lf = 80`
  - `num_hf = 30`
  - `num_active_initial = 2`
  - `num_infill = 10`
- The current engineering mock scales `weight` to an engineering-style range compatible with the `0.31` constraint while preserving the same public interface.
- Only `weight` and `stress_skin` are first-class engineering validation targets right now; the other two mock outputs can remain present but are not the focus of default reporting.

## Analytic Benchmark Notes

- Keep the fixed benchmark-function identities unless the user explicitly asks to replace them.
- The current analytic defaults are tuned around the existing E-AHF-style function set.
- The `branin` multi-fidelity case is the limiting case for the current three multi-fidelity surrogate models, which is why the current default analytic MF threshold is not set to `90%`.

## Active Learning Notes

- `sampling/so_infill.py` is treated as the WSNet-inherited baseline and should not be modified casually.
- `sampling/diso_infill.py` is the SurrogateLab-local distance-informed extension for single-objective active learning.
- `DISOInfill` is the preferred single-objective active-learning class in current SurrogateLab workflows.

## Practical Change Strategy

- Prefer small, explicit helper functions over large monolithic scripts.
- Keep benchmark and engineering defaults easy to audit from the config files.
- When changing thresholds or sample counts, preserve the benchmark-function identities and the engineering output focus unless the user explicitly asks otherwise.
- When changing the Abaqus mock, protect the call interface first and the exact local numeric scaling second.

## Non-Goals

- Do not silently redesign shared optimization or sampling APIs in SurrogateLab if the real change belongs in WSNet.
- Do not replace the current benchmark functions just to force higher scores.
- Do not widen the engineering workflow back to all four Abaqus outputs unless the user explicitly asks for that.
