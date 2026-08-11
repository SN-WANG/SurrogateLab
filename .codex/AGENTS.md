# SurrogateLab Project AGENTS

## Project Snapshot

- SurrogateLab is the benchmark and engineering-validation repository in the WSNet family.
- The repository has two main execution paths:
  - analytic benchmarks: `bench_main.py -> bench_config.py -> bench_funcs.py`
  - engineering validation: `case_main.py -> case_config.py -> AnsysModel`
- The current engineering workflow has four ANSYS responses and four validation targets:
  - raw responses and validation targets: `mass`, `total_deformation`, `temperature`, `equivalent_stress`

## Canonical Sources of Truth

- Treat the current repository code as the source of truth when older notes, reports, or task documents disagree with it.
- `bench_funcs.py` is the canonical registry for the current analytic benchmark functions and their names.
- `bench_config.py` is the canonical place for current default analytic validation settings.
- `case_config.py` is the canonical place for current default engineering validation settings.
- The engineering `AnsysModel` calls the real external `runwb2` solver. Its public contract is to instantiate the model and call `run(input_arr)` with the 4-vector `[ti65, aerogel, sic, mesh_size]`.
- Engineering runs must fail clearly when `runwb2` is unavailable; no local proxy fallback is permitted.
- Multi-fidelity is controlled by the mesh size: high fidelity uses `50 [mm]`, low fidelity uses `100 [mm]`; surrogate models keep the three thickness inputs.

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
- Analytic multi-fidelity defaults currently use a `90%` minimum accuracy threshold.
- Analytic active-learning defaults use a `20%` relative accuracy-gain target.
- Engineering ensemble defaults use a `10%` relative accuracy-gain target.
- Engineering multi-fidelity defaults use a `90%` minimum accuracy threshold.
- Engineering active-learning defaults use a `20%` relative accuracy-gain target.
- Engineering optimization defaults use:
  - single-objective mode: minimize `mass` with constraints `equivalent_stress <= 500 [MPa]` and `temperature <= 150 [C]`
  - multi-objective mode: minimize `[mass, temperature]` with constraint `equivalent_stress <= 500 [MPa]`

## Current Engineering Defaults

- Default engineering sample counts follow the current contract-style case configuration:
  - `num_train = 30`
  - `num_test = 50`
  - `num_lf = 30`
  - `num_hf = 15`
  - `num_active_initial = 2`
  - `num_infill = 21`
- All four ANSYS responses are first-class engineering validation targets.

## Naming Defaults

- Use the real external algorithm names directly in configs, logs, and result payloads:
  - `TAHS`
  - `AESMSI`
  - `MFSMLS`
  - `MMFS`
  - `CCAMFS`
  - `DISO`
  - `MICO`
  - `MOBO`
  - `MIGA`
  - `CFARSSDA`
- Do not rely on `A/B/C/...` labels or other alias-heavy config parsing as the default workflow interface.

## Analytic Benchmark Notes

- Keep the fixed benchmark-function identities unless the user explicitly asks to replace them.
- The current analytic defaults are tuned around the existing E-AHF-style function set.
- The `branin` multi-fidelity case remains the limiting case for the current three multi-fidelity surrogate models.

## Active Learning Notes

- `sampling/so_infill.py` is treated as the WSNet-inherited baseline and should not be modified casually.
- `sampling/diso_infill.py` is the SurrogateLab-local distance-informed extension for single-objective active learning.
- `DISO` is the preferred outward label for single-objective active learning, implemented by `DISOInfill` internally.

## Practical Change Strategy

- Prefer small, explicit helper functions over large monolithic scripts.
- Keep benchmark and engineering defaults easy to audit from the config files.
- When changing thresholds or sample counts, preserve the benchmark-function identities and the engineering output focus unless the user explicitly asks otherwise.
- When changing the ANSYS interface, protect the four-response ordering, the journal batch workflow, and the external-only failure semantics.
- Default outputs are written directly to the repository root as `bench_results.json` for analytic runs and `case_doe_cache.npy` / `case_results.json` for engineering runs.
- The unified `main.py` quality gate runs the engineering entry first and therefore also requires `runwb2`.

## Non-Goals

- Do not silently redesign shared optimization or sampling APIs in SurrogateLab if the real change belongs in WSNet.
- Do not replace the current benchmark functions just to force higher scores.
- Do not add proxy or cached simulation data for the engineering workflow.
