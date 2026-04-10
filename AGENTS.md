# SurrogateLab Agent Entry

Read `.codex/AGENTS.md` before making substantial changes in this repository. That file is the canonical project memory for Codex-style agents working on SurrogateLab.

The most important structural rules are:

- Treat `bench_main.py`, `bench_config.py`, `bench_funcs.py`, `case_main.py`, and `case_config.py` as the local fast-iteration workflow layer.
- Treat reusable surrogate, sampling, and optimization modules as WSNet-aligned shared infrastructure unless the user explicitly wants SurrogateLab-local divergence.
- Keep the Abaqus mock interface stable: instantiate `AbaqusModel` and call `run(input_arr)` with a single design vector.
- Keep default engineering sample counts aligned with the contract-facing case config: `num_train=30`, `num_test=50`, `num_lf=30`, `num_hf=15`, `num_active_initial=3`, `num_infill=21`.
- Prefer the real algorithm names directly in configs and workflow entry points instead of alias-heavy demo labels.
- Keep workflow outputs fixed at the repository root: `bench_results.json`, `case_doe_cache.npy`, and `case_results.json`.
