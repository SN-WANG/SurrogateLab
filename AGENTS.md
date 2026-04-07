# SurrogateLab Agent Entry

Read `.codex/AGENTS.md` before making substantial changes in this repository. That file is the canonical project memory for Codex-style agents working on SurrogateLab.

The most important structural rules are:

- Treat `bench_main.py`, `bench_config.py`, `bench_funcs.py`, `abaqus_main.py`, and `abaqus_config.py` as the local fast-iteration workflow layer.
- Treat reusable surrogate, sampling, and optimization modules as WSNet-aligned shared infrastructure unless the user explicitly wants SurrogateLab-local divergence.
- Keep the Abaqus mock interface stable: instantiate `AbaqusModel` and call `run(input_arr)` with a single design vector.
