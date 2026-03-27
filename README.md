# SurrogateLab

SurrogateLab is a research-oriented repository for engineering surrogate modeling workflows. It focuses on classical surrogate models, multi-fidelity methods, ensemble strategies, design of experiments, and infill sampling, while keeping optimization as a downstream application rather than the repository's main identity.

This repository is split from [WSNet](https://github.com/SN-WANG/WSNet). The git history is intentionally preserved so the lineage remains traceable, while the codebase itself keeps the existing implementation layout from the previous standalone surrogate-modeling workspace.

## Scope

- Classical surrogate models
- Multi-fidelity surrogate models
- Ensemble surrogate models
- DOE and infill sampling
- Benchmark and Abaqus-oriented application entry points

## Repository Layout

```text
SurrogateLab/
├── abaqus_config.py
├── abaqus_main.py
├── bench_config.py
├── bench_funcs.py
├── bench_main.py
├── models/
│   ├── classical/
│   ├── multi_fidelity/
│   ├── ensemble/
│   └── optimization/
├── sampling/
└── utils/
```

## Included Modules

### Classical Models

- `PRS`
- `RBF`
- `KRG`
- `SVR`

### Multi-Fidelity Models

- `CCA-MFS`
- `MFS-MLS`
- `MMFS`

### Ensemble Models

- `T-AHS`
- `AES-MSI`

### Sampling and Optimization

- DOE utilities
- Single-objective infill
- Multi-objective infill
- Multi-fidelity infill
- Dragonfly-based optimization helper

## Notes

- The current code is kept close to the existing implementation without logic refactoring.
- Path conventions and runtime details are preserved intentionally for later manual cleanup and validation.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
