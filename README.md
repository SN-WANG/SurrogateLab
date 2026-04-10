# SurrogateLab

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/SurrogateLab)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**SurrogateLab** is the benchmark and engineering-validation repository in the WSNet family. It keeps experiment-facing surrogate-model workflows local to this repository while reusing the shared surrogate, sampling, optimization, and utility foundations from [WSNet](https://github.com/SN-WANG/WSNet).

## 📌 Overview

SurrogateLab currently has two main tracks:

- analytic benchmark validation through `bench_main.py`
- engineering-case validation through `case_main.py`

The repository is designed around the ten contract-facing algorithm demos:

- adaptive ensemble surrogate models
- multi-fidelity surrogate models
- active learning and sequential sampling methods
- global optimization methods

The current refactor unifies evaluation around **accuracy** and **R2**, with accuracy used as the primary metric:

```text
accuracy = (1 - sum(|GT - Pred|) / (sum(|GT|) + eps)) * 100%
```

## ✨ Highlights

- Classical surrogate models: `PRS`, `RBF`, `KRG`, `SVR`
- Ensemble models: `TAHS`, `AESMSI`
- Multi-fidelity models: `MFSMLS`, `MMFS`, `CCAMFS`
- Single-objective distance-informed active learning via `DISO`
- WSNet-synced optimization backends: `MIGA`, `CFARSSDA`
- Two clean entry points for analytic and engineering workflows

## 🧪 Current Coverage

### Analytic benchmark suite

- 3 ensemble benchmark functions × 2 ensemble algorithms
- 3 multi-fidelity benchmark functions × 3 multi-fidelity algorithms
- 1 single-objective active-learning benchmark
- 1 multi-fidelity active-learning benchmark
- 1 multi-objective active-learning benchmark
- 3 unconstrained optimization benchmark functions

### Engineering Case Suite

- Engineering targets: `weight` and `stress_skin`
- 4 ensemble validation runs
- 6 multi-fidelity validation runs
- 2 single-objective active-learning validation runs with `DISO`
- 2 constrained optimization runs with `stress_skin` as the objective and `weight <= 0.31` as the constraint

## 🧱 Repository Layout

```text
SurrogateLab/
├── bench_main.py
├── bench_config.py
├── bench_funcs.py
├── case_main.py
├── case_config.py
├── models/
│   ├── classical/
│   ├── ensemble/
│   ├── multi_fidelity/
│   └── optimization/
├── sampling/
│   ├── doe.py
│   ├── mf_infill.py
│   ├── mo_infill.py
│   ├── so_infill.py
│   └── diso_infill.py
├── utils/
├── README.md
└── LICENSE
```

## 🚀 Running Experiments

### Clone the repository

```bash
git clone https://github.com/SN-WANG/SurrogateLab.git
cd SurrogateLab
```

### Install the dependencies you need

```bash
pip install numpy scipy torch tqdm
```

### Run the analytic benchmark suite

```bash
python bench_main.py
```

This writes `bench_results.json` to the repository root.

### Run the engineering case suite

```bash
python case_main.py
```

This writes `case_doe_cache.npy` and `case_results.json` to the repository root.

### Run only the active-learning demos

```bash
python bench_main.py --demos DISO MICO MOBO
python case_main.py --demos DISO
```

### Run only the optimization demos

```bash
python bench_main.py --demos MIGA CFARSSDA
python case_main.py --demos MIGA CFARSSDA
```

## 🔗 Relationship to WSNet

SurrogateLab is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while SurrogateLab keeps the benchmark definitions, engineering-case orchestration, and contract-facing validation workflows.

## 📚 Citation

If SurrogateLab is useful in your work, please cite it as a software project.

```bibtex
@software{surrogatelab2026,
  author = {Shengning Wang},
  title = {SurrogateLab},
  year = {2026},
  url = {https://github.com/SN-WANG/SurrogateLab}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
