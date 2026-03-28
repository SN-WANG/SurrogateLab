# SurrogateLab

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/SurrogateLab)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**SurrogateLab** is the benchmark and experiment repository in the WSNet family. It inherits the reusable surrogate, sampling, optimization, and utility foundations from [WSNet](https://github.com/SN-WANG/WSNet), while keeping benchmark scripts and research workflows local to this repository.

## 📌 Overview

SurrogateLab is where algorithm comparison and experiment-facing workflow code live.
It focuses on benchmark functions, application-style demos, and quick iteration around surrogate modeling methods.

The current scope includes:

- analytic benchmark suites
- Abaqus-style application demos
- ensemble and multi-fidelity surrogate comparisons
- active learning experiments
- constrained and unconstrained optimization studies

## ✨ Highlights

- Classical surrogate models: `PRS`, `RBF`, `KRG`, `SVR`
- Multi-fidelity models: `MFSMLS`, `MMFS`, `CCAMFS`
- Ensemble models: `TAHS`, `AESMSI`
- Sampling methods for DOE and infill
- Global optimization helpers: `MIGA`, `CFSSDA`
- Two main experiment entry points: `bench_main.py` and `abaqus_main.py`

## 🧱 Repository Layout

```text
SurrogateLab/
├── bench_main.py
├── bench_config.py
├── bench_funcs.py
├── abaqus_main.py
├── abaqus_config.py
├── models/
│   ├── classical/
│   ├── ensemble/
│   ├── multi_fidelity/
│   └── optimization/
├── sampling/
├── utils/
├── README.md
└── LICENSE
```

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/SN-WANG/SurrogateLab.git
cd SurrogateLab
```

### Install the dependencies you need

```bash
pip install numpy scipy matplotlib torch tqdm
```

### Run the analytic benchmark suite

```bash
python bench_main.py --demos all
```

### Run the Abaqus-style demo

```bash
python abaqus_main.py --demos all --visualize
```

### Run only the optimization demos

```bash
python bench_main.py --demos I J
python abaqus_main.py --demos I J --visualize
```

## 🔗 Relationship to WSNet

SurrogateLab is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while SurrogateLab keeps the benchmark definitions, research scripts, and application-facing experiment flows.

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
