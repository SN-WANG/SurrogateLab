# Scalable boundary benchmark and HTML report runner for SurrogateLab
# Author: Shengning Wang

import argparse
import base64
import json
import os
import random
import time
from datetime import datetime
from io import BytesIO
from statistics import mean
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/surrogatelab-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/surrogatelab-cache")

ArrayFn = Callable[[np.ndarray], np.ndarray]
BoundsTuple = Tuple[Tuple[float, float], ...]


def _as_any_2d_array(x: np.ndarray, name: str) -> np.ndarray:
    """
    Convert input samples to a 2-D float64 array.

    Args:
        x (np.ndarray): Input samples. (N, D) or (D,).
        name (str): Function name.

    Returns:
        np.ndarray: Input samples. (N, D).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} expects shape (N, D) or (D,), got {arr.shape}.")
    return arr


def _as_2d_array(x: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    """
    Convert input samples to a 2-D float64 array with a fixed dimension.

    Args:
        x (np.ndarray): Input samples. (N, D) or (D,).
        expected_dim (int): Expected feature dimension.
        name (str): Function name.

    Returns:
        np.ndarray: Input samples. (N, D).
    """
    arr = _as_any_2d_array(x, name)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"{name} expects shape (N, {expected_dim}) or ({expected_dim},), got {arr.shape}.")
    return arr


def _repeat_bounds(input_dim: int, lower: float, upper: float) -> BoundsTuple:
    """
    Build repeated box bounds for a scalable benchmark.

    Args:
        input_dim (int): Input dimension.
        lower (float): Lower bound.
        upper (float): Upper bound.

    Returns:
        BoundsTuple: Box bounds. (D, 2).
    """
    return tuple((float(lower), float(upper)) for _ in range(input_dim))


def _unit_coordinates(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Map bounded physical coordinates into the unit hypercube.

    Args:
        x (np.ndarray): Physical samples. (N, D).
        bounds (np.ndarray): Box bounds. (D, 2).

    Returns:
        np.ndarray: Unit-hypercube samples. (N, D).
    """
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


@dataclass(frozen=True)
class ScalarBenchmark:
    """
    Scalar-output benchmark specification.

    Args:
        name (str): Dimension-specific benchmark name.
        family (str): Scalable benchmark family name.
        input_dim (int): Input dimension.
        bounds (BoundsTuple): Box bounds. (D, 2).
        output_name (str): Output name.
        description (str): Short description.
        evaluator (ArrayFn): Benchmark function. (N, D) -> (N, 1).
        known_optimum (Optional[float]): Known minimum value.
        known_minimizer (Optional[Tuple[float, ...]]): One known minimizer. (D,).
    """

    name: str
    family: str
    input_dim: int
    bounds: BoundsTuple
    output_name: str
    description: str
    evaluator: ArrayFn
    known_optimum: Optional[float] = None
    known_minimizer: Optional[Tuple[float, ...]] = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the scalar benchmark.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: Responses. (N, 1).
        """
        x_arr = _as_2d_array(x, self.input_dim, self.name)
        return self.evaluator(x_arr)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Return bounds as a float64 array.

        Returns:
            np.ndarray: Box bounds. (D, 2).
        """
        return np.asarray(self.bounds, dtype=np.float64)


@dataclass(frozen=True)
class MultiFidelityBenchmark:
    """
    Multi-fidelity benchmark specification.

    Args:
        name (str): Dimension-specific benchmark name.
        family (str): Scalable benchmark family name.
        input_dim (int): Input dimension.
        bounds (BoundsTuple): Box bounds. (D, 2).
        output_name (str): Output name.
        description (str): Short description.
        high_fidelity (ArrayFn): HF function. (N, D) -> (N, 1).
        low_fidelity (ArrayFn): LF function. (N, D) -> (N, 1).
    """

    name: str
    family: str
    input_dim: int
    bounds: BoundsTuple
    output_name: str
    description: str
    high_fidelity: ArrayFn
    low_fidelity: ArrayFn

    def evaluate_high_fidelity(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the high-fidelity response.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: HF responses. (N, 1).
        """
        x_arr = _as_2d_array(x, self.input_dim, self.name)
        return self.high_fidelity(x_arr)

    def evaluate_low_fidelity(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the low-fidelity response.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: LF responses. (N, 1).
        """
        x_arr = _as_2d_array(x, self.input_dim, self.name)
        return self.low_fidelity(x_arr)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Return bounds as a float64 array.

        Returns:
            np.ndarray: Box bounds. (D, 2).
        """
        return np.asarray(self.bounds, dtype=np.float64)


@dataclass(frozen=True)
class MultiObjectiveBenchmark:
    """
    Multi-objective benchmark specification.

    Args:
        name (str): Dimension-specific benchmark name.
        family (str): Scalable benchmark family name.
        input_dim (int): Input dimension.
        bounds (BoundsTuple): Box bounds. (D, 2).
        output_names (Tuple[str, ...]): Objective names.
        description (str): Short description.
        evaluator (ArrayFn): Objective function. (N, D) -> (N, M).
    """

    name: str
    family: str
    input_dim: int
    bounds: BoundsTuple
    output_names: Tuple[str, ...]
    description: str
    evaluator: ArrayFn

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the multi-objective benchmark.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: Objective values. (N, M).
        """
        x_arr = _as_2d_array(x, self.input_dim, self.name)
        return self.evaluator(x_arr)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Return bounds as a float64 array.

        Returns:
            np.ndarray: Box bounds. (D, 2).
        """
        return np.asarray(self.bounds, dtype=np.float64)


# ============================================================
# Scalable Scalar Functions
# ============================================================

def sphere(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the dimension-normalized Sphere benchmark.

    Args:
        x (np.ndarray): Query points. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "sphere")
    y = np.mean(x_arr ** 2, axis=1)
    return y.reshape(-1, 1)


def ackley(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Ackley benchmark.

    Args:
        x (np.ndarray): Query points. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "ackley")
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    sq_term = np.sqrt(np.mean(x_arr ** 2, axis=1))
    cos_term = np.mean(np.cos(c * x_arr), axis=1)
    y = -a * np.exp(-b * sq_term) - np.exp(cos_term) + a + np.e
    return y.reshape(-1, 1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the dimension-normalized Rastrigin benchmark.

    Args:
        x (np.ndarray): Query points. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "rastrigin")
    y = 10.0 + np.mean(x_arr ** 2 - 10.0 * np.cos(2.0 * np.pi * x_arr), axis=1)
    return y.reshape(-1, 1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the dimension-normalized Rosenbrock benchmark.

    Args:
        x (np.ndarray): Query points. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "rosenbrock")
    if x_arr.shape[1] == 1:
        y = (1.0 - x_arr[:, 0]) ** 2
    else:
        terms = 100.0 * (x_arr[:, 1:] - x_arr[:, :-1] ** 2) ** 2 + (1.0 - x_arr[:, :-1]) ** 2
        y = np.mean(terms, axis=1)
    return y.reshape(-1, 1)


def griewank(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Griewank benchmark.

    Args:
        x (np.ndarray): Query points. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "griewank")
    idx = np.sqrt(np.arange(1, x_arr.shape[1] + 1, dtype=np.float64))
    y = 1.0 + np.sum(x_arr ** 2, axis=1) / 4000.0 - np.prod(np.cos(x_arr / idx), axis=1)
    return y.reshape(-1, 1)


def sobol_g(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Sobol-G sensitivity benchmark.

    Args:
        x (np.ndarray): Query points in the unit hypercube. (N, D) or (D,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_any_2d_array(x, "sobol_g")
    input_dim = x_arr.shape[1]
    a_values = 0.5 + np.arange(input_dim, dtype=np.float64)
    if input_dim >= 4:
        a_values[:4] = np.array([0.0, 0.5, 3.0, 9.0], dtype=np.float64)
    terms = (np.abs(4.0 * x_arr - 2.0) + a_values) / (1.0 + a_values)
    y = np.prod(terms, axis=1)
    return y.reshape(-1, 1)


# ============================================================
# Scalable Multi-Objective Functions
# ============================================================

def dtlz2(x: np.ndarray, num_objectives: int = 2) -> np.ndarray:
    """
    Evaluate the DTLZ2 multi-objective benchmark.

    Args:
        x (np.ndarray): Query points in the unit hypercube. (N, D) or (D,).
        num_objectives (int): Number of objectives.

    Returns:
        np.ndarray: Objective values. (N, M).
    """
    x_arr = _as_any_2d_array(x, "dtlz2")
    M = num_objectives
    g = np.sum((x_arr[:, M - 1:] - 0.5) ** 2, axis=1)
    values = []
    for m in range(M):
        y = 1.0 + g
        for j in range(M - m - 1):
            y = y * np.cos(0.5 * np.pi * x_arr[:, j])
        if m > 0:
            y = y * np.sin(0.5 * np.pi * x_arr[:, M - m - 1])
        values.append(y)
    return np.column_stack(values)


# ============================================================
# Benchmark Factories
# ============================================================

@dataclass(frozen=True)
class ScalarFamily:
    """
    Scalable scalar benchmark family descriptor.

    Args:
        name (str): Family name.
        evaluator (ArrayFn): Scalable evaluator. (N, D) -> (N, 1).
        bounds (Tuple[float, float]): Repeated scalar bounds.
        description (str): Short description.
        optimum (float): Known global minimum.
        minimizer_value (float): Repeated coordinate of a known minimizer.
        default_dim (int): Default dimension.
    """

    name: str
    evaluator: ArrayFn
    bounds: Tuple[float, float]
    description: str
    optimum: float
    minimizer_value: float
    default_dim: int


SCALAR_FAMILY_REGISTRY: Dict[str, ScalarFamily] = {
    "sphere": ScalarFamily(
        name="sphere",
        evaluator=sphere,
        bounds=(-1.0, 1.0),
        description="Smooth separable baseline with no local traps.",
        optimum=0.0,
        minimizer_value=0.0,
        default_dim=20,
    ),
    "sobol_g": ScalarFamily(
        name="sobol_g",
        evaluator=sobol_g,
        bounds=(0.0, 1.0),
        description="Unit-hypercube sensitivity benchmark with controllable effective dimension.",
        optimum=0.0,
        minimizer_value=0.5,
        default_dim=20,
    ),
    "ackley": ScalarFamily(
        name="ackley",
        evaluator=ackley,
        bounds=(-5.0, 5.0),
        description="Flat outer landscape with many local minima.",
        optimum=0.0,
        minimizer_value=0.0,
        default_dim=20,
    ),
    "rastrigin": ScalarFamily(
        name="rastrigin",
        evaluator=rastrigin,
        bounds=(-5.12, 5.12),
        description="Highly multimodal separable landscape.",
        optimum=0.0,
        minimizer_value=0.0,
        default_dim=20,
    ),
    "rosenbrock": ScalarFamily(
        name="rosenbrock",
        evaluator=rosenbrock,
        bounds=(-2.048, 2.048),
        description="Narrow coupled valley with difficult extrapolation behavior.",
        optimum=0.0,
        minimizer_value=1.0,
        default_dim=20,
    ),
    "griewank": ScalarFamily(
        name="griewank",
        evaluator=griewank,
        bounds=(-10.0, 10.0),
        description="Weakly coupled multimodal landscape with regularly distributed local minima.",
        optimum=0.0,
        minimizer_value=0.0,
        default_dim=20,
    ),
}


def _parse_dimension_name(name: str) -> Tuple[str, Optional[int]]:
    """
    Split a benchmark name with an optional ``_d`` dimension suffix.

    Args:
        name (str): Benchmark name.

    Returns:
        Tuple[str, Optional[int]]: Family name and parsed dimension.
    """
    key = name.lower()
    if "_d" not in key:
        return key, None
    family, dim_text = key.rsplit("_d", 1)
    return family, int(dim_text)


def make_scalar_benchmark(name: str, input_dim: int) -> ScalarBenchmark:
    """
    Build a scalar benchmark with the requested dimension.

    Args:
        name (str): Scalable benchmark family name.
        input_dim (int): Input dimension.

    Returns:
        ScalarBenchmark: Dimension-specific benchmark.
    """
    family = SCALAR_FAMILY_REGISTRY[name.lower()]
    bounds = _repeat_bounds(input_dim, family.bounds[0], family.bounds[1])
    minimizer = tuple(float(family.minimizer_value) for _ in range(input_dim))
    return ScalarBenchmark(
        name=f"{family.name}_d{input_dim}",
        family=family.name,
        input_dim=input_dim,
        bounds=bounds,
        output_name="response",
        description=family.description,
        evaluator=family.evaluator,
        known_optimum=family.optimum,
        known_minimizer=minimizer,
    )


def _make_low_fidelity_evaluator(spec: ScalarBenchmark) -> ArrayFn:
    """
    Build a correlated low-fidelity evaluator from a scalar benchmark.

    Args:
        spec (ScalarBenchmark): High-fidelity scalar benchmark.

    Returns:
        ArrayFn: Low-fidelity evaluator. (N, D) -> (N, 1).
    """
    bounds = spec.bounds_array

    def low_fidelity(x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d_array(x, spec.input_dim, f"{spec.name}_low_fidelity")
        unit_x = _unit_coordinates(x_arr, bounds)
        warped_unit = np.clip(0.88 * unit_x + 0.06 + 0.03 * np.sin(2.0 * np.pi * unit_x), 0.0, 1.0)
        warped_x = bounds[:, 0] + warped_unit * (bounds[:, 1] - bounds[:, 0])
        hf = spec.evaluate(x_arr)[:, 0]
        warped = spec.evaluate(warped_x)[:, 0]
        trend = 0.05 * np.mean(unit_x - 0.5, axis=1)
        return (0.72 * hf + 0.28 * warped + trend).reshape(-1, 1)

    return low_fidelity


def make_multifidelity_benchmark(name: str, input_dim: int) -> MultiFidelityBenchmark:
    """
    Build a multi-fidelity benchmark with the requested dimension.

    Args:
        name (str): Scalable benchmark family name.
        input_dim (int): Input dimension.

    Returns:
        MultiFidelityBenchmark: Dimension-specific benchmark pair.
    """
    scalar = make_scalar_benchmark(name, input_dim)
    return MultiFidelityBenchmark(
        name=f"mf_{scalar.name}",
        family=scalar.family,
        input_dim=scalar.input_dim,
        bounds=scalar.bounds,
        output_name=scalar.output_name,
        description=f"{scalar.description} Low fidelity is a biased warped-coordinate approximation.",
        high_fidelity=scalar.evaluate,
        low_fidelity=_make_low_fidelity_evaluator(scalar),
    )


def make_multiobjective_benchmark(name: str, input_dim: int, num_objectives: int = 2) -> MultiObjectiveBenchmark:
    """
    Build a multi-objective benchmark with the requested dimension.

    Args:
        name (str): Scalable benchmark family name.
        input_dim (int): Input dimension.
        num_objectives (int): Number of objectives.

    Returns:
        MultiObjectiveBenchmark: Dimension-specific multi-objective benchmark.
    """
    key = name.lower()
    if key != "dtlz2":
        raise KeyError(f"Unknown multi-objective benchmark family: '{name}'.")

    def evaluator(x: np.ndarray) -> np.ndarray:
        return dtlz2(x, num_objectives=num_objectives)

    return MultiObjectiveBenchmark(
        name=f"dtlz2_d{input_dim}_m{num_objectives}",
        family="dtlz2",
        input_dim=input_dim,
        bounds=_repeat_bounds(input_dim, 0.0, 1.0),
        output_names=tuple(f"f{idx + 1}" for idx in range(num_objectives)),
        description="Scalable DTLZ2 benchmark with a spherical Pareto front.",
        evaluator=evaluator,
    )


def get_scalar_benchmark(name: str, input_dim: Optional[int] = None) -> ScalarBenchmark:
    """
    Fetch a scalar benchmark by family name and optional dimension.

    Args:
        name (str): Benchmark family or ``family_dD`` name.
        input_dim (Optional[int]): Requested input dimension.

    Returns:
        ScalarBenchmark: Requested scalar benchmark.
    """
    family, parsed_dim = _parse_dimension_name(name)
    dim = input_dim if input_dim is not None else parsed_dim
    if dim is None:
        dim = SCALAR_FAMILY_REGISTRY[family].default_dim
    return make_scalar_benchmark(family, dim)


def get_multifidelity_benchmark(name: str, input_dim: Optional[int] = None) -> MultiFidelityBenchmark:
    """
    Fetch a multi-fidelity benchmark by family name and optional dimension.

    Args:
        name (str): Benchmark family or ``family_dD`` name.
        input_dim (Optional[int]): Requested input dimension.

    Returns:
        MultiFidelityBenchmark: Requested multi-fidelity benchmark.
    """
    key = name.lower()
    if key.startswith("mf_"):
        key = key[3:]
    family, parsed_dim = _parse_dimension_name(key)
    dim = input_dim if input_dim is not None else parsed_dim
    if dim is None:
        dim = SCALAR_FAMILY_REGISTRY[family].default_dim
    return make_multifidelity_benchmark(family, dim)


def get_multiobjective_benchmark(
    name: str,
    input_dim: Optional[int] = None,
    num_objectives: int = 2,
) -> MultiObjectiveBenchmark:
    """
    Fetch a multi-objective benchmark by family name and optional dimension.

    Args:
        name (str): Benchmark family or ``family_dD`` name.
        input_dim (Optional[int]): Requested input dimension.
        num_objectives (int): Number of objectives.

    Returns:
        MultiObjectiveBenchmark: Requested multi-objective benchmark.
    """
    family, parsed_dim = _parse_dimension_name(name)
    dim = input_dim if input_dim is not None else parsed_dim
    if dim is None:
        dim = 20
    return make_multiobjective_benchmark(family, dim, num_objectives=num_objectives)


SCALAR_BENCHMARKS: Dict[str, ScalarBenchmark] = {
    name: make_scalar_benchmark(name, family.default_dim) for name, family in SCALAR_FAMILY_REGISTRY.items()
}
MULTI_FIDELITY_BENCHMARKS: Dict[str, MultiFidelityBenchmark] = {
    name: make_multifidelity_benchmark(name, family.default_dim)
    for name, family in SCALAR_FAMILY_REGISTRY.items()
    if name in {"sobol_g", "ackley", "rosenbrock"}
}
MULTI_OBJECTIVE_BENCHMARKS: Dict[str, MultiObjectiveBenchmark] = {
    "dtlz2": make_multiobjective_benchmark("dtlz2", 20, num_objectives=2)
}


from models.classical.krg import KRG
from models.classical.prs import PRS
from models.classical.rbf import RBF
from models.classical.svr import SVR
from models.ensemble.aes_msi import AESMSI
from models.ensemble.t_ahs import TAHS
from models.multi_fidelity.cca_mfs import CCAMFS
from models.multi_fidelity.mfs_mls import MFSMLS
from models.multi_fidelity.mmfs import MMFS
from models.optimization.dragonfly import dragonfly_optimize
from models.optimization.miga import multi_island_genetic_optimize
from sampling.diso_infill import DISOInfill
from sampling.doe import lhs_design
from sampling.mf_infill import MultiFidelityInfill
from sampling.mo_infill import MultiObjectiveInfill
from utils.hue_logger import hue, logger
from utils.seeder import seed_everything


ALGORITHM_ORDER = ["TAHS", "AESMSI", "MFSMLS", "MMFS", "CCAMFS", "DISO", "MICO", "MOBO", "MIGA", "CFARSSDA"]
CLASSICAL_ORDER = ["PRS", "RBF", "KRG", "SVR"]

SCALAR_FAMILIES = ["sobol_g", "ackley", "rosenbrock"]
MULTIFIDELITY_FAMILIES = ["sobol_g", "ackley"]
OPTIMIZATION_FAMILIES = ["ackley", "rastrigin", "rosenbrock"]

DIMENSION_SWEEP = [5, 10, 20, 50, 100]
FIXED_SAMPLE_DIM = 50
SCALAR_SAMPLE_FACTORS = [1.0, 2.0, 4.0, 6.0]
MULTIFIDELITY_SAMPLE_FACTORS = [1.2, 2.0, 3.0]
ACTIVE_DIMS = [5, 20, 50]

BENCHMARK_SOURCES = [
    {
        "name": "SFU Virtual Library: Optimization Test Functions",
        "url": "https://www.sfu.ca/~ssurjano/optimization.html",
        "usage": "Function families and landscape categories for Ackley, Rastrigin, Rosenbrock, Griewank, and related tests.",
    },
    {
        "name": "SFU Virtual Library: Sobol-G Function",
        "url": "https://www.sfu.ca/~ssurjano/gfunc.html",
        "usage": "Scalable unit-hypercube sensitivity benchmark used here for smooth effective-dimension tests.",
    },
    {
        "name": "pymoo DTLZ documentation",
        "url": "https://pymoo.org/problems/many/dtlz.html",
        "usage": "DTLZ2 scalable multi-objective formulation used here with two objectives.",
    },
]


# ============================================================
# Core Utilities
# ============================================================

def scale_to_bounds(x_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Scale unit-hypercube samples to physical bounds.

    Args:
        x_norm (np.ndarray): Normalized samples. (N, D).
        bounds (np.ndarray): Box bounds. (D, 2).

    Returns:
        np.ndarray: Scaled samples. (N, D).
    """
    return bounds[:, 0] + x_norm * (bounds[:, 1] - bounds[:, 0])


def sample_lhs(bounds: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Generate a latin hypercube inside a bounded design space.

    Args:
        bounds (np.ndarray): Box bounds. (D, 2).
        num_samples (int): Number of samples.

    Returns:
        np.ndarray: Scaled samples. (N, D).
    """
    x_norm = lhs_design(num_samples, bounds.shape[0])
    return scale_to_bounds(x_norm, bounds)


def reset_random_state(seed: int) -> None:
    """
    Reset Python and NumPy random states.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def predict_mean(model: Any, x: np.ndarray) -> np.ndarray:
    """
    Return the predictive mean from a surrogate model.

    Args:
        model (Any): Surrogate model.
        x (np.ndarray): Query points. (N, D).

    Returns:
        np.ndarray: Predictive mean. (N, C).
    """
    prediction = model.predict(x)
    return prediction[0] if isinstance(prediction, tuple) else prediction


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    """
    Compute the sum-based accuracy score.

    Args:
        y_true (np.ndarray): Ground truth. (N, C).
        y_pred (np.ndarray): Prediction. (N, C).
        eps (float): Stability epsilon.

    Returns:
        float: Accuracy in percent.
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true)) + eps
    return float((1.0 - numerator / denominator) * 100.0)


def evaluate_r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    """
    Compute the aggregated coefficient of determination.

    Args:
        y_true (np.ndarray): Ground truth. (N, C).
        y_pred (np.ndarray): Prediction. (N, C).
        eps (float): Stability epsilon.

    Returns:
        float: Aggregated R2 score.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / (ss_tot + eps))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> Dict[str, float]:
    """
    Compute accuracy and R2 metrics.

    Args:
        y_true (np.ndarray): Ground truth. (N, C).
        y_pred (np.ndarray): Prediction. (N, C).
        eps (float): Stability epsilon.

    Returns:
        Dict[str, float]: Metric dictionary.
    """
    return {"accuracy": evaluate_accuracy(y_true, y_pred, eps=eps), "r2": evaluate_r2(y_true, y_pred, eps=eps)}


def compute_relative_gain(before: float, after: float, eps: float) -> float:
    """
    Compute the relative gain from a baseline score.

    Args:
        before (float): Baseline score.
        after (float): Updated score.
        eps (float): Stability epsilon.

    Returns:
        float: Relative gain.
    """
    return float((after - before) / max(abs(before), eps))


def fit_krg(x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> KRG:
    """
    Fit a Kriging surrogate with shared CLI hyperparameters.

    Args:
        x_train (np.ndarray): Training inputs. (N, D).
        y_train (np.ndarray): Training targets. (N, C).
        args (argparse.Namespace): Parsed arguments.

    Returns:
        KRG: Trained Kriging model.
    """
    model = KRG(**args.krg_params)
    model.fit(x_train, y_train)
    return model


def to_serializable(value: Any) -> Any:
    """
    Convert NumPy-heavy objects into JSON-safe Python objects.

    Args:
        value (Any): Arbitrary value.

    Returns:
        Any: JSON-safe value.
    """
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def seed_run(seed: int) -> None:
    """
    Set all random states used by the benchmark runner.

    Args:
        seed (int): Random seed.
    """
    seed_everything(seed)
    reset_random_state(seed)


def timed_call(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute a benchmark call and attach elapsed wall time.

    Args:
        fn (Callable[[], Dict[str, Any]]): Callable benchmark item.

    Returns:
        Dict[str, Any]: Benchmark result.
    """
    start = time.perf_counter()
    try:
        result = fn()
        result["status"] = "completed"
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def average_completed(entries: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    """
    Average completed seed-level entries.

    Args:
        entries (List[Dict[str, Any]]): Seed-level records.
        keys (List[str]): Numeric keys to average.

    Returns:
        Dict[str, Any]: Averaged record.
    """
    completed = [entry for entry in entries if entry["status"] == "completed"]
    if not completed:
        first = entries[0]
        return {
            "status": first["status"],
            "reason": first.get("reason"),
            "error": first.get("error"),
            "runs": entries,
        }

    item = {
        "status": "completed" if len(completed) == len(entries) else "partial",
        "num_completed": len(completed),
        "num_runs": len(entries),
        "elapsed_seconds": float(mean(entry["elapsed_seconds"] for entry in entries)),
        "runs": entries,
    }
    for key in keys:
        item[key] = float(mean(entry[key] for entry in completed))
    return item


def num_test_samples(input_dim: int, args: argparse.Namespace) -> int:
    """
    Compute a bounded test-set size for one dimension.

    Args:
        input_dim (int): Input dimension.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Number of test samples.
    """
    return int(min(max(args.test_factor * input_dim, args.min_test), args.max_test))


# ============================================================
# Model Builders And Limits
# ============================================================

def build_single_model(name: str, args: argparse.Namespace) -> Any:
    """
    Build a single-fidelity surrogate model.

    Args:
        name (str): Model name.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Any: Surrogate model.
    """
    builders: Dict[str, Callable[[], Any]] = {
        "PRS": lambda: PRS(**args.prs_params),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(**args.svr_params),
        "TAHS": lambda: TAHS(
            threshold=args.ensemble_threshold,
            prs_params=args.prs_params,
            krg_params=args.krg_params,
            svr_params=args.svr_params,
        ),
        "AESMSI": lambda: AESMSI(
            threshold=args.ensemble_threshold,
            prs_params=args.prs_params,
            krg_params=args.krg_params,
            svr_params=args.svr_params,
        ),
    }
    return builders[name]()


def build_multifidelity_model(name: str, args: argparse.Namespace) -> Any:
    """
    Build a multi-fidelity surrogate model.

    Args:
        name (str): Model name.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Any: Multi-fidelity surrogate model.
    """
    builders: Dict[str, Callable[[], Any]] = {
        "MFSMLS": lambda: MFSMLS(poly_degree=1, neighbor_factor=args.mfs_mls_neighbor_factor, ridge=args.mfs_mls_ridge),
        "MMFS": lambda: MMFS(),
        "CCAMFS": lambda: CCAMFS(),
    }
    return builders[name]()


def single_model_limit_reason(name: str, input_dim: int, num_train: int, args: argparse.Namespace) -> Optional[str]:
    """
    Return a skip reason when a single-fidelity model exceeds the configured scale limit.

    Args:
        name (str): Model name.
        input_dim (int): Input dimension.
        num_train (int): Number of training samples.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Optional[str]: Skip reason.
    """
    if name == "KRG" and (input_dim > args.max_krg_dim or num_train > args.max_krg_train):
        return f"KRG anisotropic likelihood fit is capped at D<={args.max_krg_dim}, N<={args.max_krg_train}."
    if name == "SVR" and (input_dim > args.max_svr_dim or num_train > args.max_svr_train):
        return f"SVR dual SLSQP fit is capped at D<={args.max_svr_dim}, N<={args.max_svr_train}."
    if name in {"TAHS", "AESMSI"} and (
        input_dim > args.max_ensemble_dim or num_train > args.max_ensemble_train
    ):
        return (
            f"{name} exact LOO screening is capped at D<={args.max_ensemble_dim}, "
            f"N<={args.max_ensemble_train}."
        )
    return None


def multifidelity_limit_reason(name: str, input_dim: int, num_lf: int, num_hf: int, args: argparse.Namespace) -> Optional[str]:
    """
    Return a skip reason when a multi-fidelity model exceeds the configured scale limit.

    Args:
        name (str): Model name.
        input_dim (int): Input dimension.
        num_lf (int): Number of low-fidelity samples.
        num_hf (int): Number of high-fidelity samples.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Optional[str]: Skip reason.
    """
    if name == "MFSMLS" and (input_dim > args.max_mfsmls_dim or num_hf > args.max_mfsmls_hf):
        return f"MFSMLS MLS solve is capped at D<={args.max_mfsmls_dim}, N_HF<={args.max_mfsmls_hf}."
    if name == "MMFS" and (input_dim > args.max_mmfs_dim or num_hf > args.max_mmfs_hf):
        return f"MMFS LOOCV correction is capped at D<={args.max_mmfs_dim}, N_HF<={args.max_mmfs_hf}."
    if name == "CCAMFS" and (
        input_dim > args.max_ccamfs_dim or num_hf > args.max_ccamfs_hf or num_lf > args.max_ccamfs_lf
    ):
        return (
            f"CCAMFS CCA/RBF correction is capped at D<={args.max_ccamfs_dim}, "
            f"N_HF<={args.max_ccamfs_hf}, N_LF<={args.max_ccamfs_lf}."
        )
    return None


# ============================================================
# Case Builders
# ============================================================

def scalar_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build scalar approximation sweep cases.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Scalar benchmark cases.
    """
    cases: List[Dict[str, Any]] = []
    for family in SCALAR_FAMILIES:
        for input_dim in DIMENSION_SWEEP:
            if input_dim > args.max_dim:
                continue
            cases.append(
                {
                    "sweep": "dimension",
                    "family": family,
                    "input_dim": input_dim,
                    "num_train": int(np.ceil(args.dimension_sample_factor * input_dim)),
                    "sample_factor": float(args.dimension_sample_factor),
                }
            )

        if FIXED_SAMPLE_DIM <= args.max_dim:
            for factor in SCALAR_SAMPLE_FACTORS:
                cases.append(
                    {
                        "sweep": "sample",
                        "family": family,
                        "input_dim": FIXED_SAMPLE_DIM,
                        "num_train": int(np.ceil(factor * FIXED_SAMPLE_DIM)),
                        "sample_factor": float(factor),
                    }
                )
    return cases


def multifidelity_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build multi-fidelity approximation sweep cases.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Multi-fidelity benchmark cases.
    """
    cases: List[Dict[str, Any]] = []
    for family in MULTIFIDELITY_FAMILIES:
        for input_dim in DIMENSION_SWEEP:
            if input_dim > args.max_dim:
                continue
            num_hf = max(input_dim + 3, int(np.ceil(args.multifidelity_dimension_hf_factor * input_dim)))
            cases.append(
                {
                    "sweep": "dimension",
                    "family": family,
                    "input_dim": input_dim,
                    "num_hf": num_hf,
                    "num_lf": int(np.ceil(args.lf_to_hf_ratio * num_hf)),
                    "hf_sample_factor": float(num_hf / input_dim),
                }
            )

        if FIXED_SAMPLE_DIM <= args.max_dim:
            for factor in MULTIFIDELITY_SAMPLE_FACTORS:
                num_hf = max(FIXED_SAMPLE_DIM + 3, int(np.ceil(factor * FIXED_SAMPLE_DIM)))
                cases.append(
                    {
                        "sweep": "sample",
                        "family": family,
                        "input_dim": FIXED_SAMPLE_DIM,
                        "num_hf": num_hf,
                        "num_lf": int(np.ceil(args.lf_to_hf_ratio * num_hf)),
                        "hf_sample_factor": float(num_hf / FIXED_SAMPLE_DIM),
                    }
                )
    return cases


def active_dims(args: argparse.Namespace) -> List[int]:
    """
    Return active-learning dimensions under the current maximum dimension.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[int]: Active-learning dimensions.
    """
    return [input_dim for input_dim in ACTIVE_DIMS if input_dim <= args.max_dim]


# ============================================================
# Scalar Approximation Section
# ============================================================

def run_one_scalar_algorithm(
    name: str,
    spec: ScalarBenchmark,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Fit one single-fidelity algorithm and evaluate its predictive accuracy.

    Args:
        name (str): Algorithm name.
        spec (ScalarBenchmark): Benchmark specification.
        x_train (np.ndarray): Training inputs. (N, D).
        y_train (np.ndarray): Training targets. (N, 1).
        x_test (np.ndarray): Test inputs. (N_TEST, D).
        y_test (np.ndarray): Test targets. (N_TEST, 1).
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level result.
    """
    model = build_single_model(name, args)
    model.fit(x_train, y_train)
    metrics = evaluate_metrics(y_test, predict_mean(model, x_test), eps=args.metric_eps)
    return {
        **metrics,
        "success": metrics["accuracy"] >= args.success_accuracy,
        "strong": metrics["accuracy"] >= args.strong_accuracy,
        "benchmark": spec.name,
    }


def run_scalar_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one scalar approximation case across all selected single-fidelity algorithms.

    Args:
        case (Dict[str, Any]): Scalar case definition.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Case result.
    """
    spec = make_scalar_benchmark(case["family"], case["input_dim"])
    bounds = spec.bounds_array
    num_test = num_test_samples(spec.input_dim, args)
    algorithms = CLASSICAL_ORDER + [name for name in ("TAHS", "AESMSI") if name in args.demos]

    case_result = {
        "case_id": f"{case['sweep']}:{spec.family}:d{spec.input_dim}:n{case['num_train']}",
        "sweep": case["sweep"],
        "family": spec.family,
        "benchmark": spec.name,
        "input_dim": spec.input_dim,
        "num_train": case["num_train"],
        "num_test": num_test,
        "sample_factor": case["sample_factor"],
        "algorithms": [],
    }

    logger.info(
        f"  scalar {case_result['case_id']} | N={case['num_train']} | "
        f"N/D={case['sample_factor']:.2f}"
    )

    for name in algorithms:
        reason = single_model_limit_reason(name, spec.input_dim, case["num_train"], args)
        if reason is not None:
            case_result["algorithms"].append(
                {"name": name, "status": "skipped", "skip_kind": "budget_limit", "reason": reason, "runs": []}
            )
            continue

        entries = []
        for seed in args.seeds:
            seed_run(seed)
            x_train = sample_lhs(bounds, case["num_train"])
            x_test = sample_lhs(bounds, num_test)
            y_train = spec.evaluate(x_train)
            y_test = spec.evaluate(x_test)
            entries.append(
                timed_call(lambda: run_one_scalar_algorithm(name, spec, x_train, y_train, x_test, y_test, args))
            )

        item = average_completed(entries, ["accuracy", "r2"])
        item["name"] = name
        if item["status"] in {"completed", "partial"}:
            item["success"] = item["accuracy"] >= args.success_accuracy
            item["strong"] = item["accuracy"] >= args.strong_accuracy
        case_result["algorithms"].append(item)

    return case_result


def run_scalar_section(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run the scalar approximation boundary sweep.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Scalar section records.
    """
    logger.info(f"{hue.b}Scalar Approximation Boundary Sweep{hue.q}")
    return [run_scalar_case(case, args) for case in scalar_cases(args)]


# ============================================================
# Multi-Fidelity Section
# ============================================================

def run_one_multifidelity_algorithm(
    name: str,
    spec: MultiFidelityBenchmark,
    x_lf: np.ndarray,
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    y_hf: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Fit one multi-fidelity algorithm and evaluate its predictive accuracy.

    Args:
        name (str): Algorithm name.
        spec (MultiFidelityBenchmark): Benchmark specification.
        x_lf (np.ndarray): LF inputs. (N_L, D).
        y_lf (np.ndarray): LF targets. (N_L, 1).
        x_hf (np.ndarray): HF inputs. (N_H, D).
        y_hf (np.ndarray): HF targets. (N_H, 1).
        x_test (np.ndarray): Test inputs. (N_TEST, D).
        y_test (np.ndarray): Test targets. (N_TEST, 1).
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level result.
    """
    model = build_multifidelity_model(name, args)
    model.fit(x_lf, y_lf, x_hf, y_hf)
    metrics = evaluate_metrics(y_test, predict_mean(model, x_test), eps=args.metric_eps)
    return {
        **metrics,
        "success": metrics["accuracy"] >= args.success_accuracy,
        "strong": metrics["accuracy"] >= args.strong_accuracy,
        "benchmark": spec.name,
    }


def run_multifidelity_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one multi-fidelity approximation case across all selected algorithms.

    Args:
        case (Dict[str, Any]): Multi-fidelity case definition.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Case result.
    """
    spec = make_multifidelity_benchmark(case["family"], case["input_dim"])
    bounds = spec.bounds_array
    num_test = num_test_samples(spec.input_dim, args)
    algorithms = [name for name in ("MFSMLS", "MMFS", "CCAMFS") if name in args.demos]

    case_result = {
        "case_id": f"{case['sweep']}:mf_{spec.family}:d{spec.input_dim}:hf{case['num_hf']}",
        "sweep": case["sweep"],
        "family": spec.family,
        "benchmark": spec.name,
        "input_dim": spec.input_dim,
        "num_lf": case["num_lf"],
        "num_hf": case["num_hf"],
        "num_test": num_test,
        "hf_sample_factor": case["hf_sample_factor"],
        "algorithms": [],
    }

    logger.info(
        f"  multifidelity {case_result['case_id']} | LF={case['num_lf']} | "
        f"HF/D={case['hf_sample_factor']:.2f}"
    )

    for name in algorithms:
        reason = multifidelity_limit_reason(name, spec.input_dim, case["num_lf"], case["num_hf"], args)
        if reason is not None:
            case_result["algorithms"].append(
                {"name": name, "status": "skipped", "skip_kind": "budget_limit", "reason": reason, "runs": []}
            )
            continue

        entries = []
        for seed in args.seeds:
            seed_run(seed)
            x_lf = sample_lhs(bounds, case["num_lf"])
            x_hf = sample_lhs(bounds, case["num_hf"])
            x_test = sample_lhs(bounds, num_test)
            y_lf = spec.evaluate_low_fidelity(x_lf)
            y_hf = spec.evaluate_high_fidelity(x_hf)
            y_test = spec.evaluate_high_fidelity(x_test)
            entries.append(
                timed_call(
                    lambda: run_one_multifidelity_algorithm(
                        name,
                        spec,
                        x_lf,
                        y_lf,
                        x_hf,
                        y_hf,
                        x_test,
                        y_test,
                        args,
                    )
                )
            )

        item = average_completed(entries, ["accuracy", "r2"])
        item["name"] = name
        if item["status"] in {"completed", "partial"}:
            item["success"] = item["accuracy"] >= args.success_accuracy
            item["strong"] = item["accuracy"] >= args.strong_accuracy
        case_result["algorithms"].append(item)

    return case_result


def run_multifidelity_section(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run the multi-fidelity approximation boundary sweep.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Multi-fidelity section records.
    """
    logger.info(f"{hue.b}Multi-Fidelity Boundary Sweep{hue.q}")
    return [run_multifidelity_case(case, args) for case in multifidelity_cases(args)]


# ============================================================
# Active Learning Section
# ============================================================

def compute_pareto_size(y_values: np.ndarray) -> int:
    """
    Compute the number of non-dominated points for a minimization problem.

    Args:
        y_values (np.ndarray): Objective values. (N, M).

    Returns:
        int: Number of Pareto points.
    """
    y_i = y_values[:, np.newaxis, :]
    y_j = y_values[np.newaxis, :, :]
    diff = y_j - y_i
    dominated = np.all(diff <= 0.0, axis=2) & np.any(diff < 0.0, axis=2)
    np.fill_diagonal(dominated, False)
    return int(np.sum(~np.any(dominated, axis=1)))


def run_diso_seed(input_dim: int, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one DISO active-learning seed.

    Args:
        input_dim (int): Input dimension.
        seed (int): Random seed.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level result.
    """
    seed_run(seed)
    spec = make_scalar_benchmark(args.active_family, input_dim)
    bounds = spec.bounds_array
    num_initial = max(args.active_initial_min, input_dim + args.active_initial_offset)
    num_test = num_test_samples(input_dim, args)

    x_current = sample_lhs(bounds, num_initial)
    x_test = sample_lhs(bounds, num_test)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)

    history_best: List[float] = []
    for _ in range(args.active_infill):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = DISOInfill(
            model=model_iter,
            bounds=bounds,
            x_train=x_current,
            y_train=y_current,
            criterion="ei",
            target_idx=0,
            alpha=args.diso_alpha,
            min_distance=args.diso_min_distance,
            distance_scale=args.diso_distance_scale,
        )
        x_new = strategy.propose()
        y_new = spec.evaluate(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history_best.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    gain = compute_relative_gain(before["accuracy"], after["accuracy"], eps=args.metric_eps)
    return {
        "before_accuracy": before["accuracy"],
        "after_accuracy": after["accuracy"],
        "before_r2": before["r2"],
        "after_r2": after["r2"],
        "accuracy_gain": gain,
        "history_best": history_best,
        "success": after["accuracy"] >= args.success_accuracy,
    }


def run_mico_seed(input_dim: int, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one MICO multi-fidelity active-learning seed.

    Args:
        input_dim (int): Input dimension.
        seed (int): Random seed.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level result.
    """
    seed_run(seed)
    spec = make_multifidelity_benchmark(args.active_family, input_dim)
    bounds = spec.bounds_array
    num_initial = max(args.active_initial_min, input_dim + args.active_initial_offset)
    num_lf = int(np.ceil(args.active_lf_factor * num_initial))
    num_test = num_test_samples(input_dim, args)

    x_lf = sample_lhs(bounds, num_lf)
    x_current = sample_lhs(bounds, num_initial)
    x_test = sample_lhs(bounds, num_test)
    y_lf = spec.evaluate_low_fidelity(x_lf)
    y_current = spec.evaluate_high_fidelity(x_current)
    y_test = spec.evaluate_high_fidelity(x_test)

    model_before = fit_krg(x_current, y_current, args)
    before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)

    history_best: List[float] = []
    for _ in range(args.active_infill):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiFidelityInfill(
            model=model_iter,
            x_hf=x_current,
            y_hf=y_current,
            x_lf=x_lf,
            y_lf=y_lf,
            target_idx=0,
            ratio=args.mico_ratio,
        )
        x_new = strategy.propose()
        y_new = spec.evaluate_high_fidelity(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history_best.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    gain = compute_relative_gain(before["accuracy"], after["accuracy"], eps=args.metric_eps)
    return {
        "before_accuracy": before["accuracy"],
        "after_accuracy": after["accuracy"],
        "before_r2": before["r2"],
        "after_r2": after["r2"],
        "accuracy_gain": gain,
        "history_best": history_best,
        "success": after["accuracy"] >= args.success_accuracy,
    }


def run_mobo_seed(input_dim: int, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one MOBO multi-objective active-learning seed.

    Args:
        input_dim (int): Input dimension.
        seed (int): Random seed.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level result.
    """
    seed_run(seed)
    spec = make_multiobjective_benchmark("dtlz2", input_dim, num_objectives=2)
    bounds = spec.bounds_array
    num_initial = max(args.active_initial_min, input_dim + args.active_initial_offset)
    num_test = num_test_samples(input_dim, args)

    x_current = sample_lhs(bounds, num_initial)
    x_test = sample_lhs(bounds, num_test)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)
    pareto_before = compute_pareto_size(y_current)

    for _ in range(args.active_infill):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiObjectiveInfill(
            model=model_iter,
            bounds=bounds,
            y_train=y_current,
            obj_idxs=[0, 1],
            constraint_idxs=None,
            constraint_ubs=None,
            num_samples=args.mobo_num_samples,
            num_candidates=args.mobo_num_candidates,
            num_restarts=args.mobo_num_restarts,
            beta=args.mobo_beta,
        )
        x_new = strategy.propose()
        y_new = spec.evaluate(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])

    model_after = fit_krg(x_current, y_current, args)
    after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    pareto_after = compute_pareto_size(y_current)
    gain = compute_relative_gain(before["accuracy"], after["accuracy"], eps=args.metric_eps)
    return {
        "before_accuracy": before["accuracy"],
        "after_accuracy": after["accuracy"],
        "before_r2": before["r2"],
        "after_r2": after["r2"],
        "accuracy_gain": gain,
        "pareto_size_before": pareto_before,
        "pareto_size_after": pareto_after,
        "success": after["accuracy"] >= args.success_accuracy,
    }


def run_active_algorithm(name: str, input_dim: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one active-learning algorithm at a fixed dimension.

    Args:
        name (str): Algorithm name.
        input_dim (int): Input dimension.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Active-learning result.
    """
    seed_fns = {"DISO": run_diso_seed, "MICO": run_mico_seed, "MOBO": run_mobo_seed}
    entries = [timed_call(lambda seed=seed: seed_fns[name](input_dim, seed, args)) for seed in args.seeds]
    item = average_completed(
        entries,
        ["before_accuracy", "after_accuracy", "before_r2", "after_r2", "accuracy_gain"],
    )
    item["name"] = name
    if item["status"] in {"completed", "partial"}:
        item["success"] = item["after_accuracy"] >= args.success_accuracy
    return item


def run_active_section(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run the active-learning boundary sweep.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Active-learning section records.
    """
    logger.info(f"{hue.b}Active Learning Boundary Sweep{hue.q}")
    results = []
    algorithms = [name for name in ("DISO", "MICO", "MOBO") if name in args.demos]
    for input_dim in active_dims(args):
        num_initial = max(args.active_initial_min, input_dim + args.active_initial_offset)
        case_result = {
            "case_id": f"active:{args.active_family}:d{input_dim}:init{num_initial}",
            "family": args.active_family,
            "input_dim": input_dim,
            "num_initial": num_initial,
            "num_infill": args.active_infill,
            "algorithms": [],
        }
        logger.info(f"  active d{input_dim} | initial={num_initial} | infill={args.active_infill}")
        for name in algorithms:
            if input_dim > args.max_active_dim:
                case_result["algorithms"].append(
                    {
                        "name": name,
                        "status": "skipped",
                        "skip_kind": "budget_limit",
                        "reason": f"{name} uses repeated KRG fits and is capped at D<={args.max_active_dim}.",
                        "runs": [],
                    }
                )
                continue
            case_result["algorithms"].append(run_active_algorithm(name, input_dim, args))
        results.append(case_result)
    return results


# ============================================================
# Optimization Section
# ============================================================

def objective_from_spec(spec: ScalarBenchmark) -> Callable[[np.ndarray], float]:
    """
    Build a scalar optimizer objective from a benchmark specification.

    Args:
        spec (ScalarBenchmark): Scalar benchmark.

    Returns:
        Callable[[np.ndarray], float]: Scalar objective.
    """
    return lambda x_vec: float(spec.evaluate(np.asarray(x_vec, dtype=np.float64).reshape(1, -1))[0, 0])


def run_one_optimizer(name: str, spec: ScalarBenchmark, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one global optimizer directly on a scalable benchmark.

    Args:
        name (str): Optimizer name.
        spec (ScalarBenchmark): Benchmark specification.
        seed (int): Random seed.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Seed-level optimizer result.
    """
    seed_run(seed)
    objective = objective_from_spec(spec)
    bounds = [tuple(bound) for bound in spec.bounds_array]

    if name == "MIGA":
        result = multi_island_genetic_optimize(
            func=objective,
            bounds=bounds,
            tol=args.opt_tol,
            seed=seed,
            multi_objective=False,
            **args.miga_params,
        )
    else:
        result = dragonfly_optimize(
            func=objective,
            bounds=bounds,
            tol=args.opt_tol,
            seed=seed,
            multi_objective=False,
            **args.df_params,
        )

    verified = float(spec.evaluate(result.x.reshape(1, -1))[0, 0])
    return {
        "predicted_objective": float(result.fun),
        "verified_objective": verified,
        "known_optimum": spec.known_optimum,
        "optimality_gap": verified - float(spec.known_optimum or 0.0),
        "success": verified <= args.optimization_success,
        "x_best": result.x,
    }


def run_optimization_case(family: str, input_dim: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one direct-optimization case across selected optimizers.

    Args:
        family (str): Scalar benchmark family.
        input_dim (int): Input dimension.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Optimization case result.
    """
    spec = make_scalar_benchmark(family, input_dim)
    algorithms = [name for name in ("MIGA", "CFARSSDA") if name in args.demos]
    case_result = {
        "case_id": f"optimization:{family}:d{input_dim}",
        "family": family,
        "benchmark": spec.name,
        "input_dim": input_dim,
        "known_optimum": spec.known_optimum,
        "algorithms": [],
    }
    logger.info(f"  optimization {case_result['case_id']}")

    for name in algorithms:
        entries = [timed_call(lambda seed=seed: run_one_optimizer(name, spec, seed, args)) for seed in args.seeds]
        item = average_completed(entries, ["predicted_objective", "verified_objective", "optimality_gap"])
        item["name"] = name
        if item["status"] in {"completed", "partial"}:
            item["success"] = item["verified_objective"] <= args.optimization_success
        case_result["algorithms"].append(item)

    return case_result


def run_optimization_section(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Run the direct-optimization boundary sweep.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Optimization section records.
    """
    logger.info(f"{hue.b}Optimization Boundary Sweep{hue.q}")
    results = []
    for family in OPTIMIZATION_FAMILIES:
        for input_dim in DIMENSION_SWEEP:
            if input_dim <= args.max_dim:
                results.append(run_optimization_case(family, input_dim, args))
    return results


# ============================================================
# Summary And Orchestration
# ============================================================

def iter_algorithm_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten algorithm-level records across all benchmark sections.

    Args:
        payload (Dict[str, Any]): Full benchmark payload.

    Returns:
        List[Dict[str, Any]]: Flat algorithm records.
    """
    flat: List[Dict[str, Any]] = []
    for section_name in ("scalar", "multifidelity", "active_learning", "optimization"):
        for case in payload["sections"][section_name]:
            for item in case["algorithms"]:
                flat.append({**item, "section": section_name, "case_id": case["case_id"], "input_dim": case["input_dim"]})
    return flat


def build_capability_summary(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build algorithm-level capability summaries.

    Args:
        payload (Dict[str, Any]): Full benchmark payload.

    Returns:
        Dict[str, Dict[str, Any]]: Capability summary by algorithm.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for item in iter_algorithm_items(payload):
        name = item["name"]
        current = summary.setdefault(
            name,
            {
                "num_completed": 0,
                "num_skipped": 0,
                "num_failed": 0,
                "max_completed_dim": None,
                "max_success_dim": None,
                "best_accuracy": None,
                "best_after_accuracy": None,
                "best_verified_objective": None,
                "boundary_case": None,
            },
        )
        status = item["status"]
        if status == "skipped":
            current["num_skipped"] += 1
            continue
        if status == "failed":
            current["num_failed"] += 1
            continue

        input_dim = item["input_dim"]
        current["num_completed"] += 1
        current["max_completed_dim"] = input_dim if current["max_completed_dim"] is None else max(current["max_completed_dim"], input_dim)
        if item.get("success"):
            current["max_success_dim"] = input_dim if current["max_success_dim"] is None else max(current["max_success_dim"], input_dim)
            current["boundary_case"] = item["case_id"]
        if "accuracy" in item:
            value = item["accuracy"]
            current["best_accuracy"] = value if current["best_accuracy"] is None else max(current["best_accuracy"], value)
        if "after_accuracy" in item:
            value = item["after_accuracy"]
            current["best_after_accuracy"] = value if current["best_after_accuracy"] is None else max(current["best_after_accuracy"], value)
        if "verified_objective" in item:
            value = item["verified_objective"]
            current["best_verified_objective"] = (
                value if current["best_verified_objective"] is None else min(current["best_verified_objective"], value)
            )
    return summary


def run_bench_suite(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the configured scalable benchmark suite.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Dict[str, Any]: Final benchmark payload.
    """
    payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": "scalable_boundary_v1",
        "seeds": args.seeds,
        "demos": args.demos,
        "classical_context": CLASSICAL_ORDER,
        "success_accuracy": args.success_accuracy,
        "strong_accuracy": args.strong_accuracy,
        "optimization_success": args.optimization_success,
        "sources": BENCHMARK_SOURCES,
        "settings": {
            "max_dim": args.max_dim,
            "dimension_sample_factor": args.dimension_sample_factor,
            "multifidelity_dimension_hf_factor": args.multifidelity_dimension_hf_factor,
            "lf_to_hf_ratio": args.lf_to_hf_ratio,
            "scalar_sample_factors": SCALAR_SAMPLE_FACTORS,
            "fixed_sample_dim": FIXED_SAMPLE_DIM,
            "prs_degree": args.prs_degree,
            "svr_kernel": args.svr_kernel,
            "active_infill": args.active_infill,
            "mobo_num_samples": args.mobo_num_samples,
            "mobo_num_candidates": args.mobo_num_candidates,
            "optimizer_maxiter": args.optimizer_maxiter,
            "optimizer_popsize": args.optimizer_popsize,
            "max_krg_dim": args.max_krg_dim,
            "max_krg_train": args.max_krg_train,
            "max_svr_dim": args.max_svr_dim,
            "max_svr_train": args.max_svr_train,
            "max_ensemble_dim": args.max_ensemble_dim,
            "max_ensemble_train": args.max_ensemble_train,
            "max_mfsmls_dim": args.max_mfsmls_dim,
            "max_mfsmls_hf": args.max_mfsmls_hf,
            "max_mmfs_dim": args.max_mmfs_dim,
            "max_mmfs_hf": args.max_mmfs_hf,
            "max_ccamfs_dim": args.max_ccamfs_dim,
            "max_ccamfs_hf": args.max_ccamfs_hf,
            "max_ccamfs_lf": args.max_ccamfs_lf,
            "max_active_dim": args.max_active_dim,
        },
        "sections": {
            "scalar": [],
            "multifidelity": [],
            "active_learning": [],
            "optimization": [],
        },
    }

    if any(name in args.demos for name in ("TAHS", "AESMSI")) or args.include_classical:
        payload["sections"]["scalar"] = run_scalar_section(args)
    if any(name in args.demos for name in ("MFSMLS", "MMFS", "CCAMFS")):
        payload["sections"]["multifidelity"] = run_multifidelity_section(args)
    if any(name in args.demos for name in ("DISO", "MICO", "MOBO")):
        payload["sections"]["active_learning"] = run_active_section(args)
    if any(name in args.demos for name in ("MIGA", "CFARSSDA")):
        payload["sections"]["optimization"] = run_optimization_section(args)

    payload["capability_summary"] = build_capability_summary(payload)
    return payload


def save_results(payload: Dict[str, Any]) -> str:
    """
    Save the benchmark payload as JSON.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Absolute save path.
    """
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2, ensure_ascii=False)
    return save_path


def encode_figure(fig: Any) -> str:
    """
    Encode a Matplotlib figure as a base64 PNG string.

    Args:
        fig (Any): Matplotlib figure.

    Returns:
        str: Base64-encoded PNG payload.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def find_algorithm(case: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """
    Find an algorithm record inside one benchmark case.

    Args:
        case (Dict[str, Any]): Benchmark case record.
        name (str): Algorithm name.

    Returns:
        Optional[Dict[str, Any]]: Algorithm record.
    """
    for item in case["algorithms"]:
        if item["name"] == name:
            return item
    return None


def completed_metric(case: Dict[str, Any], name: str, metric: str) -> Optional[float]:
    """
    Read a metric from a completed algorithm record.

    Args:
        case (Dict[str, Any]): Benchmark case record.
        name (str): Algorithm name.
        metric (str): Metric name.

    Returns:
        Optional[float]: Metric value.
    """
    item = find_algorithm(case, name)
    if item is None or item["status"] not in {"completed", "partial"}:
        return None
    return item.get(metric)


def plot_accuracy_by_dimension(payload: Dict[str, Any]) -> str:
    """
    Plot scalar surrogate accuracy versus input dimension.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Base64-encoded PNG payload.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = ["PRS", "RBF", "KRG", "SVR", "TAHS", "AESMSI"]
    families = ["sobol_g", "ackley", "rosenbrock"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    for ax, family in zip(axes, families):
        cases = [
            case for case in payload["sections"]["scalar"]
            if case["family"] == family and case["sweep"] == "dimension"
        ]
        cases = sorted(cases, key=lambda item: item["input_dim"])
        for algo in algorithms:
            xs, ys = [], []
            for case in cases:
                value = completed_metric(case, algo, "accuracy")
                if value is not None:
                    xs.append(case["input_dim"])
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=algo)
        ax.axhline(payload["success_accuracy"], color="0.55", linestyle="--", linewidth=1.0)
        ax.set_title(f"{family}: dimension sweep")
        ax.set_xlabel("Dimension")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)
    image = encode_figure(fig)
    plt.close(fig)
    return image


def plot_accuracy_by_sample_factor(payload: Dict[str, Any]) -> str:
    """
    Plot scalar surrogate accuracy versus sample factor at D=50.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Base64-encoded PNG payload.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = ["PRS", "RBF", "KRG", "SVR", "TAHS", "AESMSI"]
    families = ["sobol_g", "ackley", "rosenbrock"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    for ax, family in zip(axes, families):
        cases = [
            case for case in payload["sections"]["scalar"]
            if case["family"] == family and case["sweep"] == "sample"
        ]
        cases = sorted(cases, key=lambda item: item["sample_factor"])
        for algo in algorithms:
            xs, ys = [], []
            for case in cases:
                value = completed_metric(case, algo, "accuracy")
                if value is not None:
                    xs.append(case["sample_factor"])
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=algo)
        ax.axhline(payload["success_accuracy"], color="0.55", linestyle="--", linewidth=1.0)
        ax.set_title(f"{family}: D=50 sample sweep")
        ax.set_xlabel("N / D")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)
    image = encode_figure(fig)
    plt.close(fig)
    return image


def plot_multifidelity_accuracy(payload: Dict[str, Any]) -> str:
    """
    Plot multi-fidelity surrogate accuracy versus input dimension.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Base64-encoded PNG payload.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = ["MFSMLS", "MMFS", "CCAMFS"]
    families = ["sobol_g", "ackley"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, family in zip(axes, families):
        cases = [
            case for case in payload["sections"]["multifidelity"]
            if case["family"] == family and case["sweep"] == "dimension"
        ]
        cases = sorted(cases, key=lambda item: item["input_dim"])
        for algo in algorithms:
            xs, ys = [], []
            for case in cases:
                value = completed_metric(case, algo, "accuracy")
                if value is not None:
                    xs.append(case["input_dim"])
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=algo)
        ax.axhline(payload["success_accuracy"], color="0.55", linestyle="--", linewidth=1.0)
        ax.set_title(f"{family}: MF dimension sweep")
        ax.set_xlabel("Dimension")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)
    image = encode_figure(fig)
    plt.close(fig)
    return image


def plot_optimizer_objective(payload: Dict[str, Any]) -> str:
    """
    Plot verified optimizer objective values versus input dimension.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Base64-encoded PNG payload.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = ["MIGA", "CFARSSDA"]
    families = ["ackley", "rastrigin", "rosenbrock"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, family in zip(axes, families):
        cases = [case for case in payload["sections"]["optimization"] if case["family"] == family]
        cases = sorted(cases, key=lambda item: item["input_dim"])
        for algo in algorithms:
            xs, ys = [], []
            for case in cases:
                value = completed_metric(case, algo, "verified_objective")
                if value is not None:
                    xs.append(case["input_dim"])
                    ys.append(max(value, 1.0e-8))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=algo)
        ax.axhline(payload["optimization_success"], color="0.55", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{family}: direct optimization")
        ax.set_xlabel("Dimension")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Verified objective (log)")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)
    image = encode_figure(fig)
    plt.close(fig)
    return image


def format_value(value: Any) -> str:
    """
    Format a value for HTML tables.

    Args:
        value (Any): Raw value.

    Returns:
        str: Formatted value.
    """
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_capability_rows(payload: Dict[str, Any]) -> str:
    """
    Build HTML rows for the capability summary table.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Table rows.
    """
    rows = []
    for name, item in payload["capability_summary"].items():
        primary = item["best_accuracy"]
        if primary is None:
            primary = item["best_after_accuracy"]
        if primary is None:
            primary = item["best_verified_objective"]
        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{format_value(item['max_completed_dim'])}</td>"
            f"<td>{format_value(item['max_success_dim'])}</td>"
            f"<td>{item['num_completed']}</td>"
            f"<td>{item['num_skipped']}</td>"
            f"<td>{item['num_failed']}</td>"
            f"<td>{format_value(primary)}</td>"
            f"<td>{item['boundary_case'] or '-'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_budget_rows(payload: Dict[str, Any]) -> str:
    """
    Build HTML rows for the computational budget table.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Table rows.
    """
    settings = payload["settings"]
    rows = [
        ("KRG", settings["max_krg_dim"], settings["max_krg_train"], "Complete-model MLE with anisotropic theta."),
        ("SVR", settings["max_svr_dim"], settings["max_svr_train"], "Dual SLSQP solve."),
        ("TAHS/AESMSI", settings["max_ensemble_dim"], settings["max_ensemble_train"], "Exact model-library LOO screening."),
        ("MFSMLS", settings["max_mfsmls_dim"], settings["max_mfsmls_hf"], "Local weighted least-squares correction."),
        ("MMFS", settings["max_mmfs_dim"], settings["max_mmfs_hf"], "LOOCV-based multi-fidelity correction."),
        ("CCAMFS", settings["max_ccamfs_dim"], settings["max_ccamfs_hf"], "CCA and coupled RBF residual correction."),
        ("DISO/MICO/MOBO", settings["max_active_dim"], "-", "Repeated KRG refits during infill."),
    ]
    return "\n".join(
        f"<tr><td>{name}</td><td>{d}</td><td>{n}</td><td>{note}</td></tr>" for name, d, n, note in rows
    )


def build_skipped_notes(payload: Dict[str, Any]) -> str:
    """
    Build HTML list items for skipped benchmark records.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: HTML list items.
    """
    notes = {}
    for section in payload["sections"].values():
        for case in section:
            for item in case["algorithms"]:
                if item["status"] == "skipped":
                    notes.setdefault(item["name"], item["reason"])
    return "\n".join(f"<li><strong>{name}</strong>: {reason}</li>" for name, reason in notes.items())


def build_algorithm_guidance_rows() -> str:
    """
    Build HTML rows for the algorithm guide table.

    Returns:
        str: Table rows.
    """
    rows = [
        (
            "PRS",
            "单保真响应面",
            r"\(\hat y(x)=\phi(x)^\mathsf{T}\beta\)",
            "适用于低阶、平滑、近似线性的响应；在高维低样本条件下可作为稳定基线。",
            "难以表达局部强非线性、多峰结构或窄谷结构。",
        ),
        (
            "RBF",
            "单保真插值",
            r"\(\hat y(x)=\sum_i w_i \exp(-\gamma\lVert x-c_i\rVert_2^2)\)",
            "适用于中小样本、平滑非线性响应；训练直接，预测成本可控。",
            "高维中距离集中会削弱局部性，样本覆盖不足时容易退化。",
        ),
        (
            "KRG",
            "单保真 Kriging",
            r"\(\hat y(x)=f(x)^\mathsf{T}\beta+r(x)^\mathsf{T}R^{-1}(y-F\beta)\)",
            "适用于中低维、小样本且需要不确定性估计的场景；主动学习策略依赖该能力。",
            "各向异性超参数优化随维度和样本数迅速变重，不宜把预算截断误读为理论上限。",
        ),
        (
            "SVR",
            "单保真回归",
            r"\(\min \frac12\lVert w\rVert^2+C\sum_i(\xi_i+\xi_i^*)\)",
            "线性核适合作为高维稳健基线；对噪声和异常响应相对不敏感。",
            "默认参数偏保守，非线性复杂函数上通常不如专门代理模型。",
        ),
        (
            "TAHS",
            "组合代理",
            r"\(\hat y=\sum_m \omega_m\hat y_m\)",
            "适用于低维到中维、单一模型选择不确定的情形；通过模型库筛选降低单模型失误风险。",
            "当前实现需要对候选模型做精确留一评估，KRG 子模型会显著提高计算成本。",
        ),
        (
            "AESMSI",
            "组合代理",
            r"\(\hat y=\sum_m \omega_m(x)\hat y_m(x)\)",
            "适用于响应结构随区域变化、不同基模型在不同区域表现互补的场景。",
            "同样受模型库留一筛选成本限制；高维时应优先缩小候选模型或增加预算。",
        ),
        (
            "MFSMLS",
            "多保真代理",
            r"\(\hat y_H(x)=p(x)^\mathsf{T}a(x)\)",
            "适用于低保真与高保真强相关、且低保真样本明显多于高保真样本的任务；本次边界实验显示其高维完成能力较强。",
            "局部多项式基在高维会膨胀，若高保真样本极少，应保持低有效维度或强低保真相关性。",
        ),
        (
            "MMFS",
            "多保真代理",
            r"\(\hat y_H(x)=\rho(x)\hat y_L(x)+\delta(x)\)",
            "适用于低保真趋势可靠、只需学习尺度和偏差修正的任务。",
            "内部含 LOOCV 型校正参数搜索，高维或高保真样本增加时计算成本上升明显。",
        ),
        (
            "CCAMFS",
            "多保真代理",
            r"\(Z_H=P_HU,\;Z_L=P_LV\)",
            "适用于高低保真样本在联合输入-响应空间中存在可对齐相关结构的任务。",
            "CCA 对协方差矩阵条件数敏感；低有效维高维输入中可能出现 ill-conditioned warning。",
        ),
        (
            "DISO",
            "单目标主动学习",
            r"\(x_{new}=\arg\max_x \alpha(x)d(x)\)",
            "适用于昂贵单目标函数，目标是在有限追加样本下提升 KRG 预测质量或搜索质量。",
            "每轮需要重拟合 KRG；高维时建议先降低有效维度或使用候选池。",
        ),
        (
            "MICO",
            "多保真主动学习",
            r"\(s_i=\Delta_N(i)\Delta_D(i)\)",
            "适用于已有低保真候选池、需要选择少量高保真补点的场景；当低保真候选覆盖主变化方向时效率较高。",
            "候选质量决定上限；若低保真池过稀，高维中可能无法覆盖重要区域。",
        ),
        (
            "MOBO",
            "多目标主动学习",
            r"\(x_{new}=\arg\max_x \mathrm{acq}(x)\)",
            "适用于多目标 Pareto 前沿探索，尤其适合低维到中维的候选生成流程。",
            "当前随机候选规模固定，高维时候选覆盖不足会先于模型能力成为瓶颈。",
        ),
        (
            "MIGA",
            "直接优化",
            r"\(x^*=\arg\min_x \hat f(x)\)",
            "适用于代理模型已给定、需要全局搜索候选解的单目标优化。",
            "优化结果受代理模型误差控制；应始终用真实函数或高保真模型复核候选点。",
        ),
        (
            "CFARSSDA",
            "直接优化",
            r"\(x^*=\arg\min_x \hat f(x)\)",
            "适用于需要与遗传类策略形成互补的群智能搜索流程。",
            "高维优化中种群预算决定覆盖能力；默认设置更适合快速验证而非最终寻优。",
        ),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{name}</td><td>{kind}</td><td>{formula}</td><td>{use_case}</td><td>{limit}</td>"
        "</tr>"
        for name, kind, formula, use_case, limit in rows
    )


def build_recommendation_rows(payload: Dict[str, Any]) -> str:
    """
    Build HTML rows for scenario-level recommendations.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Table rows.
    """
    summary = payload["capability_summary"]
    rows = [
        (
            "低维、小样本、需要不确定性",
            "KRG、DISO",
            f"KRG 在本次预算内完成到 {format_value(summary['KRG']['max_completed_dim'])} 维；DISO 完成到 {format_value(summary['DISO']['max_completed_dim'])} 维。",
            "优先用于主动学习和小样本代理建模；样本或维度增加时应监控 KRG 拟合时间。",
        ),
        (
            "单保真高维快速基线",
            "PRS、RBF、SVR",
            f"PRS/RBF 完成到 {format_value(summary['PRS']['max_completed_dim'])} 维；SVR 在当前预算内完成到 {format_value(summary['SVR']['max_completed_dim'])} 维。",
            "适合先建立可运行基线，再决定是否投入更重的 KRG 或组合代理。",
        ),
        (
            "单保真模型选择不确定",
            "TAHS、AESMSI",
            f"组合代理在当前留一筛选预算内完成到 {format_value(summary['TAHS']['max_completed_dim'])} 维。",
            "适合中低维工程响应；高维应用应降低候选模型数量或改进留一评估成本。",
        ),
        (
            "低保真数据充足且相关性强",
            "MFSMLS、MMFS、CCAMFS",
            f"MFSMLS 完成到 {format_value(summary['MFSMLS']['max_completed_dim'])} 维，CCAMFS 完成到 {format_value(summary['CCAMFS']['max_completed_dim'])} 维。",
            "优先考虑多保真代理；若低保真与高保真相关性弱，三类方法都会显著受限。",
        ),
        (
            "低保真候选池已存在",
            "MICO",
            f"MICO 在当前主动学习预算内完成到 {format_value(summary['MICO']['max_completed_dim'])} 维。",
            "适合从低保真候选池中选择高保真追加点；候选池应覆盖主要设计方向。",
        ),
        (
            "多目标探索",
            "MOBO",
            f"MOBO 完成到 {format_value(summary['MOBO']['max_completed_dim'])} 维，达到阈值最高维为 {format_value(summary['MOBO']['max_success_dim'])}。",
            "适合低维 Pareto 前沿探索；高维时应增大候选池或引入降维策略。",
        ),
        (
            "代理驱动直接优化",
            "MIGA、CFARSSDA",
            f"两个优化器均完成到 {format_value(summary['MIGA']['max_completed_dim'])} 维。",
            "可用于快速产生候选解；最终结论应以真实函数验证值而非代理目标值为准。",
        ),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{scenario}</td><td>{algorithms}</td><td>{evidence}</td><td>{advice}</td>"
        "</tr>"
        for scenario, algorithms, evidence, advice in rows
    )


def build_report_html(payload: Dict[str, Any], images: Dict[str, str]) -> str:
    """
    Build a standalone academic Chinese HTML report.

    Args:
        payload (Dict[str, Any]): Benchmark payload.
        images (Dict[str, str]): Embedded figure payloads.

    Returns:
        str: HTML document.
    """
    settings = payload["settings"]
    sources = "\n".join(
        f"<li><a href=\"{item['url']}\">{item['name']}</a>: {item['usage']}</li>"
        for item in payload["sources"]
    )
    return fr"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SurrogateLab 代理模型算法选择说明</title>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    body {{ margin: 0; color: #17202a; background: #f6f7f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 58px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 34px 0 12px; font-size: 21px; }}
    h3 {{ margin: 22px 0 8px; font-size: 16px; }}
    p, li {{ line-height: 1.75; }}
    .lead {{ font-size: 16px; color: #314151; }}
    .meta {{ color: #53616f; margin-bottom: 22px; }}
    .panel {{ background: #fff; border: 1px solid #e2e6eb; border-radius: 8px; padding: 18px 20px; margin: 18px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ background: #ffffff; border: 1px solid #e2e6eb; border-radius: 8px; padding: 14px 15px; }}
    .card .k {{ color: #647382; font-size: 13px; }}
    .card .v {{ font-size: 23px; font-weight: 700; margin-top: 4px; }}
    .formula {{ margin: 10px 0; padding: 11px 13px; background: #f0f4f8; border-left: 4px solid #426a8c; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e8edf2; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f7; font-weight: 650; }}
    img {{ width: 100%; display: block; background: #fff; border: 1px solid #e2e6eb; border-radius: 8px; margin: 12px 0 22px; }}
    code {{ background: #edf1f5; border-radius: 4px; padding: 1px 5px; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 22px 16px 42px; }} }}
  </style>
</head>
<body>
<main>
  <h1>SurrogateLab 代理模型算法选择说明</h1>
  <div class="meta">生成时间：{payload['created_at']}；随机种子：{payload['seeds']}；测试套件：{payload['suite']}。</div>

  <section class="grid">
    <div class="card"><div class="k">最高设计维度</div><div class="v">{settings['max_dim']}</div></div>
    <div class="card"><div class="k">单保真 case</div><div class="v">{len(payload['sections']['scalar'])}</div></div>
    <div class="card"><div class="k">多保真 case</div><div class="v">{len(payload['sections']['multifidelity'])}</div></div>
    <div class="card"><div class="k">优化 case</div><div class="v">{len(payload['sections']['optimization'])}</div></div>
  </section>

  <section class="panel">
    <h2>说明目标</h2>
    <p class="lead">本文档的核心目标是为 SurrogateLab 中的代理建模、组合代理、多保真建模、主动学习和代理驱动优化提供算法选择依据。边界测试不是目的本身，而是用于回答一个实际问题：在给定样本预算、维度范围和当前实现条件下，哪些算法更适合特定类型的任务。</p>
    <p>所有实验均使用解析 benchmark 函数，以便真实响应可直接计算。输入矩阵记为 \(X\in\mathbb{{R}}^{{N\times D}}\)，其中 \(N\) 为样本数，\(D\) 为设计变量维度；输出矩阵记为 \(Y\in\mathbb{{R}}^{{N\times C}}\)。标量响应任务中 \(C=1\)，多目标 DTLZ2 任务中 \(C=2\)。本文档中的维度边界和样本边界均为当前实验配置下的观测结果，不代表算法理论极限。</p>
  </section>

  <h2>指标定义</h2>
  <div class="formula">\[
    \mathrm{{Accuracy}}=\left(1-\frac{{\sum_i |Y_i^{{true}}-Y_i^{{pred}}|}}{{\sum_i |Y_i^{{true}}|+\varepsilon}}\right)\times 100.
  \]</div>
  <div class="formula">\[
    R^2=1-\frac{{\sum_i (Y_i^{{true}}-Y_i^{{pred}})^2}}{{\sum_i (Y_i^{{true}}-\bar Y^{{true}})^2+\varepsilon}}.
  \]</div>
  <p><strong>Accuracy</strong> 是本项目的主指标，表示预测绝对误差相对于真实响应绝对值总量的剩余比例；<strong>\(R^2\)</strong> 是辅助指标，用于观察方差解释能力。主动学习记录使用补点前后的 accuracy 计算相对增益：\(\Delta=(A_{{after}}-A_{{before}})/\max(|A_{{before}}|,\varepsilon)\)。</p>
  <p><strong>完成数</strong>表示算法完成拟合、预测和指标计算的 case 数。<strong>跳过数</strong>表示该 case 超出预设计算预算而未执行；例如 KRG 的默认边界为 \(D\le {settings['max_krg_dim']}\) 且 \(N\le {settings['max_krg_train']}\)，TAHS/AESMSI 的默认边界为 \(D\le {settings['max_ensemble_dim']}\) 且 \(N\le {settings['max_ensemble_train']}\)。跳过记录不表示算法数值失败。<strong>失败数</strong>表示运行中出现异常。<strong>达到阈值最高维</strong>表示在已完成 case 中，accuracy 不低于 {payload['success_accuracy']:.0f}%；对直接优化记录，则表示真实函数验证目标值不高于 {payload['optimization_success']:.2f}。</p>

  <h2>实验配置</h2>
  <table>
    <thead><tr><th>算法族</th><th>维度上限</th><th>样本上限</th><th>限制对象</th></tr></thead>
    <tbody>{build_budget_rows(payload)}</tbody>
  </table>
  <p>维度 sweep 采用 \(D\in {DIMENSION_SWEEP}\)。单保真维度 sweep 的训练样本数为 \(N={settings['dimension_sample_factor']}D\)；固定 50 维样本量 sweep 采用 \(N/D\in {SCALAR_SAMPLE_FACTORS}\)。多保真维度 sweep 采用约 \(N_{{HF}}={settings['multifidelity_dimension_hf_factor']}D\)，且 \(N_{{LF}}={settings['lf_to_hf_ratio']}N_{{HF}}\)。主动学习部分使用初始样本加 {settings['active_infill']} 个 infill 点；多目标候选池大小为 {settings['mobo_num_samples']}，候选保留数为 {settings['mobo_num_candidates']}。</p>

  <h2>测试函数</h2>
  <h3>单保真标量函数</h3>
  <div class="formula">\[
    f_{{Ackley}}(x)=-20\exp\left(-0.2\sqrt{{\frac1D\sum_j x_j^2}}\right)-\exp\left(\frac1D\sum_j \cos(2\pi x_j)\right)+20+e.
  \]</div>
  <div class="formula">\[
    f_{{Rosenbrock}}(x)=\frac1{{D-1}}\sum_{{j=1}}^{{D-1}}\left[100(x_{{j+1}}-x_j^2)^2+(1-x_j)^2\right].
  \]</div>
  <div class="formula">\[
    f_{{SobolG}}(x)=\prod_{{j=1}}^D \frac{{|4x_j-2|+a_j}}{{1+a_j}},\qquad x\in[0,1]^D.
  \]</div>
  <p>Ackley 用于检验多峰与平坦外区，Rosenbrock 用于检验强耦合窄谷，Sobol-G 用于检验乘积型灵敏度结构。函数值按维度作适度归一化，使不同维度下的 accuracy 更具有可比性。</p>
  <h3>多保真函数</h3>
  <div class="formula">\[
    Y_{{HF}}(x)=f(x),\qquad
    Y_{{LF}}(x)=0.72f(x)+0.28f(w(x))+0.05\,\mathrm{{mean}}(u(x)-0.5).
  \]</div>
  <p>低保真函数保留高保真响应主趋势，同时引入坐标扭曲与小幅偏置，用于评估 MFSMLS、MMFS、CCAMFS 对跨保真相关性的利用能力。</p>
  <h3>多目标与优化函数</h3>
  <div class="formula">\[
    g(x)=\sum_{{j=M}}^D (x_j-0.5)^2,
  \]</div>
  <p>DTLZ2 目标函数采用标准球面 Pareto 前沿映射。直接优化部分使用 Ackley、Rastrigin 与 Rosenbrock 的可扩展标量版本，记录优化器候选点在真实函数上的验证目标值。</p>

  <h2>算法介绍</h2>
  <table>
    <thead><tr><th>算法</th><th>类型</th><th>核心形式</th><th>适用场景</th><th>主要限制</th></tr></thead>
    <tbody>{build_algorithm_guidance_rows()}</tbody>
  </table>

  <h2>使用建议</h2>
  <p>以下建议基于本次实验结果、默认计算预算和当前实现方式。若实际工程问题的响应尺度、噪声、低保真相关性或约束结构发生变化，应重新运行相同流程进行校准。</p>
  <table>
    <thead><tr><th>场景</th><th>优先算法</th><th>实验依据</th><th>建议</th></tr></thead>
    <tbody>{build_recommendation_rows(payload)}</tbody>
  </table>

  <h2>算法能力摘要</h2>
  <table>
    <thead>
      <tr><th>算法</th><th>已完成最高维</th><th>达到阈值最高维</th><th>完成数</th><th>跳过数</th><th>失败数</th><th>最佳指标</th><th>边界 case</th></tr>
    </thead>
    <tbody>{build_capability_rows(payload)}</tbody>
  </table>

  <h2>可视化结果</h2>
  <p>下列曲线用于支撑算法选择建议。图中横轴为维度或样本比例，纵轴为 accuracy 或真实函数验证目标值；直接优化图采用对数纵轴，以便比较不同维度下的目标值数量级。</p>
  <h3>单保真维度 sweep</h3>
  <img alt="Scalar dimension sweep" src="data:image/png;base64,{images['scalar_dimension']}">
  <h3>单保真 50 维样本量 sweep</h3>
  <img alt="Scalar sample sweep" src="data:image/png;base64,{images['scalar_sample']}">
  <h3>多保真维度 sweep</h3>
  <img alt="Multifidelity dimension sweep" src="data:image/png;base64,{images['multifidelity']}">
  <h3>直接优化维度 sweep</h3>
  <img alt="Optimization sweep" src="data:image/png;base64,{images['optimization']}">

  <section class="panel">
    <h2>跳过记录说明</h2>
    <ul>{build_skipped_notes(payload)}</ul>
  </section>

  <section class="panel">
    <h2>参考来源</h2>
    <ul>{sources}</ul>
  </section>
</main>
</body>
</html>
"""


def save_html_report(payload: Dict[str, Any]) -> str:
    """
    Save a standalone HTML report for the boundary benchmark.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Absolute save path.
    """
    images = {
        "scalar_dimension": plot_accuracy_by_dimension(payload),
        "scalar_sample": plot_accuracy_by_sample_factor(payload),
        "multifidelity": plot_multifidelity_accuracy(payload),
        "optimization": plot_optimizer_objective(payload),
    }
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_boundary_report.html")
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(build_report_html(payload, images))
    return save_path


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the scalable benchmark suite.

    Returns:
        argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(
        description="SurrogateLab scalable boundary benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("General")
    general.add_argument("--seeds", nargs="+", type=int, default=[1], help="Seed list used by the benchmark suite.")
    general.add_argument("--max_dim", type=int, default=100, help="Maximum benchmark dimension.")
    general.add_argument(
        "--demos",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=list(ALGORITHM_ORDER),
        help="Contract-facing algorithms to run.",
    )
    general.add_argument("--include_classical", action="store_true", default=True, help="Include PRS/RBF/KRG/SVR context.")

    sampling = parser.add_argument_group("Sampling")
    sampling.add_argument("--dimension_sample_factor", type=float, default=2.0, help="N/D for dimension sweeps.")
    sampling.add_argument("--multifidelity_dimension_hf_factor", type=float, default=1.2, help="HF/D for MF dimension sweeps.")
    sampling.add_argument("--lf_to_hf_ratio", type=float, default=2.5, help="LF/HF sample ratio.")
    sampling.add_argument("--test_factor", type=int, default=8, help="N_TEST/D before min/max clipping.")
    sampling.add_argument("--min_test", type=int, default=200, help="Minimum number of test samples.")
    sampling.add_argument("--max_test", type=int, default=800, help="Maximum number of test samples.")

    models = parser.add_argument_group("Surrogate Models")
    models.add_argument("--ensemble_threshold", type=float, default=0.5, help="Threshold used by TAHS and AES-MSI.")
    models.add_argument("--prs_degree", type=int, default=1, help="Polynomial degree for scalable PRS context.")
    models.add_argument("--prs_alpha", type=float, default=1.0e-8, help="Ridge regularization for PRS.")
    models.add_argument("--svr_kernel", type=str, default="linear", choices=["rbf", "linear"], help="SVR kernel type.")
    models.add_argument("--svr_gamma", type=float, default=None, help="SVR kernel coefficient.")
    models.add_argument("--svr_C", type=float, default=0.1, help="SVR regularization parameter.")
    models.add_argument("--svr_epsilon", type=float, default=0.2, help="SVR epsilon-insensitive tube width.")
    models.add_argument("--krg_theta0", type=float, default=0.1, help="Initial KRG theta.")
    models.add_argument("--krg_theta_bounds", type=float, nargs=2, default=[1.0e-3, 10.0], help="KRG theta bounds.")
    models.add_argument("--mfs_mls_neighbor_factor", type=float, default=1.2, help="MFS-MLS neighborhood factor.")
    models.add_argument("--mfs_mls_ridge", type=float, default=1.0e-4, help="MFS-MLS ridge.")

    limits = parser.add_argument_group("Scale Limits")
    limits.add_argument("--max_krg_dim", type=int, default=50, help="Largest D for direct KRG accuracy tests.")
    limits.add_argument("--max_krg_train", type=int, default=120, help="Largest N for direct KRG accuracy tests.")
    limits.add_argument("--max_svr_dim", type=int, default=50, help="Largest D for direct SVR accuracy tests.")
    limits.add_argument("--max_svr_train", type=int, default=80, help="Largest N for direct SVR accuracy tests.")
    limits.add_argument("--max_ensemble_dim", type=int, default=20, help="Largest D for TAHS/AESMSI tests.")
    limits.add_argument("--max_ensemble_train", type=int, default=40, help="Largest N for TAHS/AESMSI tests.")
    limits.add_argument("--max_mfsmls_dim", type=int, default=100, help="Largest D for MFSMLS tests.")
    limits.add_argument("--max_mfsmls_hf", type=int, default=180, help="Largest HF count for MFSMLS tests.")
    limits.add_argument("--max_mmfs_dim", type=int, default=20, help="Largest D for MMFS tests.")
    limits.add_argument("--max_mmfs_hf", type=int, default=40, help="Largest HF count for MMFS tests.")
    limits.add_argument("--max_ccamfs_dim", type=int, default=50, help="Largest D for CCAMFS tests.")
    limits.add_argument("--max_ccamfs_hf", type=int, default=120, help="Largest HF count for CCAMFS tests.")
    limits.add_argument("--max_ccamfs_lf", type=int, default=300, help="Largest LF count for CCAMFS tests.")
    limits.add_argument("--max_active_dim", type=int, default=20, help="Largest D for KRG-based active-learning tests.")

    active = parser.add_argument_group("Active Learning")
    active.add_argument("--active_family", type=str, default="ackley", choices=SCALAR_FAMILIES, help="Active-learning family.")
    active.add_argument("--active_initial_min", type=int, default=10, help="Minimum active-learning initial samples.")
    active.add_argument("--active_initial_offset", type=int, default=5, help="Initial samples are D plus this offset.")
    active.add_argument("--active_infill", type=int, default=8, help="Number of active-learning infill points.")
    active.add_argument("--active_lf_factor", type=float, default=2.5, help="LF samples relative to initial HF samples.")
    active.add_argument("--diso_alpha", type=float, default=4.0, help="DISO distance penalty intensity.")
    active.add_argument("--diso_min_distance", type=float, default=0.02, help="DISO minimum normalized distance.")
    active.add_argument("--diso_distance_scale", type=float, default=None, help="DISO optional distance scale.")
    active.add_argument("--mico_ratio", type=float, default=0.5, help="MICO exploration ratio.")
    active.add_argument("--mobo_num_samples", type=int, default=1500, help="MOBO random samples.")
    active.add_argument("--mobo_num_candidates", type=int, default=120, help="MOBO candidate count.")
    active.add_argument("--mobo_num_restarts", type=int, default=4, help="MOBO restart count.")
    active.add_argument("--mobo_beta", type=float, default=0.3, help="MOBO variance weight.")

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--optimizer_popsize", type=int, default=2, help="Population-size multiplier.")
    optimization.add_argument("--optimizer_maxiter", type=int, default=10, help="Maximum optimizer iterations.")
    optimization.add_argument("--miga_num_islands", type=int, default=4, help="Number of MIGA islands.")
    optimization.add_argument("--miga_migration_interval", type=int, default=5, help="MIGA migration interval.")
    optimization.add_argument("--miga_migration_size", type=int, default=2, help="MIGA migration size.")
    optimization.add_argument("--opt_tol", type=float, default=1.0e-6, help="Stopping tolerance.")

    metrics = parser.add_argument_group("Metrics")
    metrics.add_argument("--metric_eps", type=float, default=1.0e-12, help="Stability epsilon for metrics.")
    metrics.add_argument("--success_accuracy", type=float, default=80.0, help="Usable predictive accuracy threshold.")
    metrics.add_argument("--strong_accuracy", type=float, default=90.0, help="Strong predictive accuracy threshold.")
    metrics.add_argument("--optimization_success", type=float, default=0.2, help="Usable direct-optimization objective threshold.")

    args = parser.parse_args()
    args.demos = list(dict.fromkeys(args.demos))
    args.seeds = list(args.seeds)
    args.krg_params = {
        "poly": "constant",
        "kernel": "gaussian",
        "theta0": args.krg_theta0,
        "theta_bounds": tuple(args.krg_theta_bounds),
    }
    args.prs_params = {"degree": args.prs_degree, "alpha": args.prs_alpha}
    args.svr_params = {
        "kernel": args.svr_kernel,
        "gamma": args.svr_gamma,
        "C": args.svr_C,
        "epsilon": args.svr_epsilon,
    }
    args.miga_params = {
        "popsize": args.optimizer_popsize,
        "maxiter": args.optimizer_maxiter,
        "num_islands": args.miga_num_islands,
        "migration_interval": args.miga_migration_interval,
        "migration_size": args.miga_migration_size,
    }
    args.df_params = {"popsize": args.optimizer_popsize, "maxiter": args.optimizer_maxiter}
    return args


def print_summary(payload: Dict[str, Any]) -> None:
    """
    Print a compact algorithm-level capability summary.

    Args:
        payload (Dict[str, Any]): Benchmark payload.
    """
    logger.info(f"{hue.b}Capability Summary{hue.q}")
    for name, item in payload["capability_summary"].items():
        logger.info(
            f"  {name}: completed={item['num_completed']} | skipped={item['num_skipped']} | "
            f"failed={item['num_failed']} | max_success_dim={item['max_success_dim']}"
        )


def main() -> None:
    """
    Execute the scalable analytic benchmark workflow.
    """
    args = get_args()

    logger.info(f"{hue.b}SurrogateLab Scalable Boundary Benchmarks{hue.q}")
    logger.info(f"  seeds     : {args.seeds}")
    logger.info(f"  demos     : {args.demos}")
    logger.info(f"  max_dim   : {args.max_dim}")

    payload = run_bench_suite(args)
    print_summary(payload)
    save_path = save_results(payload)
    html_path = save_html_report(payload)
    logger.info(f"{hue.g}Benchmark results saved to {save_path}{hue.q}")
    logger.info(f"{hue.g}Boundary report saved to {html_path}{hue.q}")


if __name__ == "__main__":
    main()
