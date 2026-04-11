# Benchmark function registry for SurrogateLab
# Author: Shengning Wang

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np


ArrayFn = Callable[[np.ndarray], np.ndarray]


def _as_2d_array(x: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    """
    Convert user input to a 2-D float64 array.

    Args:
        x (np.ndarray): Input array. (N, D) or (D,).
        expected_dim (int): Expected feature dimension.
        name (str): Function name for error messages.

    Returns:
        np.ndarray: Reshaped array. (N, D).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"{name} expects shape (N, {expected_dim}) or ({expected_dim},), got {arr.shape}.")
    return arr


@dataclass(frozen=True)
class ScalarBenchmark:
    """
    Scalar-output benchmark specification.

    Args:
        name (str): Benchmark name.
        input_dim (int): Input dimension.
        bounds (Tuple[Tuple[float, float], ...]): Box bounds. (D, 2).
        output_name (str): Output name.
        description (str): Short description.
        evaluator (ArrayFn): Benchmark function. (N, D) -> (N, 1).
        known_optimum (Optional[float]): Known minimum value.
        known_minimizers (Tuple[Tuple[float, ...], ...]): Known minimizers.
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
    output_name: str
    description: str
    evaluator: ArrayFn
    known_optimum: Optional[float] = None
    known_minimizers: Tuple[Tuple[float, ...], ...] = field(default_factory=tuple)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the scalar benchmark.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: Responses. (N, 1).
        """
        return self.evaluator(x)

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
        name (str): Benchmark name.
        input_dim (int): Input dimension.
        bounds (Tuple[Tuple[float, float], ...]): Box bounds. (D, 2).
        output_name (str): Output name.
        description (str): Short description.
        high_fidelity (ArrayFn): HF function. (N, D) -> (N, 1).
        low_fidelity (ArrayFn): LF function. (N, D) -> (N, 1).
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
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
        return self.high_fidelity(x)

    def evaluate_low_fidelity(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the low-fidelity response.

        Args:
            x (np.ndarray): Query points. (N, D) or (D,).

        Returns:
            np.ndarray: LF responses. (N, 1).
        """
        return self.low_fidelity(x)

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
        name (str): Benchmark name.
        input_dim (int): Input dimension.
        bounds (Tuple[Tuple[float, float], ...]): Box bounds. (D, 2).
        output_names (Tuple[str, ...]): Objective names.
        description (str): Short description.
        evaluator (ArrayFn): Objective function. (N, D) -> (N, M).
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
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
        return self.evaluator(x)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Return bounds as a float64 array.

        Returns:
            np.ndarray: Box bounds. (D, 2).
        """
        return np.asarray(self.bounds, dtype=np.float64)


def forrester(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Forrester benchmark.

    Args:
        x (np.ndarray): Query points. (N, 1) or (1,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 1, "forrester")
    z = x_arr[:, 0]
    y = (6.0 * z - 2.0) ** 2 * np.sin(12.0 * z - 4.0)
    return y.reshape(-1, 1)


def gramacy_lee(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Gramacy-Lee benchmark.

    Args:
        x (np.ndarray): Query points. (N, 1) or (1,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 1, "gramacy_lee")
    z = np.clip(x_arr[:, 0], 0.5, 2.5)
    y = np.sin(10.0 * np.pi * z) / (2.0 * z) + (z - 1.0) ** 4
    return y.reshape(-1, 1)


def branin(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Branin-Hoo benchmark.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 2, "branin")
    x1 = x_arr[:, 0]
    x2 = x_arr[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
    return y.reshape(-1, 1)


def branin_low_fidelity(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the low-fidelity Branin variant.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 2, "branin_low_fidelity")
    hf = branin(x_arr)[:, 0]
    y = 0.9 * hf + 0.4 * np.sin(x_arr[:, 0]) - 0.2 * x_arr[:, 1] + 2.0
    return y.reshape(-1, 1)


def hartman3(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the three-dimensional Hartman benchmark.

    Args:
        x (np.ndarray): Query points. (N, 3) or (3,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 3, "hartman3")
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
    a_mat = np.array(
        [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]],
        dtype=np.float64,
    )
    p_mat = 1.0e-4 * np.array(
        [[3689.0, 1170.0, 2673.0], [4699.0, 4387.0, 7470.0], [1091.0, 8732.0, 5547.0], [381.0, 5743.0, 8828.0]],
        dtype=np.float64,
    )
    total = np.zeros(x_arr.shape[0], dtype=np.float64)
    for idx in range(4):
        total += alpha[idx] * np.exp(-np.sum(a_mat[idx] * (x_arr - p_mat[idx]) ** 2, axis=1))
    return (-total).reshape(-1, 1)


def currin_exponential(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Currin exponential benchmark.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 2, "currin_exponential")
    x1 = np.clip(x_arr[:, 0], 1.0e-6, 1.0)
    x2 = np.clip(x_arr[:, 1], 1.0e-6, 1.0)
    numerator = 2300.0 * x1 ** 3 + 1900.0 * x1 ** 2 + 2092.0 * x1 + 60.0
    denominator = 100.0 * x1 ** 3 + 500.0 * x1 ** 2 + 4.0 * x1 + 20.0
    y = (1.0 - np.exp(-1.0 / (2.0 * x2))) * (numerator / denominator)
    return y.reshape(-1, 1)


def currin_exponential_low_fidelity(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the low-fidelity Currin approximation.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 2, "currin_exponential_low_fidelity")
    shifts = np.array([[0.05, 0.05], [0.05, -0.05], [-0.05, 0.05], [-0.05, -0.05]], dtype=np.float64)
    values = []
    for shift in shifts:
        shifted = x_arr + shift
        shifted[:, 0] = np.clip(shifted[:, 0], 0.0, 1.0)
        shifted[:, 1] = np.clip(shifted[:, 1], 0.0, 1.0)
        values.append(currin_exponential(shifted)[:, 0])
    return (0.25 * np.sum(values, axis=0)).reshape(-1, 1)


def park91b(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Park91B high-fidelity benchmark.

    Args:
        x (np.ndarray): Query points. (N, 4) or (4,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 4, "park91b")
    y = (2.0 / 3.0) * np.exp(x_arr[:, 0] + x_arr[:, 1]) - x_arr[:, 3] * np.sin(x_arr[:, 2]) + x_arr[:, 2]
    return y.reshape(-1, 1)


def park91b_low_fidelity(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Park91B low-fidelity benchmark.

    Args:
        x (np.ndarray): Query points. (N, 4) or (4,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 4, "park91b_low_fidelity")
    return (1.2 * park91b(x_arr)[:, 0] - 1.0).reshape(-1, 1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the Rastrigin benchmark.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 2, "rastrigin")
    y = 20.0 + np.sum(x_arr ** 2 - 10.0 * np.cos(2.0 * np.pi * x_arr), axis=1)
    return y.reshape(-1, 1)


def rosenbrock5(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the five-dimensional Rosenbrock benchmark.

    Args:
        x (np.ndarray): Query points. (N, 5) or (5,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 5, "rosenbrock5")
    y = np.sum(100.0 * (x_arr[:, 1:] - x_arr[:, :-1] ** 2) ** 2 + (1.0 - x_arr[:, :-1]) ** 2, axis=1)
    return y.reshape(-1, 1)


def vlmop2(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the VLMOP2 benchmark.

    Args:
        x (np.ndarray): Query points. (N, 2) or (2,).

    Returns:
        np.ndarray: Objective values. (N, 2).
    """
    x_arr = _as_2d_array(x, 2, "vlmop2")
    shift = 1.0 / np.sqrt(2.0)
    f1 = 1.0 - np.exp(-((x_arr[:, 0] - shift) ** 2 + (x_arr[:, 1] - shift) ** 2))
    f2 = 1.0 - np.exp(-((x_arr[:, 0] + shift) ** 2 + (x_arr[:, 1] + shift) ** 2))
    return np.column_stack([f1, f2])


SCALAR_BENCHMARKS: Dict[str, ScalarBenchmark] = {
    "forrester": ScalarBenchmark(
        name="forrester",
        input_dim=1,
        bounds=((0.0, 1.0),),
        output_name="response",
        description="One-dimensional oscillatory Forrester benchmark.",
        evaluator=forrester,
    ),
    "gramacy_lee": ScalarBenchmark(
        name="gramacy_lee",
        input_dim=1,
        bounds=((0.5, 2.5),),
        output_name="response",
        description="One-dimensional Gramacy-Lee interpolation benchmark.",
        evaluator=gramacy_lee,
    ),
    "branin": ScalarBenchmark(
        name="branin",
        input_dim=2,
        bounds=((-5.0, 10.0), (0.0, 15.0)),
        output_name="response",
        description="Two-dimensional Branin-Hoo benchmark.",
        evaluator=branin,
        known_optimum=0.39788735772973816,
        known_minimizers=((-np.pi, 12.275), (np.pi, 2.275), (3.0 * np.pi, 2.475)),
    ),
    "hartman3": ScalarBenchmark(
        name="hartman3",
        input_dim=3,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Three-dimensional Hartman benchmark.",
        evaluator=hartman3,
        known_optimum=-3.8627797869493365,
        known_minimizers=((0.114614, 0.555649, 0.852547),),
    ),
    "currin_exponential": ScalarBenchmark(
        name="currin_exponential",
        input_dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Two-dimensional Currin exponential benchmark.",
        evaluator=currin_exponential,
    ),
    "park91b": ScalarBenchmark(
        name="park91b",
        input_dim=4,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Four-dimensional Park91B benchmark.",
        evaluator=park91b,
    ),
    "rastrigin": ScalarBenchmark(
        name="rastrigin",
        input_dim=2,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        output_name="response",
        description="Two-dimensional Rastrigin benchmark.",
        evaluator=rastrigin,
        known_optimum=0.0,
        known_minimizers=((0.0, 0.0),),
    ),
    "rosenbrock5": ScalarBenchmark(
        name="rosenbrock5",
        input_dim=5,
        bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
        output_name="response",
        description="Five-dimensional Rosenbrock benchmark.",
        evaluator=rosenbrock5,
        known_optimum=0.0,
        known_minimizers=((1.0, 1.0, 1.0, 1.0, 1.0),),
    ),
}


def borehole_high_fidelity(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the standard Borehole high-fidelity benchmark.

    Args:
        x (np.ndarray): Query points. (N, 8) or (8,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 8, "borehole_high_fidelity")
    rw = x_arr[:, 0]
    r = x_arr[:, 1]
    tu = x_arr[:, 2]
    hu = x_arr[:, 3]
    tl = x_arr[:, 4]
    hl = x_arr[:, 5]
    l = x_arr[:, 6]
    kw = x_arr[:, 7]
    log_term = np.log(r / rw)
    numerator = 2.0 * np.pi * tu * (hu - hl)
    denominator = log_term * (1.0 + 2.0 * l * tu / (log_term * rw ** 2 * kw) + tu / tl)
    return (numerator / denominator).reshape(-1, 1)


def borehole_low_fidelity(x: np.ndarray) -> np.ndarray:
    """
    Evaluate a biased Borehole low-fidelity benchmark.

    Args:
        x (np.ndarray): Query points. (N, 8) or (8,).

    Returns:
        np.ndarray: Responses. (N, 1).
    """
    x_arr = _as_2d_array(x, 8, "borehole_low_fidelity")
    hf = borehole_high_fidelity(x_arr)[:, 0]
    y = 0.92 * hf + 0.15 * np.sin(x_arr[:, 1] / 5000.0) + 0.08 * np.cos(x_arr[:, 3] / 120.0)
    return y.reshape(-1, 1)


MULTI_FIDELITY_BENCHMARKS: Dict[str, MultiFidelityBenchmark] = {
    "borehole": MultiFidelityBenchmark(
        name="borehole",
        input_dim=8,
        bounds=(
            (0.05, 0.15),
            (100.0, 50000.0),
            (63070.0, 115600.0),
            (990.0, 1110.0),
            (63.1, 116.0),
            (700.0, 820.0),
            (1120.0, 1680.0),
            (9855.0, 12045.0),
        ),
        output_name="response",
        description="Standard Borehole benchmark with a biased low-fidelity approximation.",
        high_fidelity=borehole_high_fidelity,
        low_fidelity=borehole_low_fidelity,
    ),
    "currin_exponential": MultiFidelityBenchmark(
        name="currin_exponential",
        input_dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Currin exponential high-/low-fidelity benchmark pair.",
        high_fidelity=currin_exponential,
        low_fidelity=currin_exponential_low_fidelity,
    ),
    "branin": MultiFidelityBenchmark(
        name="branin",
        input_dim=2,
        bounds=((-5.0, 10.0), (0.0, 15.0)),
        output_name="response",
        description="Branin benchmark with a biased low-fidelity approximation.",
        high_fidelity=branin,
        low_fidelity=branin_low_fidelity,
    ),
    "park91b": MultiFidelityBenchmark(
        name="park91b",
        input_dim=4,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Park91B benchmark with a linear-scaled low-fidelity model.",
        high_fidelity=park91b,
        low_fidelity=park91b_low_fidelity,
    ),
}


MULTI_OBJECTIVE_BENCHMARKS: Dict[str, MultiObjectiveBenchmark] = {
    "vlmop2": MultiObjectiveBenchmark(
        name="vlmop2",
        input_dim=2,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        output_names=("f1", "f2"),
        description="Two-objective VLMOP2 benchmark.",
        evaluator=vlmop2,
    ),
}


def get_scalar_benchmark(name: str) -> ScalarBenchmark:
    """
    Fetch a scalar benchmark by name.

    Args:
        name (str): Benchmark name.

    Returns:
        ScalarBenchmark: Requested scalar benchmark.
    """
    key = name.lower()
    if key not in SCALAR_BENCHMARKS:
        raise KeyError(f"Unknown scalar benchmark: '{name}'.")
    return SCALAR_BENCHMARKS[key]


def get_multifidelity_benchmark(name: str) -> MultiFidelityBenchmark:
    """
    Fetch a multi-fidelity benchmark by name.

    Args:
        name (str): Benchmark name.

    Returns:
        MultiFidelityBenchmark: Requested multi-fidelity benchmark.
    """
    key = name.lower()
    if key not in MULTI_FIDELITY_BENCHMARKS:
        raise KeyError(f"Unknown multi-fidelity benchmark: '{name}'.")
    return MULTI_FIDELITY_BENCHMARKS[key]


def get_multiobjective_benchmark(name: str) -> MultiObjectiveBenchmark:
    """
    Fetch a multi-objective benchmark by name.

    Args:
        name (str): Benchmark name.

    Returns:
        MultiObjectiveBenchmark: Requested multi-objective benchmark.
    """
    key = name.lower()
    if key not in MULTI_OBJECTIVE_BENCHMARKS:
        raise KeyError(f"Unknown multi-objective benchmark: '{name}'.")
    return MULTI_OBJECTIVE_BENCHMARKS[key]
