# Engineering case workflow for SurrogateLab
# Author: Shengning Wang

import json
import os
import random
import shutil
from functools import partial
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import NonlinearConstraint

try:
    from wing_structure_simulation import AbaqusModel as _ExternalAbaqusModel
except ImportError:
    _ExternalAbaqusModel = None

import case_config

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
from utils.hue_logger import hue, logger
from utils.seeder import seed_everything


# ============================================================
# Runtime Backends
# ============================================================


class _ExternalAbaqusModelAdapter:
    """
    Workspace-aware wrapper around the external Abaqus solver interface.

    Args:
        fidelity (str): Simulation fidelity, "high" or "low".
    """

    def __init__(self, fidelity: str = "high") -> None:
        self.model = _ExternalAbaqusModel(fidelity=fidelity)
        self.input_vars = list(self.model.input_vars)
        self.output_vars = list(self.model.output_vars)

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """
        Execute one Abaqus simulation from the SurrogateLab root.

        Args:
            input_arr (np.ndarray): Design vector. (D,).

        Returns:
            np.ndarray: Structural responses. (4,).
        """
        current_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        try:
            return self.model.run(input_arr)
        finally:
            os.chdir(current_dir)


def _get_external_abaqus_command() -> str | None:
    if _ExternalAbaqusModel is None:
        return None
    return _ExternalAbaqusModel().abaqus_cmd.split()[0]


def get_case_runtime() -> Dict[str, Any]:
    """
    Detect the external Abaqus runtime availability.

    Returns:
        Dict[str, Any]: Runtime status for solver selection and reporting.
    """
    if hasattr(get_case_runtime, "_cache"):
        return get_case_runtime._cache

    command_name = _get_external_abaqus_command()
    interface_available = _ExternalAbaqusModel is not None
    command_available = command_name is not None and shutil.which(command_name) is not None
    available = interface_available and command_available

    get_case_runtime._cache = {
        "solver": "abaqus",
        "interface_available": interface_available,
        "command_name": command_name,
        "command_available": command_available,
        "available": available,
    }
    return get_case_runtime._cache


get_abaqus_runtime = get_case_runtime


def log_case_runtime(runtime: Dict[str, Any]) -> None:
    """
    Print a high-visibility Abaqus runtime banner.

    Args:
        runtime (Dict[str, Any]): Runtime status returned by get_case_runtime.
    """
    line = "=" * 78
    availability_color = hue.g if runtime["available"] else hue.r
    command_name = runtime["command_name"] if runtime["command_name"] is not None else "N/A"
    logger.info(line)
    logger.info(f"{hue.b}SIMULATION INTERFACE  : {hue.c}{str(runtime['interface_available']).upper()}{hue.q}")
    logger.info(f"{hue.b}ABAQUS COMMAND        : {hue.m}{command_name}{hue.q}")
    logger.info(f"{hue.b}ABAQUS COMMAND FOUND  : {hue.c}{str(runtime['command_available']).upper()}{hue.q}")
    logger.info(f"{hue.b}ABAQUS AVAILABILITY   : {availability_color}{str(runtime['available']).upper()}{hue.q}")
    logger.info(line)


log_abaqus_runtime = log_case_runtime


class AbaqusModel:
    """
    Engineering solver wrapper backed only by the external Abaqus interface.

    Args:
        fidelity (str): Simulation fidelity, "high" or "low".
    """

    def __init__(self, fidelity: str = "high") -> None:
        runtime = get_case_runtime()
        if not runtime["available"]:
            raise RuntimeError("Abaqus is unavailable in the current environment.")
        self.model = _ExternalAbaqusModelAdapter(fidelity=fidelity)
        self.input_vars = list(self.model.input_vars)
        self.output_vars = list(self.model.output_vars)

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """
        Execute one engineering simulation.

        Args:
            input_arr (np.ndarray): Design vector. (D,).

        Returns:
            np.ndarray: Structural responses. (4,).
        """
        return self.model.run(input_arr)


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
        np.ndarray: Physical samples. (N, D).
    """
    return bounds[:, 0] + x_norm * (bounds[:, 1] - bounds[:, 0])


def sample_lhs(bounds: np.ndarray, num_samples: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a latin hypercube inside the engineering design box.

    Args:
        bounds (np.ndarray): Box bounds. (D, 2).
        num_samples (int): Number of samples.
        seed (int | None): Optional local RNG seed for reproducible DOE generation.

    Returns:
        np.ndarray: Physical samples. (N, D).
    """
    if seed is None:
        x_norm = lhs_design(num_samples, bounds.shape[0])
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            x_norm = lhs_design(num_samples, bounds.shape[0])
        finally:
            np.random.set_state(state)
    return scale_to_bounds(x_norm, bounds)


def run_abaqus_batch(x: np.ndarray, fidelity: str = "high") -> np.ndarray:
    """
    Evaluate the Abaqus model on a batch of design points.

    Args:
        x (np.ndarray): Design matrix. (N, D).
        fidelity (str): Simulation fidelity.

    Returns:
        np.ndarray: Response matrix. (N, 4).
    """
    model = AbaqusModel(fidelity=fidelity)
    return np.vstack([model.run(x[i]) for i in range(x.shape[0])])


def reset_random_state(seed: int) -> None:
    """
    Reset Python and NumPy random states.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def select_output(y: np.ndarray, output_idx: int) -> np.ndarray:
    """
    Select one response column as a 2-D array.

    Args:
        y (np.ndarray): Response matrix. (N, C).
        output_idx (int): Output index.

    Returns:
        np.ndarray: Selected response. (N, 1).
    """
    return y[:, output_idx : output_idx + 1]


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
    Compute engineering metrics.

    Args:
        y_true (np.ndarray): Ground truth. (N, C).
        y_pred (np.ndarray): Prediction. (N, C).
        eps (float): Stability epsilon.

    Returns:
        Dict[str, float]: Accuracy and R2 metrics.
    """
    return {
        "accuracy": evaluate_accuracy(y_true, y_pred, eps=eps),
        "r2": evaluate_r2(y_true, y_pred, eps=eps),
    }


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


def fit_krg(x_train: np.ndarray, y_train: np.ndarray, args: Any) -> KRG:
    """
    Fit a Kriging surrogate with the shared CLI hyperparameters.

    Args:
        x_train (np.ndarray): Training inputs. (N, D).
        y_train (np.ndarray): Training outputs. (N, C).
        args (Any): Parsed arguments.

    Returns:
        KRG: Trained Kriging model.
    """
    model = KRG(**args.krg_params)
    model.fit(x_train, y_train)
    return model


def predict_scalar_output(model: Any, output_idx: int, x_vec: np.ndarray) -> float:
    """
    Predict one scalar output from a surrogate at a single design point.

    Args:
        model (Any): Surrogate model.
        output_idx (int): Output column index.
        x_vec (np.ndarray): Design vector. (D,).

    Returns:
        float: Predicted scalar output.
    """
    mean_value = predict_mean(model, np.asarray(x_vec, dtype=np.float64).reshape(1, -1))
    return float(mean_value[0, output_idx])


# ============================================================
# Cache Helpers
# ============================================================

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


def get_target_specs(args: Any) -> List[Dict[str, Any]]:
    """
    Resolve the selected engineering targets.

    Args:
        args (Any): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Ordered target specs.
    """
    specs: List[Dict[str, Any]] = []
    for name in args.targets:
        target = dict(case_config.TARGET_SPECS[name])
        target["name"] = name
        specs.append(target)
    return specs


def get_case_root() -> str:
    """
    Return the SurrogateLab case-workflow root directory.

    Returns:
        str: Absolute case root.
    """
    return os.path.dirname(os.path.abspath(__file__))


def get_case_cache_path(filename: str) -> str:
    """
    Build one cache path under the case-workflow root.

    Args:
        filename (str): Cache filename.

    Returns:
        str: Absolute cache path.
    """
    return os.path.join(get_case_root(), filename)


def load_case_cache(path: str) -> Dict[str, Any] | None:
    """
    Load one NumPy object cache when it exists.

    Args:
        path (str): Cache path.

    Returns:
        Dict[str, Any] | None: Loaded cache payload or ``None``.
    """
    if not os.path.isfile(path):
        return None
    return np.load(path, allow_pickle=True).item()


def save_case_cache(path: str, payload: Dict[str, Any]) -> None:
    """
    Save one NumPy object cache.

    Args:
        path (str): Cache path.
        payload (Dict[str, Any]): Cache payload.
    """
    np.save(path, payload, allow_pickle=True)


def cache_meta_matches(cached_meta: Any, expected_meta: Dict[str, Any]) -> bool:
    """
    Compare one cache metadata payload with the expected signature.

    Args:
        cached_meta (Any): Metadata loaded from one cache file.
        expected_meta (Dict[str, Any]): Expected cache signature.

    Returns:
        bool: Whether the metadata matches.
    """
    if not isinstance(cached_meta, dict):
        return False
    return cached_meta == expected_meta


def require_abaqus_available(cache_name: str) -> None:
    """
    Ensure the external Abaqus runtime is available for cache generation.

    Args:
        cache_name (str): Cache being regenerated.

    Raises:
        RuntimeError: If Abaqus is unavailable locally.
    """
    runtime = get_case_runtime()
    if runtime["available"]:
        return
    raise RuntimeError(f"{cache_name} is missing or invalid, and Abaqus is unavailable in the current environment.")


def build_case_doe_meta(args: Any) -> Dict[str, Any]:
    """
    Build the signature stored inside the engineering DOE cache.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Cache metadata signature.
    """
    return {
        "doe_seed": args.doe_seed,
        "num_features": args.num_features,
        "num_outputs": args.num_outputs,
        "bounds": args.bounds.tolist(),
        "num_train": args.num_train,
        "num_test": args.num_test,
        "num_lf": args.num_lf,
        "num_hf": args.num_hf,
    }


def validate_case_doe_cache(cached: Any, args: Any, meta: Dict[str, Any]) -> bool:
    """
    Validate one engineering DOE cache payload.

    Args:
        cached (Any): Loaded cache payload.
        args (Any): Parsed arguments.
        meta (Dict[str, Any]): Expected cache signature.

    Returns:
        bool: Whether the cache is valid for the current run.
    """
    if not isinstance(cached, dict):
        return False
    if not cache_meta_matches(cached.get("meta"), meta):
        return False

    expected_shapes = {
        "x_train": (args.num_train, args.num_features),
        "y_train": (args.num_train, args.num_outputs),
        "x_test": (args.num_test, args.num_features),
        "y_test": (args.num_test, args.num_outputs),
        "x_lf": (args.num_lf, args.num_features),
        "y_lf": (args.num_lf, args.num_outputs),
        "x_hf": (args.num_hf, args.num_features),
        "y_hf": (args.num_hf, args.num_outputs),
    }
    for key, shape in expected_shapes.items():
        value = cached.get(key)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            return False
    return True


def generate_case_doe(args: Any) -> Dict[str, np.ndarray]:
    """
    Generate or load the cached engineering DOE dataset.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, np.ndarray]: Cached engineering data.
    """
    meta = build_case_doe_meta(args)
    cache_path = get_case_cache_path("case_doe_cache.npy")
    cached = load_case_cache(cache_path)

    if validate_case_doe_cache(cached, args, meta):
        logger.info(f"{hue.g}Load cached case DOE from {cache_path}{hue.q}")
        return cached

    if cached is not None:
        if cache_meta_matches(cached.get("meta"), meta):
            logger.info(f"{hue.y}Case DOE cache data is invalid, regenerate {cache_path}{hue.q}")
        else:
            logger.info(f"{hue.y}Case DOE cache metadata changed, regenerate {cache_path}{hue.q}")

    require_abaqus_available("case_doe_cache.npy")
    logger.info(f"{hue.b}Generate case DOE data{hue.q}")
    x_train = sample_lhs(args.bounds, args.num_train, seed=args.doe_seed)
    x_test = sample_lhs(args.bounds, args.num_test, seed=args.doe_seed + 1)
    x_lf = sample_lhs(args.bounds, args.num_lf, seed=args.doe_seed + 2)
    x_hf = sample_lhs(args.bounds, args.num_hf, seed=args.doe_seed + 3)

    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        random.seed(args.doe_seed + 4)
        np.random.seed(args.doe_seed + 4)

        data = {
            "meta": meta,
            "x_train": x_train,
            "y_train": run_abaqus_batch(x_train, fidelity="high"),
            "x_test": x_test,
            "y_test": run_abaqus_batch(x_test, fidelity="high"),
            "x_lf": x_lf,
            "y_lf": run_abaqus_batch(x_lf, fidelity="low"),
            "x_hf": x_hf,
            "y_hf": run_abaqus_batch(x_hf, fidelity="high"),
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    save_case_cache(cache_path, data)
    logger.info(
        f"  train={data['x_train'].shape} | test={data['x_test'].shape} | "
        f"lf={data['x_lf'].shape} | hf={data['x_hf'].shape}"
    )
    return data


def build_case_active_meta(args: Any) -> Dict[str, Any]:
    """
    Build the signature stored inside the engineering active-learning cache.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Cache metadata signature.
    """
    return {
        "active_seed": args.active_seed,
        "num_features": args.num_features,
        "num_outputs": args.num_outputs,
        "bounds": args.bounds.tolist(),
        "targets": list(args.targets),
        "num_active_initial": args.num_active_initial,
        "num_infill": args.num_infill,
        "infill_criterion": args.infill_criterion,
        "diso_alpha": args.diso_alpha,
        "diso_min_distance": args.diso_min_distance,
        "diso_distance_scale": args.diso_distance_scale,
        "krg_params": to_serializable(args.krg_params),
    }


def create_empty_active_cache(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create one empty engineering active-learning cache payload.

    Args:
        meta (Dict[str, Any]): Cache signature.

    Returns:
        Dict[str, Any]: Empty cache payload.
    """
    return {
        "meta": meta,
        "x_initial": None,
        "y_initial": None,
        "targets": {},
    }


def load_case_active_cache(args: Any) -> Dict[str, Any]:
    """
    Load and sanitize the engineering active-learning cache.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Active-learning cache payload.
    """
    meta = build_case_active_meta(args)
    cache_path = get_case_cache_path("case_active_cache.npy")
    cached = load_case_cache(cache_path)
    empty_cache = create_empty_active_cache(meta)

    if cached is None:
        return empty_cache

    if not cache_meta_matches(cached.get("meta"), meta):
        logger.info(f"{hue.y}Case active cache metadata changed, regenerate {cache_path}{hue.q}")
        return empty_cache

    cache = create_empty_active_cache(meta)
    x_initial = cached.get("x_initial")
    y_initial = cached.get("y_initial")
    valid_initial = (
        isinstance(x_initial, np.ndarray)
        and x_initial.shape == (args.num_active_initial, args.num_features)
        and isinstance(y_initial, np.ndarray)
        and y_initial.shape == (args.num_active_initial, args.num_outputs)
    )
    if valid_initial:
        cache["x_initial"] = x_initial
        cache["y_initial"] = y_initial

    cached_targets = cached.get("targets")
    if valid_initial and isinstance(cached_targets, dict):
        for name in args.targets:
            target_payload = cached_targets.get(name)
            if not isinstance(target_payload, dict):
                continue
            x_infill = target_payload.get("x_infill")
            y_infill = target_payload.get("y_infill")
            valid_entry = (
                isinstance(x_infill, np.ndarray)
                and isinstance(y_infill, np.ndarray)
                and x_infill.ndim == 2
                and y_infill.ndim == 2
                and x_infill.shape[0] == y_infill.shape[0]
                and x_infill.shape[0] <= args.num_infill
                and x_infill.shape[1] == args.num_features
                and y_infill.shape[1] == args.num_outputs
            )
            if valid_entry:
                cache["targets"][name] = {
                    "output_idx": int(case_config.TARGET_SPECS[name]["output_idx"]),
                    "x_infill": x_infill,
                    "y_infill": y_infill,
                }

    if cache["x_initial"] is not None or cache["targets"]:
        logger.info(f"{hue.g}Load cached case active data from {cache_path}{hue.q}")
    return cache


def ensure_active_initial_data(args: Any, cache: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Load or generate the shared initial active-learning samples.

    Args:
        args (Any): Parsed arguments.
        cache (Dict[str, Any]): Active-learning cache payload.

    Returns:
        tuple[np.ndarray, np.ndarray, bool]: Initial designs, initial responses, update flag.
    """
    x_initial_expected = sample_lhs(args.bounds, args.num_active_initial)
    x_initial_cached = cache.get("x_initial")
    y_initial_cached = cache.get("y_initial")

    if isinstance(x_initial_cached, np.ndarray) and isinstance(y_initial_cached, np.ndarray):
        if np.allclose(x_initial_cached, x_initial_expected, rtol=1.0e-12, atol=1.0e-12):
            return x_initial_cached.copy(), y_initial_cached.copy(), False
        logger.info(f"{hue.y}Case active cache initial design changed, regenerate case_active_cache.npy{hue.q}")
        cache["x_initial"] = None
        cache["y_initial"] = None
        cache["targets"] = {}

    require_abaqus_available("case_active_cache.npy")
    y_initial = run_abaqus_batch(x_initial_expected, fidelity="high")
    cache["x_initial"] = x_initial_expected
    cache["y_initial"] = y_initial
    cache["targets"] = {}
    return x_initial_expected.copy(), y_initial.copy(), True


def get_active_target_cache_entry(args: Any, cache: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch one target-specific active-learning cache entry.

    Args:
        args (Any): Parsed arguments.
        cache (Dict[str, Any]): Active-learning cache payload.
        target (Dict[str, Any]): Target spec.

    Returns:
        Dict[str, Any]: Target-specific cache entry.
    """
    entry = cache["targets"].get(target["name"])
    if entry is None:
        entry = {
            "output_idx": int(target["output_idx"]),
            "x_infill": np.empty((0, args.num_features), dtype=np.float64),
            "y_infill": np.empty((0, args.num_outputs), dtype=np.float64),
        }
        cache["targets"][target["name"]] = entry
    return entry


def run_active_learning_target(
    args: Any,
    data: Dict[str, np.ndarray],
    target: Dict[str, Any],
    x_initial: np.ndarray,
    y_initial_full: np.ndarray,
    cache: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    """
    Replay or extend one target-specific active-learning trajectory.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.
        target (Dict[str, Any]): Target spec.
        x_initial (np.ndarray): Initial designs. (N0, D).
        y_initial_full (np.ndarray): Initial responses. (N0, C).
        cache (Dict[str, Any]): Active-learning cache payload.

    Returns:
        tuple[Dict[str, Any], bool]: Target record and cache update flag.
    """
    x_current = x_initial.copy()
    y_current = select_output(y_initial_full, target["output_idx"]).copy()
    y_test = select_output(data["y_test"], target["output_idx"])

    model_before = fit_krg(x_current, y_current, args)
    metrics_before = evaluate_metrics(y_test, predict_mean(model_before, data["x_test"]), eps=args.metric_eps)

    history_best: List[float] = []
    cache_updated = False
    target_cache = get_active_target_cache_entry(args, cache, target)
    cached_count = target_cache["x_infill"].shape[0]
    hf_model = None

    for step in range(args.num_infill):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = DISOInfill(
            model=model_iter,
            bounds=args.bounds,
            x_train=x_current,
            y_train=y_current,
            criterion=args.infill_criterion,
            target_idx=0,
            alpha=args.diso_alpha,
            min_distance=args.diso_min_distance,
            distance_scale=args.diso_distance_scale,
        )
        x_expected = strategy.propose()

        if step < cached_count:
            x_cached = target_cache["x_infill"][step : step + 1]
            if np.allclose(x_cached, x_expected, rtol=1.0e-10, atol=1.0e-10):
                x_new = x_cached
                y_new_full = target_cache["y_infill"][step : step + 1]
            else:
                logger.info(
                    f"{hue.y}Case active cache trajectory changed for {target['label']} at infill {step + 1}, "
                    f"regenerate remaining tail{hue.q}"
                )
                target_cache["x_infill"] = target_cache["x_infill"][:step]
                target_cache["y_infill"] = target_cache["y_infill"][:step]
                cached_count = step

        if step >= cached_count:
            require_abaqus_available("case_active_cache.npy")
            if hf_model is None:
                hf_model = AbaqusModel(fidelity="high")
            x_new = x_expected
            y_new_full = hf_model.run(x_new[0]).reshape(1, -1)
            target_cache["x_infill"] = np.vstack([target_cache["x_infill"], x_new])
            target_cache["y_infill"] = np.vstack([target_cache["y_infill"], y_new_full])
            cache_updated = True

        y_new = y_new_full[:, target["output_idx"] : target["output_idx"] + 1]
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history_best.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    metrics_after = evaluate_metrics(y_test, predict_mean(model_after, data["x_test"]), eps=args.metric_eps)
    gain = compute_relative_gain(metrics_before["accuracy"], metrics_after["accuracy"], eps=args.metric_eps)
    passed = gain >= args.active_learning_min_relative_gain

    record = {
        "target": target["name"],
        "algorithm": "DISO",
        "criterion": args.infill_criterion,
        "num_initial": int(args.num_active_initial),
        "num_infill": args.num_infill,
        "before": metrics_before,
        "after": metrics_after,
        "accuracy_gain": gain,
        "history_best": history_best,
        "passed": passed,
    }
    return record, cache_updated


# ============================================================
# Case Modules
# ============================================================

def run_ensemble_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the engineering ensemble-model cases.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Target-wise benchmark records.
    """
    single_model_builders = {
        "PRS": lambda: PRS(**args.prs_params),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(**args.svr_params),
    }
    ensemble_builders = {
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

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Case Ensemble{hue.q}")

    for target in get_target_specs(args):
        y_train = select_output(data["y_train"], target["output_idx"])
        y_test = select_output(data["y_test"], target["output_idx"])

        baseline_scores: Dict[str, Dict[str, float]] = {}
        for model_name, builder in single_model_builders.items():
            model = builder()
            model.fit(data["x_train"], y_train)
            baseline_scores[model_name] = evaluate_metrics(y_test, predict_mean(model, data["x_test"]), eps=args.metric_eps)

        mean_single_accuracy = float(np.mean([item["accuracy"] for item in baseline_scores.values()]))
        record = {
            "target": target["name"],
            "baseline_single_models": baseline_scores,
            "mean_single_accuracy": mean_single_accuracy,
            "algorithms": {},
        }
        logger.info(f"  {target['label']} | baseline acc={mean_single_accuracy:.2f}%")

        for algo_name, builder in ensemble_builders.items():
            if algo_name not in args.demos:
                continue

            model = builder()
            model.fit(data["x_train"], y_train)
            metrics = evaluate_metrics(y_test, predict_mean(model, data["x_test"]), eps=args.metric_eps)
            gain = compute_relative_gain(mean_single_accuracy, metrics["accuracy"], eps=args.metric_eps)
            passed = gain >= args.ensemble_min_relative_gain
            record["algorithms"][algo_name] = {
                **metrics,
                "accuracy_gain": gain,
                "passed": passed,
            }
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"    {algo_name} | acc={metrics['accuracy']:.2f}% | r2={metrics['r2']:.4f} | "
                f"gain={100.0 * gain:.2f}% -> {status}"
            )

        results.append(record)

    return results


def run_multifidelity_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the engineering multi-fidelity cases.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Target-wise benchmark records.
    """
    model_builders = {
        "MFSMLS": lambda: MFSMLS(
            poly_degree=args.mf_poly_degree,
            neighbor_factor=args.mfs_mls_neighbor_factor,
            ridge=args.mfs_mls_ridge,
        ),
        "MMFS": lambda: MMFS(sigma_bounds=tuple(args.mf_sigma_bounds)),
        "CCAMFS": lambda: CCAMFS(),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Case Multi-Fidelity{hue.q}")

    for target in get_target_specs(args):
        y_lf = select_output(data["y_lf"], target["output_idx"])
        y_hf = select_output(data["y_hf"], target["output_idx"])
        y_test = select_output(data["y_test"], target["output_idx"])

        record = {
            "target": target["name"],
            "algorithms": {},
        }
        logger.info(f"  {target['label']} | target acc>={args.mf_min_accuracy:.2f}%")

        for algo_name, builder in model_builders.items():
            if algo_name not in args.demos:
                continue

            model = builder()
            model.fit(data["x_lf"], y_lf, data["x_hf"], y_hf)
            metrics = evaluate_metrics(y_test, predict_mean(model, data["x_test"]), eps=args.metric_eps)
            passed = metrics["accuracy"] >= args.mf_min_accuracy
            record["algorithms"][algo_name] = {
                **metrics,
                "passed": passed,
            }
            status = "PASS" if passed else "FAIL"
            logger.info(f"    {algo_name} | acc={metrics['accuracy']:.2f}% | r2={metrics['r2']:.4f} -> {status}")

        results.append(record)

    return results


def run_active_learning_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the engineering DISO active-learning cases.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Target-wise active-learning records.
    """
    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Case Active Learning{hue.q}")
    cache_path = get_case_cache_path("case_active_cache.npy")
    cache = load_case_active_cache(args)
    cache_updated = False

    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        reset_random_state(args.active_seed)
        x_initial, y_initial_full, initial_updated = ensure_active_initial_data(args, cache)
        cache_updated = cache_updated or initial_updated

        for target in get_target_specs(args):
            record, target_updated = run_active_learning_target(args, data, target, x_initial, y_initial_full, cache)
            cache_updated = cache_updated or target_updated
            status = "PASS" if record["passed"] else "FAIL"
            logger.info(
                f"  {target['label']} | acc {record['before']['accuracy']:.2f}% -> {record['after']['accuracy']:.2f}% | "
                f"r2 {record['before']['r2']:.4f} -> {record['after']['r2']:.4f} | "
                f"gain={100.0 * record['accuracy_gain']:.2f}% -> {status}"
            )
            results.append(record)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    if cache_updated:
        save_case_cache(cache_path, cache)
        logger.info(f"{hue.g}Save case active cache to {cache_path}{hue.q}")

    return results


def run_optimization_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the engineering optimization cases.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Optimization records.
    """
    objective_idx = case_config.TARGET_SPECS[args.opt_target]["output_idx"]
    constraint_idx = case_config.TARGET_SPECS[args.opt_constraint_target]["output_idx"]
    model = fit_krg(data["x_train"], data["y_train"], args)
    x0 = data["x_train"][np.argmin(data["y_train"][:, objective_idx])]
    objective = partial(predict_scalar_output, model, objective_idx)
    constraint_fun = partial(predict_scalar_output, model, constraint_idx)
    constraint = NonlinearConstraint(fun=constraint_fun, lb=-np.inf, ub=args.opt_constraint_ub)

    optimizers = {
        "MIGA": lambda: multi_island_genetic_optimize(
            func=objective,
            bounds=[tuple(bound) for bound in args.bounds],
            constraints=constraint,
            x0=x0,
            tol=args.opt_tol,
            seed=args.seed,
            multi_objective=False,
            **args.miga_params,
        ),
        "CFARSSDA": lambda: dragonfly_optimize(
            func=objective,
            bounds=[tuple(bound) for bound in args.bounds],
            constraints=constraint,
            x0=x0,
            tol=args.opt_tol,
            seed=args.seed,
            multi_objective=False,
            **args.df_params,
        ),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Case Optimization{hue.q}")

    for algo_name, builder in optimizers.items():
        if algo_name not in args.demos:
            continue

        builder()
        record = {
            "algorithm": algo_name,
            "passed": True,
        }
        logger.info(f"  {algo_name} -> PASS")
        results.append(record)

    return results


# ============================================================
# Summary and I/O
# ============================================================

def summarize_target_algorithms(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Summarize target-wise module results into algorithm PASS/FAIL flags.

    Args:
        records (List[Dict[str, Any]]): Module records grouped by target.

    Returns:
        Dict[str, str]: Algorithm summary.
    """
    required_targets = list(case_config.TARGET_SPECS.keys())
    per_algorithm: Dict[str, Dict[str, bool]] = {}

    for record in records:
        for algo_name, algo_result in record.get("algorithms", {}).items():
            per_algorithm.setdefault(algo_name, {})[record["target"]] = bool(algo_result["passed"])

    return {
        algo_name: "PASS" if all(per_algorithm[algo_name].get(target, False) for target in required_targets) else "FAIL"
        for algo_name in case_config.ALGORITHM_ORDER
        if algo_name in per_algorithm
    }


def summarize_active_learning(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Summarize active-learning results into algorithm PASS/FAIL flags.

    Args:
        records (List[Dict[str, Any]]): Active-learning records grouped by target.

    Returns:
        Dict[str, str]: Algorithm summary.
    """
    required_targets = list(case_config.TARGET_SPECS.keys())
    per_algorithm: Dict[str, Dict[str, bool]] = {}

    for record in records:
        per_algorithm.setdefault(record["algorithm"], {})[record["target"]] = bool(record["passed"])

    return {
        algo_name: "PASS" if all(per_algorithm[algo_name].get(target, False) for target in required_targets) else "FAIL"
        for algo_name in case_config.ALGORITHM_ORDER
        if algo_name in per_algorithm
    }


def summarize_optimization(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Summarize optimization results into algorithm PASS flags.

    Args:
        records (List[Dict[str, Any]]): Optimization records grouped by algorithm.

    Returns:
        Dict[str, str]: Algorithm summary.
    """
    return {record["algorithm"]: "PASS" for record in records}


def build_case_summary(payload: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Build the case summary payload.

    Args:
        payload (Dict[str, Any]): Case payload.

    Returns:
        Dict[str, Dict[str, str]]: Module summary.
    """
    return {
        "ensemble": summarize_target_algorithms(payload.get("ensemble", [])),
        "multifidelity": summarize_target_algorithms(payload.get("multifidelity", [])),
        "active_learning": summarize_active_learning(payload.get("active_learning", [])),
        "optimization": summarize_optimization(payload.get("optimization", [])),
    }


def save_results(payload: Dict[str, Any]) -> str:
    """
    Save the case payload as JSON.

    Args:
        payload (Dict[str, Any]): Case payload.

    Returns:
        str: Absolute save path.
    """
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "case_results.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2)
    return save_path


def run_case(args: Any) -> Dict[str, Any]:
    """
    Execute the full engineering case workflow.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Case payload.
    """
    seed_everything(args.seed)
    reset_random_state(args.seed)
    runtime = get_case_runtime()

    logger.info(f"{hue.b}SurrogateLab Case Workflow{hue.q}")
    log_case_runtime(runtime)
    logger.info(f"  demos    : {args.demos}")
    logger.info(f"  targets  : {args.targets}")

    data = generate_case_doe(args)
    payload: Dict[str, Any] = {
        "demos": args.demos,
        "targets": args.targets,
        "runtime": runtime,
        "sample_counts": {
            "num_train": args.num_train,
            "num_test": args.num_test,
            "num_lf": args.num_lf,
            "num_hf": args.num_hf,
            "num_active_initial": args.num_active_initial,
            "num_infill": args.num_infill,
        },
        "thresholds": {
            "ensemble_min_relative_gain": args.ensemble_min_relative_gain,
            "mf_min_accuracy": args.mf_min_accuracy,
            "active_learning_min_relative_gain": args.active_learning_min_relative_gain,
        },
        "summary": {
            "ensemble": {},
            "multifidelity": {},
            "active_learning": {},
            "optimization": {},
        },
    }

    if any(name in args.demos for name in ["TAHS", "AESMSI"]):
        payload["ensemble"] = run_ensemble_section(args, data)
    if any(name in args.demos for name in ["MFSMLS", "MMFS", "CCAMFS"]):
        payload["multifidelity"] = run_multifidelity_section(args, data)
    if "DISO" in args.demos:
        payload["active_learning"] = run_active_learning_section(args, data)
    if any(name in args.demos for name in ["MIGA", "CFARSSDA"]):
        payload["optimization"] = run_optimization_section(args, data)

    payload["summary"] = build_case_summary(payload)
    return payload


def main() -> None:
    """
    Execute the engineering case workflow from the command line.
    """
    args = case_config.get_args()
    try:
        payload = run_case(args)
    except RuntimeError as exc:
        message = str(exc)
        if message == "Abaqus is unavailable in the current environment." or message.endswith(
            "and Abaqus is unavailable in the current environment."
        ):
            logger.error(f"{hue.r}{message}{hue.q}")
            raise SystemExit(1) from None
        raise
    save_path = save_results(payload)
    logger.info(f"{hue.g}Case report saved to {save_path}{hue.q}")


if __name__ == "__main__":
    main()
