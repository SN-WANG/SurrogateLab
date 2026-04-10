# Analytic benchmark runner for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations
import json
import os
import random
from statistics import mean
from typing import Any, Callable, Dict, List

import numpy as np

import bench_config
import bench_funcs

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


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE_NAME = "bench_results.json"
RESULT_FILE_PATH = os.path.join(PROJECT_DIR, RESULT_FILE_NAME)


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


def sample_lhs(bounds: np.ndarray, num_samples: int, lhs_iterations: int) -> np.ndarray:
    """
    Generate a maximin Latin hypercube inside a bounded design space.

    Args:
        bounds (np.ndarray): Box bounds. (D, 2).
        num_samples (int): Number of samples.
        lhs_iterations (int): Maximin search iterations.

    Returns:
        np.ndarray: Scaled samples. (N, D).
    """
    x_norm = lhs_design(num_samples, bounds.shape[0], iterations=lhs_iterations)
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


def fit_krg(x_train: np.ndarray, y_train: np.ndarray, args: Any) -> KRG:
    """
    Fit a Kriging surrogate with shared CLI hyperparameters.

    Args:
        x_train (np.ndarray): Training inputs. (N, D).
        y_train (np.ndarray): Training targets. (N, C).
        args (Any): Parsed arguments.

    Returns:
        KRG: Trained Kriging model.
    """
    model = KRG(**args.krg_params)
    model.fit(x_train, y_train)
    return model


def build_mfs_mls_model(args: Any) -> MFSMLS:
    """
    Build the MFS-MLS model with the current workspace signature.

    Args:
        args (Any): Parsed arguments.

    Returns:
        MFSMLS: Configured MFS-MLS model.
    """
    return MFSMLS(
        poly_degree=1,
        neighbor_factor=args.mfs_mls_neighbor_factor,
        ridge=args.mfs_mls_ridge,
    )


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


def run_ensemble_section(args: Any) -> List[Dict[str, Any]]:
    """
    Run the ensemble surrogate benchmarks.

    Args:
        args (Any): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Per-case benchmark records.
    """
    single_model_builders: Dict[str, Callable[[], Any]] = {
        "PRS": lambda: PRS(),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(),
    }
    ensemble_builders: Dict[str, Callable[[], Any]] = {
        "TAHS": lambda: TAHS(threshold=args.ensemble_threshold, krg_params=args.krg_params),
        "AESMSI": lambda: AESMSI(threshold=args.ensemble_threshold, krg_params=args.krg_params),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Ensemble Benchmarks{hue.q}")

    for case_name in args.ensemble_cases:
        reset_random_state(args.seed)
        spec = bench_funcs.get_scalar_benchmark(case_name)
        config = bench_config.DEFAULT_ENSEMBLE_CASES[case_name]
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        bounds = spec.bounds_array

        x_train = sample_lhs(bounds, config["num_train"], lhs_iterations)
        x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
        y_train = spec.evaluate(x_train)
        y_test = spec.evaluate(x_test)

        baseline_scores: Dict[str, Dict[str, float]] = {}
        for model_name, builder in single_model_builders.items():
            model = builder()
            model.fit(x_train, y_train)
            baseline_scores[model_name] = evaluate_metrics(y_test, predict_mean(model, x_test), eps=args.metric_eps)

        mean_single_accuracy = float(mean(item["accuracy"] for item in baseline_scores.values()))
        case_result = {
            "case": case_name,
            "input_dim": spec.input_dim,
            "num_train": config["num_train"],
            "num_test": config["num_test"],
            "single_models": baseline_scores,
            "mean_single_accuracy": mean_single_accuracy,
            "algorithms": {},
        }
        logger.info(f"  {spec.name}: baseline acc={hue.c}{mean_single_accuracy:.2f}%{hue.q}")

        for algo_name, builder in ensemble_builders.items():
            if algo_name not in args.demos:
                continue

            model = builder()
            model.fit(x_train, y_train)
            metrics = evaluate_metrics(y_test, predict_mean(model, x_test), eps=args.metric_eps)
            gain = compute_relative_gain(mean_single_accuracy, metrics["accuracy"], eps=args.metric_eps)
            passed = gain >= args.ensemble_min_relative_gain
            case_result["algorithms"][algo_name] = {**metrics, "accuracy_gain": gain, "passed": passed}
            logger.info(
                f"    {algo_name} | acc={metrics['accuracy']:.2f}% | r2={metrics['r2']:.4f} | "
                f"gain={100.0 * gain:.2f}% -> {'PASS' if passed else 'FAIL'}"
            )

        results.append(case_result)

    return results


def run_multifidelity_section(args: Any) -> List[Dict[str, Any]]:
    """
    Run the multi-fidelity surrogate benchmarks.

    Args:
        args (Any): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Per-case benchmark records.
    """
    model_builders: Dict[str, Callable[[], Any]] = {
        "MFSMLS": lambda: build_mfs_mls_model(args),
        "MMFS": lambda: MMFS(),
        "CCAMFS": lambda: CCAMFS(),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Multi-Fidelity Benchmarks{hue.q}")

    for case_name in args.multifidelity_cases:
        reset_random_state(args.seed)
        spec = bench_funcs.get_multifidelity_benchmark(case_name)
        config = bench_config.DEFAULT_MULTIFIDELITY_CASES[case_name]
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        bounds = spec.bounds_array

        x_lf = sample_lhs(bounds, config["num_lf"], lhs_iterations)
        x_hf = sample_lhs(bounds, config["num_hf"], lhs_iterations)
        x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
        y_lf = spec.evaluate_low_fidelity(x_lf)
        y_hf = spec.evaluate_high_fidelity(x_hf)
        y_test = spec.evaluate_high_fidelity(x_test)

        case_result = {
            "case": case_name,
            "input_dim": spec.input_dim,
            "num_lf": config["num_lf"],
            "num_hf": config["num_hf"],
            "num_test": config["num_test"],
            "algorithms": {},
        }
        logger.info(f"  {spec.name}: target acc>={args.mf_min_accuracy:.2f}%")

        for algo_name, builder in model_builders.items():
            if algo_name not in args.demos:
                continue

            model = builder()
            model.fit(x_lf, y_lf, x_hf, y_hf)
            metrics = evaluate_metrics(y_test, predict_mean(model, x_test), eps=args.metric_eps)
            passed = metrics["accuracy"] >= args.mf_min_accuracy
            case_result["algorithms"][algo_name] = {**metrics, "passed": passed}
            logger.info(
                f"    {algo_name} | acc={metrics['accuracy']:.2f}% | r2={metrics['r2']:.4f} -> "
                f"{'PASS' if passed else 'FAIL'}"
            )

        results.append(case_result)

    return results


def run_single_objective_active_case(args: Any) -> Dict[str, Any]:
    """
    Run the distance-informed single-objective active learning case.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Benchmark record.
    """
    config = bench_config.DEFAULT_SINGLE_OBJECTIVE_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = bench_funcs.get_scalar_benchmark(config["name"])
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    bounds = spec.bounds_array

    x_current = sample_lhs(bounds, config["num_initial"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    metrics_before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)

    history_best: List[float] = []
    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = DISOInfill(
            model=model_iter,
            bounds=bounds,
            x_train=x_current,
            y_train=y_current,
            criterion=config["criterion"],
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
    metrics_after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    gain = compute_relative_gain(metrics_before["accuracy"], metrics_after["accuracy"], eps=args.metric_eps)

    logger.info(f"{hue.b}Single-Objective Active Learning{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}%"
    )

    return {
        "case": spec.name,
        "algorithm": "DISO",
        "num_initial": config["num_initial"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "criterion": config["criterion"],
        "before": metrics_before,
        "after": metrics_after,
        "accuracy_gain": gain,
        "history_best": history_best,
        "passed": gain >= args.active_learning_min_relative_gain,
    }


def run_multi_fidelity_active_case(args: Any) -> Dict[str, Any]:
    """
    Run the multi-fidelity active learning benchmark.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Benchmark record.
    """
    config = bench_config.DEFAULT_MULTI_FIDELITY_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = bench_funcs.get_multifidelity_benchmark(config["name"])
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    bounds = spec.bounds_array

    x_lf = sample_lhs(bounds, config["num_lf"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    x_current = sample_lhs(bounds, config["num_hf_initial"], lhs_iterations)
    y_lf = spec.evaluate_low_fidelity(x_lf)
    y_current = spec.evaluate_high_fidelity(x_current)
    y_test = spec.evaluate_high_fidelity(x_test)

    model_before = fit_krg(x_current, y_current, args)
    metrics_before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)

    history_best: List[float] = []
    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiFidelityInfill(
            model=model_iter,
            x_hf=x_current,
            y_hf=y_current,
            x_lf=x_lf,
            y_lf=y_lf,
            target_idx=0,
            ratio=config["ratio"],
        )
        x_new = strategy.propose()
        y_new = spec.evaluate_high_fidelity(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history_best.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    metrics_after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    gain = compute_relative_gain(metrics_before["accuracy"], metrics_after["accuracy"], eps=args.metric_eps)

    logger.info(f"{hue.b}Multi-Fidelity Active Learning{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}%"
    )

    return {
        "case": spec.name,
        "algorithm": "MICO",
        "num_hf_initial": config["num_hf_initial"],
        "num_lf": config["num_lf"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "ratio": config["ratio"],
        "before": metrics_before,
        "after": metrics_after,
        "accuracy_gain": gain,
        "history_best": history_best,
        "passed": gain >= args.active_learning_min_relative_gain,
    }


def run_multi_objective_active_case(args: Any) -> Dict[str, Any]:
    """
    Run the multi-objective active learning benchmark.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Benchmark record.
    """
    config = bench_config.DEFAULT_MULTI_OBJECTIVE_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = bench_funcs.get_multiobjective_benchmark(config["name"])
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    bounds = spec.bounds_array

    x_current = sample_lhs(bounds, config["num_initial"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    metrics_before = evaluate_metrics(y_test, predict_mean(model_before, x_test), eps=args.metric_eps)
    pareto_before = compute_pareto_size(y_current)

    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiObjectiveInfill(
            model=model_iter,
            bounds=bounds,
            y_train=y_current,
            obj_idxs=list(range(y_current.shape[1])),
            constraint_idxs=None,
            constraint_ubs=None,
            num_samples=config["num_samples"],
            num_candidates=config["num_candidates"],
            num_restarts=config["num_restarts"],
            beta=config["beta"],
        )
        x_new = strategy.propose()
        y_new = spec.evaluate(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])

    model_after = fit_krg(x_current, y_current, args)
    metrics_after = evaluate_metrics(y_test, predict_mean(model_after, x_test), eps=args.metric_eps)
    pareto_after = compute_pareto_size(y_current)
    gain = compute_relative_gain(metrics_before["accuracy"], metrics_after["accuracy"], eps=args.metric_eps)

    logger.info(f"{hue.b}Multi-Objective Active Learning{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}%"
    )

    return {
        "case": spec.name,
        "algorithm": "MOBO",
        "num_initial": config["num_initial"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "num_samples": config["num_samples"],
        "num_candidates": config["num_candidates"],
        "num_restarts": config["num_restarts"],
        "beta": config["beta"],
        "before": metrics_before,
        "after": metrics_after,
        "accuracy_gain": gain,
        "pareto_size_before": pareto_before,
        "pareto_size_after": pareto_after,
        "passed": gain >= args.active_learning_min_relative_gain,
    }


def run_optimization_section(args: Any) -> List[Dict[str, Any]]:
    """
    Run the optimization benchmarks.

    Args:
        args (Any): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Per-case optimization records.
    """
    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Optimization Benchmarks{hue.q}")

    for case_name in args.optimization_cases:
        reset_random_state(args.seed)
        spec = bench_funcs.get_scalar_benchmark(case_name)
        config = bench_config.DEFAULT_OPTIMIZATION_CASES[case_name]
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        bounds = spec.bounds_array

        x_train = sample_lhs(bounds, config["num_train"], lhs_iterations)
        y_train = spec.evaluate(x_train)
        surrogate = fit_krg(x_train, y_train, args)

        def objective(x_vec: np.ndarray) -> float:
            mean_value, _ = surrogate.predict(np.asarray(x_vec, dtype=np.float64).reshape(1, -1))
            return float(mean_value[0, 0])

        case_result: Dict[str, Any] = {
            "case": case_name,
            "num_train": config["num_train"],
            "known_optimum": spec.known_optimum,
            "algorithms": {},
        }

        if "MIGA" in args.demos:
            result = multi_island_genetic_optimize(
                func=objective,
                bounds=[tuple(bound) for bound in bounds],
                tol=args.opt_tol,
                seed=args.seed,
                multi_objective=False,
                **args.miga_params,
            )
            verified = float(spec.evaluate(result.x.reshape(1, -1))[0, 0])
            optimum_gap = None if spec.known_optimum is None else abs(verified - spec.known_optimum)
            case_result["algorithms"]["MIGA"] = {
                "x_best": result.x,
                "predicted_value": float(result.fun),
                "verified_value": verified,
                "optimum_gap": optimum_gap,
            }
            logger.info(f"  {spec.name} / MIGA | pred={result.fun:.6f} | verified={verified:.6f}")

        if "CFARSSDA" in args.demos:
            result = dragonfly_optimize(
                func=objective,
                bounds=[tuple(bound) for bound in bounds],
                tol=args.opt_tol,
                seed=args.seed,
                multi_objective=False,
                **args.df_params,
            )
            verified = float(spec.evaluate(result.x.reshape(1, -1))[0, 0])
            optimum_gap = None if spec.known_optimum is None else abs(verified - spec.known_optimum)
            case_result["algorithms"]["CFARSSDA"] = {
                "x_best": result.x,
                "predicted_value": float(result.fun),
                "verified_value": verified,
                "optimum_gap": optimum_gap,
            }
            logger.info(f"  {spec.name} / CFARSSDA | pred={result.fun:.6f} | verified={verified:.6f}")

        results.append(case_result)

    return results


def run_bench_once(args: Any, seed: int) -> Dict[str, Any]:
    """
    Run one analytic benchmark pass for a fixed seed.

    Args:
        args (Any): Parsed arguments.
        seed (int): Random seed.

    Returns:
        Dict[str, Any]: Per-seed benchmark payload.
    """
    args.seed = seed
    seed_everything(seed)
    reset_random_state(seed)

    payload: Dict[str, Any] = {
        "seed_mode": "single_seed",
        "seed": seed,
        "demos": args.demos,
        "thresholds": {
            "ensemble_min_relative_gain": args.ensemble_min_relative_gain,
            "mf_min_accuracy": args.mf_min_accuracy,
            "active_learning_min_relative_gain": args.active_learning_min_relative_gain,
        },
    }

    if any(name in args.demos for name in ("TAHS", "AESMSI")):
        payload["ensemble"] = run_ensemble_section(args)

    if any(name in args.demos for name in ("MFSMLS", "MMFS", "CCAMFS")):
        payload["multifidelity"] = run_multifidelity_section(args)

    if "DISO" in args.demos:
        payload["single_objective_active_learning"] = run_single_objective_active_case(args)

    if "MICO" in args.demos:
        payload["multi_fidelity_active_learning"] = run_multi_fidelity_active_case(args)

    if "MOBO" in args.demos:
        payload["multi_objective_active_learning"] = run_multi_objective_active_case(args)

    if any(name in args.demos for name in ("MIGA", "CFARSSDA")):
        payload["optimization"] = run_optimization_section(args)

    return payload


def aggregate_ensemble_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate ensemble benchmark payloads across multiple seeds.

    Args:
        runs (List[Dict[str, Any]]): Per-seed payloads.

    Returns:
        List[Dict[str, Any]]: Averaged ensemble summary.
    """
    if not runs or "ensemble" not in runs[0]:
        return []

    threshold = runs[0]["thresholds"]["ensemble_min_relative_gain"]
    summary: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(runs[0]["ensemble"]):
        item = {
            "case": case["case"],
            "target_improvement": threshold,
            "mean_single_accuracy": float(mean(run["ensemble"][case_idx]["mean_single_accuracy"] for run in runs)),
            "algorithms": [],
        }
        for algo_name in case["algorithms"].keys():
            entries = [run["ensemble"][case_idx]["algorithms"][algo_name] for run in runs]
            avg_improvement = float(mean(entry["accuracy_gain"] for entry in entries))
            item["algorithms"].append(
                {
                    "name": algo_name,
                    "accuracy": float(mean(entry["accuracy"] for entry in entries)),
                    "r2": float(mean(entry["r2"] for entry in entries)),
                    "improvement": avg_improvement,
                    "passed": avg_improvement >= threshold,
                }
            )
        summary.append(item)

    return summary


def aggregate_multifidelity_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate multi-fidelity benchmark payloads across multiple seeds.

    Args:
        runs (List[Dict[str, Any]]): Per-seed payloads.

    Returns:
        List[Dict[str, Any]]: Averaged multi-fidelity summary.
    """
    if not runs or "multifidelity" not in runs[0]:
        return []

    threshold = runs[0]["thresholds"]["mf_min_accuracy"]
    summary: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(runs[0]["multifidelity"]):
        item = {"case": case["case"], "target_accuracy": threshold, "algorithms": []}
        for algo_name in case["algorithms"].keys():
            entries = [run["multifidelity"][case_idx]["algorithms"][algo_name] for run in runs]
            avg_accuracy = float(mean(entry["accuracy"] for entry in entries))
            item["algorithms"].append(
                {
                    "name": algo_name,
                    "accuracy": avg_accuracy,
                    "r2": float(mean(entry["r2"] for entry in entries)),
                    "passed": avg_accuracy >= threshold,
                }
            )
        summary.append(item)

    return summary


def aggregate_active_learning_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate active-learning payloads across multiple seeds.

    Args:
        runs (List[Dict[str, Any]]): Per-seed payloads.

    Returns:
        Dict[str, Dict[str, Any]]: Averaged active-learning summary.
    """
    if not runs:
        return {}

    threshold = runs[0]["thresholds"]["active_learning_min_relative_gain"]
    mapping = {
        "single_objective": ("single_objective_active_learning", "Single-Objective Active Learning"),
        "multi_fidelity": ("multi_fidelity_active_learning", "Multi-Fidelity Active Learning"),
        "multi_objective": ("multi_objective_active_learning", "Multi-Objective Active Learning"),
    }
    summary: Dict[str, Dict[str, Any]] = {}

    for key, (payload_key, title) in mapping.items():
        entries = [run[payload_key] for run in runs if payload_key in run]
        if not entries:
            continue
        summary[key] = {
            "title": title,
            "case": entries[0]["case"],
            "target_improvement": threshold,
            "before_accuracy": float(mean(entry["before"]["accuracy"] for entry in entries)),
            "after_accuracy": float(mean(entry["after"]["accuracy"] for entry in entries)),
            "before_r2": float(mean(entry["before"]["r2"] for entry in entries)),
            "after_r2": float(mean(entry["after"]["r2"] for entry in entries)),
            "improvement": float(mean(entry["accuracy_gain"] for entry in entries)),
            "passed": float(mean(entry["accuracy_gain"] for entry in entries)) >= threshold,
        }

    return summary


def aggregate_optimization_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate optimization payloads across multiple seeds.

    Args:
        runs (List[Dict[str, Any]]): Per-seed payloads.

    Returns:
        List[Dict[str, Any]]: Averaged optimization summary.
    """
    if not runs or "optimization" not in runs[0]:
        return []

    summary: List[Dict[str, Any]] = []
    template_cases = runs[0]["optimization"]

    for case_idx, case in enumerate(template_cases):
        for algo_name in case["algorithms"].keys():
            entries = [
                run["optimization"][case_idx]["algorithms"].get(algo_name)
                for run in runs
                if len(run.get("optimization", [])) > case_idx and algo_name in run["optimization"][case_idx]["algorithms"]
            ]
            if not entries:
                continue
            optimum_gap_values = [entry["optimum_gap"] for entry in entries if entry["optimum_gap"] is not None]
            summary.append(
                {
                    "case": case["case"],
                    "name": algo_name,
                    "success": len(entries) == len(runs),
                    "predicted_value": float(mean(entry["predicted_value"] for entry in entries)),
                    "verified_value": float(mean(entry["verified_value"] for entry in entries)),
                    "optimum_gap": float(mean(optimum_gap_values)) if optimum_gap_values else None,
                }
            )

    return summary


def build_average_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the averaged summary for a multi-seed benchmark run.

    Args:
        runs (List[Dict[str, Any]]): Per-seed benchmark payloads.

    Returns:
        Dict[str, Any]: Averaged benchmark summary.
    """
    return {
        "seed_mode": "multi_seed",
        "seeds": [run["seed"] for run in runs],
        "num_runs": len(runs),
        "demos": runs[0]["demos"],
        "thresholds": runs[0]["thresholds"],
        "ensemble": aggregate_ensemble_runs(runs),
        "multifidelity": aggregate_multifidelity_runs(runs),
        "active_learning": aggregate_active_learning_runs(runs),
        "optimization": aggregate_optimization_runs(runs),
    }


def run_bench_suite(args: Any) -> Dict[str, Any]:
    """
    Run the configured benchmark suite and return the final payload.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Final benchmark payload.
    """
    if args.seed_mode == "single_seed":
        return run_bench_once(args, args.seed)

    runs = [run_bench_once(args, seed) for seed in args.seeds]
    return {
        "seed_mode": "multi_seed",
        "seeds": args.seeds,
        "raw_runs": runs,
        "average": build_average_summary(runs),
    }


def save_results(payload: Dict[str, Any]) -> str:
    """
    Save the benchmark payload as JSON.

    Args:
        payload (Dict[str, Any]): Benchmark payload.

    Returns:
        str: Absolute save path.
    """
    with open(RESULT_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2)
    return RESULT_FILE_PATH


def main() -> None:
    """
    Execute the analytic benchmark workflow.
    """
    args = bench_config.get_args()

    logger.info(f"{hue.b}SurrogateLab Analytic Benchmarks{hue.q}")
    logger.info(f"  seed_mode : {args.seed_mode}")
    logger.info(f"  seed      : {args.seed}")
    logger.info(f"  seeds     : {args.seeds}")
    logger.info(f"  demos     : {args.demos}")

    payload = run_bench_suite(args)
    save_path = save_results(payload)
    logger.info(f"{hue.g}Benchmark report saved to {save_path}{hue.q}")


if __name__ == "__main__":
    main()
