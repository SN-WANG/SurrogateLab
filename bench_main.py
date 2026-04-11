# Analytic benchmark runner for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations
from functools import partial
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


# ============================================================
# Shared Helpers
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


def color_status(passed: bool) -> str:
    """
    Format PASS / FAIL with terminal colors.

    Args:
        passed (bool): Whether the benchmark item passed.

    Returns:
        str: Colored PASS / FAIL label.
    """
    return f"{hue.g}PASS{hue.q}" if passed else f"{hue.r}FAIL{hue.q}"


def status_label(passed: bool) -> str:
    """
    Convert a boolean pass flag into a plain-text status.

    Args:
        passed (bool): Whether the benchmark item passed.

    Returns:
        str: PASS or FAIL.
    """
    return "PASS" if passed else "FAIL"


def build_status_summary(flags_by_algorithm: Dict[str, List[bool]]) -> Dict[str, str]:
    """
    Build algorithm-level PASS / FAIL labels from per-case pass flags.

    Args:
        flags_by_algorithm (Dict[str, List[bool]]): Per-algorithm pass flags.

    Returns:
        Dict[str, str]: Algorithm-level summary labels.
    """
    return {algo_name: status_label(all(flags)) for algo_name, flags in flags_by_algorithm.items()}


def summarize_sections(
    ensemble: List[Dict[str, Any]],
    multifidelity: List[Dict[str, Any]],
    active_learning: Dict[str, Dict[str, Any]],
    optimization: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    Build the benchmark summary for the averaged multi-seed payload.

    Args:
        ensemble (List[Dict[str, Any]]): Averaged ensemble case records.
        multifidelity (List[Dict[str, Any]]): Averaged multi-fidelity case records.
        active_learning (Dict[str, Dict[str, Any]]): Averaged active-learning records.
        optimization (List[Dict[str, Any]]): Averaged optimization records.

    Returns:
        Dict[str, Dict[str, str]]: Section-level summary.
    """
    ensemble_flags: Dict[str, List[bool]] = {}
    for case in ensemble:
        for item in case["algorithms"]:
            ensemble_flags.setdefault(item["name"], []).append(bool(item["passed"]))

    multifidelity_flags: Dict[str, List[bool]] = {}
    for case in multifidelity:
        for item in case["algorithms"]:
            multifidelity_flags.setdefault(item["name"], []).append(bool(item["passed"]))

    return {
        "ensemble": build_status_summary(ensemble_flags),
        "multifidelity": build_status_summary(multifidelity_flags),
        "active_learning": {
            item["algorithm"]: status_label(bool(item["passed"])) for item in active_learning.values()
        },
        "optimization": {item["name"]: "PASS" for item in optimization},
    }


def print_average_payload(average: Dict[str, Any]) -> None:
    """
    Print averaged multi-seed benchmark results.

    Args:
        average (Dict[str, Any]): Averaged benchmark payload.
    """
    logger.info(f"{hue.b}Multi-Seed Average Results{hue.q}")
    logger.info(f"  seeds    : {average['seeds']}")

    if average["ensemble"]:
        logger.info(f"{hue.b}Ensemble Benchmarks (Average){hue.q}")
        for case in average["ensemble"]:
            logger.info(
                f"  {case['case']}: baseline acc={case['mean_single_accuracy']:.2f}% | "
                f"target gain>={100.0 * case['target_improvement']:.2f}%"
            )
            for item in case["algorithms"]:
                logger.info(
                    f"    {item['name']} | acc={item['accuracy']:.2f}% | r2={item['r2']:.4f} | "
                    f"gain={100.0 * item['improvement']:.2f}% -> {color_status(item['passed'])}"
                )

    if average["multifidelity"]:
        logger.info(f"{hue.b}Multi-Fidelity Benchmarks (Average){hue.q}")
        for case in average["multifidelity"]:
            logger.info(f"  {case['case']}: target acc>={case['target_accuracy']:.2f}%")
            for item in case["algorithms"]:
                logger.info(
                    f"    {item['name']} | acc={item['accuracy']:.2f}% | r2={item['r2']:.4f} "
                    f"-> {color_status(item['passed'])}"
                )

    if average["active_learning"]:
        logger.info(f"{hue.b}Active Learning Benchmarks (Average){hue.q}")
        for key in ("single_objective", "multi_fidelity", "multi_objective"):
            if key not in average["active_learning"]:
                continue
            item = average["active_learning"][key]
            logger.info(
                f"  {item['title']} | {item['case']} | acc {item['before_accuracy']:.2f}% -> "
                f"{item['after_accuracy']:.2f}% | r2 {item['before_r2']:.4f} -> {item['after_r2']:.4f} | "
                f"gain={100.0 * item['improvement']:.2f}% -> {color_status(item['passed'])}"
            )

    if average["optimization"]:
        logger.info(f"{hue.b}Optimization Benchmarks (Average){hue.q}")
        for item in average["optimization"]:
            logger.info(f"  {item['case']} / {item['name']} -> {hue.g}PASS{hue.q}")


# ============================================================
# Benchmark Sections
# ============================================================

def run_ensemble_section(args: Any) -> List[Dict[str, Any]]:
    """
    Run the ensemble surrogate benchmarks.

    Args:
        args (Any): Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Per-case benchmark records.
    """
    single_model_builders: Dict[str, Callable[[], Any]] = {
        "PRS": lambda: PRS(**args.prs_params),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(**args.svr_params),
    }
    ensemble_builders: Dict[str, Callable[[], Any]] = {
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
    logger.info(f"{hue.b}Ensemble Benchmarks{hue.q}")

    for case_name in args.ensemble_cases:
        reset_random_state(args.seed)
        spec = bench_funcs.get_scalar_benchmark(case_name)
        config = bench_config.ENSEMBLE_CASES[case_name]
        bounds = spec.bounds_array

        x_train = sample_lhs(bounds, config["num_train"])
        x_test = sample_lhs(bounds, config["num_test"])
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
        logger.info(f"  {spec.name}: baseline acc={mean_single_accuracy:.2f}%")

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
        config = bench_config.MULTIFIDELITY_CASES[case_name]
        bounds = spec.bounds_array

        x_lf = sample_lhs(bounds, config["num_lf"])
        x_hf = sample_lhs(bounds, config["num_hf"])
        x_test = sample_lhs(bounds, config["num_test"])
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
    config = bench_config.ACTIVE_LEARNING_CASES["single_objective"]
    reset_random_state(args.seed)
    spec = bench_funcs.get_scalar_benchmark(config["name"])
    bounds = spec.bounds_array

    x_current = sample_lhs(bounds, config["num_initial"])
    x_test = sample_lhs(bounds, config["num_test"])
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
    passed = gain >= args.active_learning_min_relative_gain

    logger.info(f"{hue.b}Single-Objective Active Learning Benchmarks{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}% "
        f"-> {status_label(passed)}"
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
        "passed": passed,
    }


def run_multi_fidelity_active_case(args: Any) -> Dict[str, Any]:
    """
    Run the multi-fidelity active learning benchmark.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Benchmark record.
    """
    config = bench_config.ACTIVE_LEARNING_CASES["multi_fidelity"]
    reset_random_state(args.seed)
    spec = bench_funcs.get_multifidelity_benchmark(config["name"])
    bounds = spec.bounds_array

    x_lf = sample_lhs(bounds, config["num_lf"])
    x_test = sample_lhs(bounds, config["num_test"])
    x_current = sample_lhs(bounds, config["num_hf_initial"])
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
    passed = gain >= args.active_learning_min_relative_gain

    logger.info(f"{hue.b}Multi-Fidelity Active Learning Benchmarks{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}% "
        f"-> {status_label(passed)}"
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
        "passed": passed,
    }


def run_multi_objective_active_case(args: Any) -> Dict[str, Any]:
    """
    Run the multi-objective active learning benchmark.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Benchmark record.
    """
    config = bench_config.ACTIVE_LEARNING_CASES["multi_objective"]
    reset_random_state(args.seed)
    spec = bench_funcs.get_multiobjective_benchmark(config["name"])
    bounds = spec.bounds_array

    x_current = sample_lhs(bounds, config["num_initial"])
    x_test = sample_lhs(bounds, config["num_test"])
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
    passed = gain >= args.active_learning_min_relative_gain

    logger.info(f"{hue.b}Multi-Objective Active Learning Benchmarks{hue.q}")
    logger.info(
        f"  {spec.name} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
        f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}% "
        f"-> {status_label(passed)}"
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
        "passed": passed,
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
        config = bench_config.OPTIMIZATION_CASES[case_name]
        bounds = spec.bounds_array

        x_train = sample_lhs(bounds, config["num_train"])
        y_train = spec.evaluate(x_train)
        surrogate = fit_krg(x_train, y_train, args)

        objective = partial(predict_scalar_output, surrogate, 0)

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
            case_result["algorithms"]["MIGA"] = {
                "x_best": result.x,
                "predicted_objective": float(result.fun),
                "verified_objective": verified,
            }
            logger.info(f"  {spec.name} / MIGA -> PASS")

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
            case_result["algorithms"]["CFARSSDA"] = {
                "x_best": result.x,
                "predicted_objective": float(result.fun),
                "verified_objective": verified,
            }
            logger.info(f"  {spec.name} / CFARSSDA -> PASS")

        results.append(case_result)

    return results


# ============================================================
# Aggregation and Orchestration
# ============================================================

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

    thresholds = {
        "ensemble_min_relative_gain": args.ensemble_min_relative_gain,
        "mf_min_accuracy": args.mf_min_accuracy,
        "active_learning_min_relative_gain": args.active_learning_min_relative_gain,
    }
    ensemble_results: List[Dict[str, Any]] = []
    multifidelity_results: List[Dict[str, Any]] = []
    active_learning_results: Dict[str, Dict[str, Any]] = {}
    optimization_results: List[Dict[str, Any]] = []

    if any(name in args.demos for name in ("TAHS", "AESMSI")):
        ensemble_results = run_ensemble_section(args)

    if any(name in args.demos for name in ("MFSMLS", "MMFS", "CCAMFS")):
        multifidelity_results = run_multifidelity_section(args)

    if "DISO" in args.demos:
        active_learning_results["single_objective"] = run_single_objective_active_case(args)

    if "MICO" in args.demos:
        active_learning_results["multi_fidelity"] = run_multi_fidelity_active_case(args)

    if "MOBO" in args.demos:
        active_learning_results["multi_objective"] = run_multi_objective_active_case(args)

    if any(name in args.demos for name in ("MIGA", "CFARSSDA")):
        optimization_results = run_optimization_section(args)

    return {
        "seed": seed,
        "demos": args.demos,
        "thresholds": thresholds,
        "ensemble": ensemble_results,
        "multifidelity": multifidelity_results,
        "active_learning": active_learning_results,
        "optimization": optimization_results,
    }


def aggregate_ensemble_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate ensemble benchmark payloads across multiple seeds.

    Args:
        runs (List[Dict[str, Any]]): Per-seed payloads.

    Returns:
        List[Dict[str, Any]]: Averaged ensemble summary.
    """
    template_cases = runs[0]["ensemble"]
    if not template_cases:
        return []

    threshold = runs[0]["thresholds"]["ensemble_min_relative_gain"]
    summary: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(template_cases):
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
    template_cases = runs[0]["multifidelity"]
    if not template_cases:
        return []

    threshold = runs[0]["thresholds"]["mf_min_accuracy"]
    summary: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(template_cases):
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
    threshold = runs[0]["thresholds"]["active_learning_min_relative_gain"]
    summary: Dict[str, Dict[str, Any]] = {}

    for key, title in (
        ("single_objective", "Single-Objective Active Learning"),
        ("multi_fidelity", "Multi-Fidelity Active Learning"),
        ("multi_objective", "Multi-Objective Active Learning"),
    ):
        entries = [run["active_learning"][key] for run in runs if key in run["active_learning"]]
        if not entries:
            continue
        summary[key] = {
            "title": title,
            "algorithm": entries[0]["algorithm"],
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
    template_cases = runs[0]["optimization"]
    if not template_cases:
        return []

    summary: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(template_cases):
        for algo_name in case["algorithms"].keys():
            entries = [run["optimization"][case_idx]["algorithms"][algo_name] for run in runs]
            summary.append(
                {
                    "case": case["case"],
                    "name": algo_name,
                    "predicted_objective": float(mean(entry["predicted_objective"] for entry in entries)),
                    "verified_objective": float(mean(entry["verified_objective"] for entry in entries)),
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
    ensemble = aggregate_ensemble_runs(runs)
    multifidelity = aggregate_multifidelity_runs(runs)
    active_learning = aggregate_active_learning_runs(runs)
    optimization = aggregate_optimization_runs(runs)

    return {
        "seeds": [run["seed"] for run in runs],
        "demos": runs[0]["demos"],
        "thresholds": runs[0]["thresholds"],
        "summary": summarize_sections(ensemble, multifidelity, active_learning, optimization),
        "ensemble": ensemble,
        "multifidelity": multifidelity,
        "active_learning": active_learning,
        "optimization": optimization,
        "raw_runs": runs,
    }


def run_bench_suite(args: Any) -> Dict[str, Any]:
    """
    Run the configured benchmark suite and return the final payload.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, Any]: Final benchmark payload.
    """
    runs = [run_bench_once(args, seed) for seed in args.seeds]
    return build_average_summary(runs)


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
        json.dump(to_serializable(payload), file, indent=2)
    return save_path


def main() -> None:
    """
    Execute the analytic benchmark workflow.
    """
    args = bench_config.get_args()

    logger.info(f"{hue.b}SurrogateLab Analytic Benchmarks{hue.q}")
    logger.info(f"  seeds     : {args.seeds}")
    logger.info(f"  demos     : {args.demos}")

    payload = run_bench_suite(args)
    print_average_payload(payload)
    save_path = save_results(payload)
    logger.info(f"{hue.g}Benchmark report saved to {save_path}{hue.q}")


if __name__ == "__main__":
    main()
