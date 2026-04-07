# Engineering Abaqus benchmark runner for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import NonlinearConstraint

import abaqus_config

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


class AbaqusModel:
    """
    Mock Abaqus FEM solver for local engineering validation.

    This class keeps the same lightweight call contract as the real server-side
    Abaqus model: instantiate the model and call ``run(input_arr)`` with a
    single design vector.

    Args:
        fidelity (str): Simulation fidelity, ``"high"`` or ``"low"``.
    """

    def __init__(self, fidelity: str = "high") -> None:
        self.fidelity = fidelity
        self.input_vars = ["thick1", "thick2", "thick3"]
        self.output_vars = ["weight", "displacement", "stress_skin", "stress_stiff"]

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """
        Execute one Abaqus-style simulation.

        Args:
            input_arr (np.ndarray): Design vector. (D,).

        Returns:
            np.ndarray: Structural responses. (4,).
        """
        x = np.asarray(input_arr, dtype=np.float64).reshape(-1)

        def _branin(x1: float, x2: float) -> float:
            a = 1.0
            b = 5.1 / (4.0 * np.pi ** 2)
            c = 5.0 / np.pi
            r = 6.0
            s = 10.0
            t = 1.0 / (8.0 * np.pi)
            return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s

        y_weight = (_branin(x[0], x[1]) + 2.0 * x[2]) / 300.0
        y_displacement = 0.5 * x[0] ** 2 + 1.2 * x[1] + np.sin(x[2])
        y_stress_skin = _branin(x[1], x[2]) + 0.8 * x[0]
        y_stress_stiff = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2 + (x[2] - 5.0) ** 2
        y = np.array([y_weight, y_displacement, y_stress_skin, y_stress_stiff], dtype=np.float64)

        if self.fidelity == "low":
            bias = np.array([0.003, 5.0, 5.0, 5.0], dtype=np.float64)
            noise = np.array([1.0e-4, 0.1, 0.1, 0.1], dtype=np.float64)
            y = 0.85 * y + bias + np.random.normal(0.0, noise, size=y.shape)

        return y


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


def sample_lhs(bounds: np.ndarray, num_samples: int, lhs_iterations: int) -> np.ndarray:
    """
    Generate a maximin Latin hypercube inside the engineering design box.

    Args:
        bounds (np.ndarray): Box bounds. (D, 2).
        num_samples (int): Number of samples.
        lhs_iterations (int): Maximin search iterations.

    Returns:
        np.ndarray: Physical samples. (N, D).
    """
    x_norm = lhs_design(num_samples, bounds.shape[0], iterations=lhs_iterations)
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
        target = dict(abaqus_config.TARGET_SPECS[name])
        target["name"] = name
        specs.append(target)
    return specs


def generate_and_cache_doe(args: Any) -> Dict[str, np.ndarray]:
    """
    Generate or load the cached engineering DOE dataset.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Dict[str, np.ndarray]: Cached engineering data.
    """
    cache_path = os.path.join(args.save_dir, "abaqus_doe_cache.npy")
    if os.path.isfile(cache_path):
        logger.info(f"{hue.g}Load cached engineering DOE from {cache_path}{hue.q}")
        return np.load(cache_path, allow_pickle=True).item()

    logger.info(f"{hue.b}Generate engineering DOE data{hue.q}")
    x_train = sample_lhs(args.bounds, args.num_train, args.lhs_iterations)
    x_test = sample_lhs(args.bounds, args.num_test, args.lhs_iterations)
    x_lf = sample_lhs(args.bounds, args.num_lf, args.lhs_iterations)
    x_hf = sample_lhs(args.bounds, args.num_hf, args.lhs_iterations)

    data = {
        "x_train": x_train,
        "y_train": run_abaqus_batch(x_train, fidelity="high"),
        "x_test": x_test,
        "y_test": run_abaqus_batch(x_test, fidelity="high"),
        "x_lf": x_lf,
        "y_lf": run_abaqus_batch(x_lf, fidelity="low"),
        "x_hf": x_hf,
        "y_hf": run_abaqus_batch(x_hf, fidelity="high"),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(cache_path, data, allow_pickle=True)
    logger.info(
        f"  train={data['x_train'].shape} | test={data['x_test'].shape} | "
        f"lf={data['x_lf'].shape} | hf={data['x_hf'].shape}"
    )
    return data


def run_ensemble_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the engineering ensemble-model benchmarks.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Target-wise benchmark records.
    """
    single_model_builders = {
        "PRS": lambda: PRS(),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(),
    }
    ensemble_builders = {
        "A": ("TAHS", lambda: TAHS(threshold=args.ensemble_threshold, krg_params=args.krg_params)),
        "B": ("AESMSI", lambda: AESMSI(threshold=args.ensemble_threshold, krg_params=args.krg_params)),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Engineering Ensemble Cases{hue.q}")

    for target in get_target_specs(args):
        y_train = select_output(data["y_train"], target["output_idx"])
        y_test = select_output(data["y_test"], target["output_idx"])

        baseline_scores: Dict[str, Dict[str, float]] = {}
        for model_name, builder in single_model_builders.items():
            model = builder()
            model.fit(data["x_train"], y_train)
            baseline_scores[model_name] = evaluate_metrics(
                y_test,
                predict_mean(model, data["x_test"]),
                eps=args.metric_eps,
            )

        mean_single_accuracy = float(np.mean([item["accuracy"] for item in baseline_scores.values()]))
        record = {
            "target": target["name"],
            "baseline_single_models": baseline_scores,
            "mean_single_accuracy": mean_single_accuracy,
            "algorithms": {},
        }
        logger.info(f"  {target['label']} | baseline acc={hue.c}{mean_single_accuracy:.2f}%{hue.q}")

        for demo_label, (algo_name, builder) in ensemble_builders.items():
            if demo_label not in args.demos:
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
    Run the engineering multi-fidelity benchmarks.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Target-wise benchmark records.
    """
    model_builders = {
        "C": ("MFSMLS", lambda: MFSMLS(poly_degree=args.mf_poly_degree)),
        "D": ("MMFS", lambda: MMFS(sigma_bounds=tuple(args.mf_sigma_bounds))),
        "E": ("CCAMFS", lambda: CCAMFS()),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}Engineering Multi-Fidelity Cases{hue.q}")

    for target in get_target_specs(args):
        y_lf = select_output(data["y_lf"], target["output_idx"])
        y_hf = select_output(data["y_hf"], target["output_idx"])
        y_test = select_output(data["y_test"], target["output_idx"])

        record = {
            "target": target["name"],
            "algorithms": {},
        }
        logger.info(f"  {target['label']} | target acc>={args.mf_min_accuracy:.2f}%")

        for demo_label, (algo_name, builder) in model_builders.items():
            if demo_label not in args.demos:
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
    logger.info(f"{hue.b}Engineering Active-Learning Cases{hue.q}")
    x_initial = sample_lhs(args.bounds, args.num_active_initial, args.lhs_iterations)
    y_initial_full = run_abaqus_batch(x_initial, fidelity="high")

    for target in get_target_specs(args):
        x_current = x_initial.copy()
        y_current = select_output(y_initial_full, target["output_idx"]).copy()
        y_test = select_output(data["y_test"], target["output_idx"])

        model_before = fit_krg(x_current, y_current, args)
        metrics_before = evaluate_metrics(y_test, predict_mean(model_before, data["x_test"]), eps=args.metric_eps)

        history_best: List[float] = []
        hf_model = AbaqusModel(fidelity="high")
        for _ in range(args.num_infill):
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
            x_new = strategy.propose()
            y_new_full = hf_model.run(x_new[0])
            y_new = y_new_full[target["output_idx"] : target["output_idx"] + 1].reshape(1, 1)
            x_current = np.vstack([x_current, x_new])
            y_current = np.vstack([y_current, y_new])
            history_best.append(float(np.min(y_current[:, 0])))

        model_after = fit_krg(x_current, y_current, args)
        metrics_after = evaluate_metrics(y_test, predict_mean(model_after, data["x_test"]), eps=args.metric_eps)
        gain = compute_relative_gain(metrics_before["accuracy"], metrics_after["accuracy"], eps=args.metric_eps)
        passed = gain >= args.active_learning_min_relative_gain

        logger.info(
            f"  {target['label']} | acc {metrics_before['accuracy']:.2f}% -> {metrics_after['accuracy']:.2f}% | "
            f"r2 {metrics_before['r2']:.4f} -> {metrics_after['r2']:.4f} | gain={100.0 * gain:.2f}%"
        )
        results.append(
            {
                "target": target["name"],
                "criterion": args.infill_criterion,
                "num_initial": int(args.num_active_initial),
                "num_infill": args.num_infill,
                "before": metrics_before,
                "after": metrics_after,
                "accuracy_gain": gain,
                "history_best": history_best,
                "passed": passed,
            }
        )

    return results


def run_optimization_section(args: Any, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run the constrained engineering optimization cases.

    Args:
        args (Any): Parsed arguments.
        data (Dict[str, np.ndarray]): Engineering DOE data.

    Returns:
        List[Dict[str, Any]]: Optimization records.
    """
    objective_idx = abaqus_config.TARGET_SPECS[args.opt_target]["output_idx"]
    constraint_idx = abaqus_config.TARGET_SPECS[args.opt_constraint_target]["output_idx"]
    model = fit_krg(data["x_train"], data["y_train"], args)
    x0 = data["x_train"][np.argmin(data["y_train"][:, objective_idx])]
    hf_model = AbaqusModel(fidelity="high")

    def objective(x_vec: np.ndarray) -> float:
        mean, _ = model.predict(np.asarray(x_vec, dtype=np.float64).reshape(1, -1))
        return float(mean[0, objective_idx])

    def constraint_fun(x_vec: np.ndarray) -> float:
        mean, _ = model.predict(np.asarray(x_vec, dtype=np.float64).reshape(1, -1))
        return float(mean[0, constraint_idx])

    constraint = NonlinearConstraint(fun=constraint_fun, lb=-np.inf, ub=args.opt_constraint_ub)
    optimizers = {
        "I": (
            "MIGA",
            lambda: multi_island_genetic_optimize(
                func=objective,
                bounds=[tuple(bound) for bound in args.bounds],
                constraints=constraint,
                x0=x0,
                tol=args.opt_tol,
                seed=args.seed,
                multi_objective=False,
                **args.miga_params,
            ),
        ),
        "J": (
            "CFSSDA",
            lambda: dragonfly_optimize(
                func=objective,
                bounds=[tuple(bound) for bound in args.bounds],
                constraints=constraint,
                x0=x0,
                tol=args.opt_tol,
                seed=args.seed,
                multi_objective=False,
                **args.df_params,
            ),
        ),
    }

    results: List[Dict[str, Any]] = []
    logger.info(
        f"{hue.b}Engineering Optimization Cases{hue.q} | "
        f"objective={args.opt_target} | constraint={args.opt_constraint_target}<={args.opt_constraint_ub:.3f}"
    )

    for demo_label, (algo_name, builder) in optimizers.items():
        if demo_label not in args.demos:
            continue

        result = builder()
        verified = hf_model.run(result.x)
        predicted_constraint = constraint_fun(result.x)
        record = {
            "algorithm": algo_name,
            "x_best": result.x,
            "predicted_objective": float(result.fun),
            "predicted_constraint": float(predicted_constraint),
            "verified_objective": float(verified[objective_idx]),
            "verified_constraint": float(verified[constraint_idx]),
            "constraint_upper_bound": args.opt_constraint_ub,
            "constraint_satisfied": bool(verified[constraint_idx] <= args.opt_constraint_ub),
            "nit": getattr(result, "nit", None),
            "success": getattr(result, "success", None),
        }
        logger.info(
            f"  {algo_name} | pred={record['predicted_objective']:.6f} | "
            f"verified={record['verified_objective']:.6f} | "
            f"{args.opt_constraint_target}={record['verified_constraint']:.6f}"
        )
        results.append(record)

    return results


def plot_metric_section(section_name: str, records: List[Dict[str, Any]], save_path: str) -> None:
    """
    Save a simple accuracy/R2 summary plot for one section.

    Args:
        section_name (str): Section name.
        records (List[Dict[str, Any]]): Section records.
        save_path (str): Output path.
    """
    import matplotlib.pyplot as plt

    labels: List[str] = []
    accuracy: List[float] = []
    r2_values: List[float] = []
    for record in records:
        target = record["target"]
        for algo_name, metrics in record["algorithms"].items():
            labels.append(f"{target}:{algo_name}")
            accuracy.append(metrics["accuracy"])
            r2_values.append(metrics["r2"])

    if not labels:
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.4), 4))
    axes[0].bar(x, accuracy, color="tab:blue")
    axes[0].set_title(f"{section_name} Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, r2_values, color="tab:orange")
    axes[1].set_title(f"{section_name} R2")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_active_learning(records: List[Dict[str, Any]], save_path: str) -> None:
    """
    Save a before/after active-learning accuracy plot.

    Args:
        records (List[Dict[str, Any]]): Active-learning records.
        save_path (str): Output path.
    """
    import matplotlib.pyplot as plt

    labels = [record["target"] for record in records]
    acc_before = [record["before"]["accuracy"] for record in records]
    acc_after = [record["after"]["accuracy"] for record in records]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.0), 4))
    ax.bar(x - width / 2.0, acc_before, width, label="Before", color="tab:gray")
    ax.bar(x + width / 2.0, acc_after, width, label="After", color="tab:green")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Engineering Active Learning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_optimization(records: List[Dict[str, Any]], save_path: str) -> None:
    """
    Save a simple optimization verification plot.

    Args:
        records (List[Dict[str, Any]]): Optimization records.
        save_path (str): Output path.
    """
    import matplotlib.pyplot as plt

    labels = [record["algorithm"] for record in records]
    objectives = [record["verified_objective"] for record in records]
    constraints = [record["verified_constraint"] for record in records]
    ub = records[0]["constraint_upper_bound"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(x, objectives, color="tab:blue")
    axes[0].set_title("Verified Objective")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, constraints, color="tab:orange")
    axes[1].axhline(ub, color="tab:red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Verified Constraint")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_results(args: Any, payload: Dict[str, Any]) -> str:
    """
    Save the engineering payload as JSON.

    Args:
        args (Any): Parsed arguments.
        payload (Dict[str, Any]): Engineering payload.

    Returns:
        str: Absolute save path.
    """
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(args.save_dir, "engineering_results.json"))
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2)
    return save_path


def maybe_visualize(args: Any, payload: Dict[str, Any]) -> None:
    """
    Save section summary figures when visualization is enabled.

    Args:
        args (Any): Parsed arguments.
        payload (Dict[str, Any]): Engineering payload.
    """
    if not args.visualize:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    if "ensemble" in payload:
        plot_metric_section("Ensemble", payload["ensemble"], os.path.join(args.save_dir, "fig_ensemble_engineering.png"))
    if "multifidelity" in payload:
        plot_metric_section(
            "Multi-Fidelity",
            payload["multifidelity"],
            os.path.join(args.save_dir, "fig_multifidelity_engineering.png"),
        )
    if "active_learning" in payload:
        plot_active_learning(payload["active_learning"], os.path.join(args.save_dir, "fig_active_learning_engineering.png"))
    if "optimization" in payload:
        plot_optimization(payload["optimization"], os.path.join(args.save_dir, "fig_optimization_engineering.png"))


def main() -> None:
    """
    Execute the full Abaqus engineering workflow.
    """
    args = abaqus_config.get_args()
    seed_everything(args.seed)
    reset_random_state(args.seed)

    logger.info(f"{hue.b}SurrogateLab Abaqus Engineering Workflow{hue.q}")
    logger.info(f"  demos    : {args.demos}")
    logger.info(f"  targets  : {args.targets}")
    logger.info(f"  save_dir : {args.save_dir}")

    data = generate_and_cache_doe(args)
    payload: Dict[str, Any] = {
        "seed": args.seed,
        "demos": args.demos,
        "targets": args.targets,
        "thresholds": {
            "ensemble_min_relative_gain": args.ensemble_min_relative_gain,
            "mf_min_accuracy": args.mf_min_accuracy,
            "active_learning_min_relative_gain": args.active_learning_min_relative_gain,
            "weight_constraint_ub": args.opt_constraint_ub,
        },
    }

    if any(label in args.demos for label in ["A", "B"]):
        payload["ensemble"] = run_ensemble_section(args, data)

    if any(label in args.demos for label in ["C", "D", "E"]):
        payload["multifidelity"] = run_multifidelity_section(args, data)

    if "F" in args.demos:
        payload["active_learning"] = run_active_learning_section(args, data)

    if any(label in args.demos for label in ["I", "J"]):
        payload["optimization"] = run_optimization_section(args, data)

    payload["summary"] = {
        "ensemble_runs": sum(len(record["algorithms"]) for record in payload.get("ensemble", [])),
        "multifidelity_runs": sum(len(record["algorithms"]) for record in payload.get("multifidelity", [])),
        "active_learning_runs": len(payload.get("active_learning", [])),
        "optimization_runs": len(payload.get("optimization", [])),
    }

    maybe_visualize(args, payload)
    save_path = save_results(args, payload)
    logger.info(f"{hue.g}Engineering report saved to {save_path}{hue.q}")


if __name__ == "__main__":
    main()
