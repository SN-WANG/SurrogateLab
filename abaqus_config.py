# Engineering Abaqus benchmark configuration for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Iterable, List

import numpy as np


ALGORITHM_ORDER = ["A", "B", "C", "D", "E", "F", "I", "J"]

DEMO_ALIAS_MAP = {
    "all": ALGORITHM_ORDER,
    "ensemble": ["A", "B"],
    "multifidelity": ["C", "D", "E"],
    "active_learning": ["F"],
    "optimization": ["I", "J"],
}

TARGET_SPECS = OrderedDict(
    {
        "weight": {
            "output_idx": 0,
            "label": "weight",
            "description": "Structural weight response.",
        },
        "stress_skin": {
            "output_idx": 2,
            "label": "stress_skin",
            "description": "Outer skin stress response.",
        },
    }
)

DEFAULT_BOUNDS_LOWER = [4.0, 4.0, 4.0]
DEFAULT_BOUNDS_UPPER = [10.0, 10.0, 10.0]
DEFAULT_INPUT_NAMES = ["thick1", "thick2", "thick3"]
DEFAULT_OUTPUT_NAMES = ["weight", "displacement", "stress_skin", "stress_stiff"]

DEFAULT_MIGA_PARAMS = {
    "popsize": 10,
    "maxiter": 50,
    "num_islands": 4,
    "migration_interval": 10,
    "migration_size": 2,
}

DEFAULT_DRAGONFLY_PARAMS = {
    "popsize": 20,
    "maxiter": 120,
}

DEFAULT_THRESHOLD_PARAMS = {
    "ensemble_min_relative_gain": 0.10,
    "mf_min_accuracy": 90.0,
    "active_learning_min_relative_gain": 0.20,
    "weight_constraint_ub": 0.31,
}


def _expand_demo_selection(selection: Iterable[str]) -> List[str]:
    """
    Expand demo aliases into ordered labels.

    Args:
        selection (Iterable[str]): Demo labels or aliases.

    Returns:
        List[str]: Ordered demo labels.
    """
    expanded: List[str] = []
    for item in selection:
        key = item.lower()
        expanded.extend(DEMO_ALIAS_MAP.get(key, [item.upper()]))

    ordered: List[str] = []
    for label in ALGORITHM_ORDER:
        if label in expanded:
            ordered.append(label)
    return ordered


def _expand_target_selection(selection: List[str]) -> List[str]:
    """
    Expand target aliases into ordered engineering targets.

    Args:
        selection (List[str]): Requested target names.

    Returns:
        List[str]: Ordered target names.
    """
    if len(selection) == 1 and selection[0].lower() == "all":
        return list(TARGET_SPECS.keys())

    normalized = [item.lower() for item in selection]
    unknown = [item for item in normalized if item not in TARGET_SPECS]
    if unknown:
        valid = ", ".join(TARGET_SPECS.keys())
        raise ValueError(f"Unknown engineering target(s): {unknown}. Valid choices: {valid}.")
    return normalized


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Abaqus engineering workflow.

    Returns:
        argparse.Namespace: Parsed engineering configuration.
    """
    parser = argparse.ArgumentParser(description="SurrogateLab Abaqus engineering benchmark runner.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_dir", type=str, default="./abaqus_outputs", help="Directory for cached data and reports.")
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["all"],
        help="Algorithms to run: A B C D E F I J or aliases all / ensemble / multifidelity / active_learning / optimization.",
    )
    parser.add_argument("--targets", nargs="+", default=["all"], help="Engineering targets: weight / stress_skin / all.")
    parser.add_argument("--visualize", action="store_true", help="Save summary figures to save_dir.")

    parser.add_argument("--num_features", type=int, default=3, help="Number of design variables.")
    parser.add_argument("--num_outputs", type=int, default=4, help="Number of simulation outputs.")
    parser.add_argument("--bounds_lower", type=float, nargs="+", default=DEFAULT_BOUNDS_LOWER, help="Lower bounds.")
    parser.add_argument("--bounds_upper", type=float, nargs="+", default=DEFAULT_BOUNDS_UPPER, help="Upper bounds.")
    parser.add_argument("--input_names", type=str, nargs="+", default=DEFAULT_INPUT_NAMES, help="Input variable names.")
    parser.add_argument("--output_names", type=str, nargs="+", default=DEFAULT_OUTPUT_NAMES, help="Output variable names.")

    parser.add_argument("--num_train", type=int, default=12, help="Number of HF training samples.")
    parser.add_argument("--num_test", type=int, default=10, help="Number of HF test samples.")
    parser.add_argument("--num_lf", type=int, default=80, help="Number of LF samples.")
    parser.add_argument("--num_hf", type=int, default=30, help="Number of MF HF samples.")
    parser.add_argument("--num_active_initial", type=int, default=2, help="Number of initial HF samples for engineering active learning.")
    parser.add_argument("--num_infill", type=int, default=10, help="Number of active-learning infill iterations.")
    parser.add_argument("--lhs_iterations", type=int, default=50, help="Maximin LHS search iterations.")

    parser.add_argument("--ensemble_threshold", type=float, default=0.5, help="Threshold used by TAHS and AES-MSI.")
    parser.add_argument(
        "--ensemble_min_relative_gain",
        type=float,
        default=DEFAULT_THRESHOLD_PARAMS["ensemble_min_relative_gain"],
        help="Minimum relative accuracy gain required by ensemble models.",
    )
    parser.add_argument(
        "--mf_min_accuracy",
        type=float,
        default=DEFAULT_THRESHOLD_PARAMS["mf_min_accuracy"],
        help="Minimum accuracy required by multi-fidelity models.",
    )
    parser.add_argument(
        "--active_learning_min_relative_gain",
        type=float,
        default=DEFAULT_THRESHOLD_PARAMS["active_learning_min_relative_gain"],
        help="Minimum relative accuracy gain required after DISO infill.",
    )
    parser.add_argument("--metric_eps", type=float, default=1.0e-12, help="Stability epsilon for metrics.")

    parser.add_argument("--krg_poly", type=str, default="constant", help="KRG regression basis.")
    parser.add_argument("--krg_kernel", type=str, default="gaussian", help="KRG correlation kernel.")
    parser.add_argument("--krg_theta0", type=float, default=1.0, help="Initial KRG theta.")
    parser.add_argument(
        "--krg_theta_bounds",
        type=float,
        nargs=2,
        default=[1.0e-6, 100.0],
        help="Lower and upper bounds for KRG theta.",
    )

    parser.add_argument("--mf_poly_degree", type=int, default=2, help="Polynomial degree for MFS-MLS.")
    parser.add_argument(
        "--mf_sigma_bounds",
        type=float,
        nargs=2,
        default=[0.01, 10.0],
        help="Sigma bounds for MMFS.",
    )
    parser.add_argument("--infill_criterion", type=str, default="ei", choices=["ei", "poi", "lcb", "mse"], help="Base DISO acquisition.")
    parser.add_argument("--diso_alpha", type=float, default=4.0, help="Distance penalty intensity for DISO infill.")
    parser.add_argument(
        "--diso_min_distance",
        type=float,
        default=0.02,
        help="Minimum normalized distance to existing samples for DISO infill.",
    )
    parser.add_argument(
        "--diso_distance_scale",
        type=float,
        default=None,
        help="Optional characteristic distance h for DISO infill.",
    )

    parser.add_argument("--opt_target", type=str, default="stress_skin", choices=list(TARGET_SPECS.keys()), help="Optimization objective output.")
    parser.add_argument("--opt_constraint_target", type=str, default="weight", choices=list(TARGET_SPECS.keys()), help="Constraint output.")
    parser.add_argument(
        "--opt_constraint_ub",
        type=float,
        default=DEFAULT_THRESHOLD_PARAMS["weight_constraint_ub"],
        help="Upper bound for the optimization constraint.",
    )

    parser.add_argument("--miga_popsize", type=int, default=DEFAULT_MIGA_PARAMS["popsize"], help="MIGA population multiplier.")
    parser.add_argument("--miga_maxiter", type=int, default=DEFAULT_MIGA_PARAMS["maxiter"], help="Maximum MIGA iterations.")
    parser.add_argument("--miga_num_islands", type=int, default=DEFAULT_MIGA_PARAMS["num_islands"], help="Number of MIGA islands.")
    parser.add_argument(
        "--miga_migration_interval",
        type=int,
        default=DEFAULT_MIGA_PARAMS["migration_interval"],
        help="MIGA migration interval.",
    )
    parser.add_argument(
        "--miga_migration_size",
        type=int,
        default=DEFAULT_MIGA_PARAMS["migration_size"],
        help="Number of migrants exchanged by MIGA.",
    )
    parser.add_argument("--df_popsize", type=int, default=DEFAULT_DRAGONFLY_PARAMS["popsize"], help="CFSSDA population multiplier.")
    parser.add_argument("--df_maxiter", type=int, default=DEFAULT_DRAGONFLY_PARAMS["maxiter"], help="Maximum CFSSDA iterations.")
    parser.add_argument("--opt_tol", type=float, default=1.0e-6, help="Stopping tolerance for the optimizers.")

    args = parser.parse_args()
    args.demos = _expand_demo_selection(args.demos)
    args.targets = _expand_target_selection(args.targets)
    args.bounds = np.asarray(list(zip(args.bounds_lower, args.bounds_upper)), dtype=np.float64)
    args.krg_params = {
        "poly": args.krg_poly,
        "kernel": args.krg_kernel,
        "theta0": args.krg_theta0,
        "theta_bounds": tuple(args.krg_theta_bounds),
    }
    args.miga_params = {
        "popsize": args.miga_popsize,
        "maxiter": args.miga_maxiter,
        "num_islands": args.miga_num_islands,
        "migration_interval": args.miga_migration_interval,
        "migration_size": args.miga_migration_size,
    }
    args.df_params = {
        "popsize": args.df_popsize,
        "maxiter": args.df_maxiter,
    }
    return args
