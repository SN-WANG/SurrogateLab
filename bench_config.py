# Benchmark configuration for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Dict, Iterable, List


ALGORITHM_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

DEMO_ALIAS_MAP = {
    "all": ALGORITHM_ORDER,
    "ensemble": ["A", "B"],
    "multifidelity": ["C", "D", "E"],
    "active_learning": ["F", "G", "H"],
    "optimization": ["I", "J"],
}

DEFAULT_ENSEMBLE_CASES = OrderedDict(
    {
        "forrester": {"num_train": 12, "num_test": 200, "lhs_iterations": 30},
        "branin": {"num_train": 18, "num_test": 200, "lhs_iterations": 30},
        "hartman3": {"num_train": 24, "num_test": 200, "lhs_iterations": 30},
    }
)

DEFAULT_MULTIFIDELITY_CASES = OrderedDict(
    {
        "currin_exponential": {"num_lf": 60, "num_hf": 24, "num_test": 300, "lhs_iterations": 20},
        "branin": {"num_lf": 100, "num_hf": 36, "num_test": 300, "lhs_iterations": 20},
        "park91b": {"num_lf": 120, "num_hf": 45, "num_test": 300, "lhs_iterations": 20},
    }
)

DEFAULT_SINGLE_OBJECTIVE_ACTIVE_CASE = {
    "name": "branin",
    "num_initial": 12,
    "num_test": 300,
    "num_infill": 6,
    "criterion": "ei",
    "lhs_iterations": 30,
}

DEFAULT_MULTI_FIDELITY_ACTIVE_CASE = {
    "name": "currin_exponential",
    "num_hf_initial": 4,
    "num_lf": 60,
    "num_test": 300,
    "num_infill": 10,
    "ratio": 0.5,
    "lhs_iterations": 30,
}

DEFAULT_MULTI_OBJECTIVE_ACTIVE_CASE = {
    "name": "vlmop2",
    "num_initial": 2,
    "num_test": 500,
    "num_infill": 14,
    "num_samples": 3000,
    "num_candidates": 120,
    "num_restarts": 4,
    "beta": 0.3,
    "lhs_iterations": 30,
}

DEFAULT_OPTIMIZATION_CASES = OrderedDict(
    {
        "branin": {"num_train": 24, "lhs_iterations": 30},
        "hartman3": {"num_train": 36, "lhs_iterations": 30},
        "rastrigin": {"num_train": 30, "lhs_iterations": 30},
    }
)

DEFAULT_MIGA_PARAMS = {
    "popsize": 10,
    "maxiter": 30,
    "num_islands": 4,
    "migration_interval": 10,
    "migration_size": 2,
}

DEFAULT_DRAGONFLY_PARAMS = {
    "popsize": 20,
    "maxiter": 50,
}

DEFAULT_THRESHOLD_PARAMS = {
    "ensemble_min_relative_gain": 0.10,
    "mf_min_accuracy": 87.0,
    "active_learning_min_relative_gain": 0.20,
}


def _expand_demo_selection(selection: Iterable[str]) -> List[str]:
    """
    Expand demo aliases into ordered algorithm labels.

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


def _expand_case_selection(selection: List[str], defaults: Dict[str, dict]) -> List[str]:
    """
    Expand case aliases into ordered case names.

    Args:
        selection (List[str]): Requested case names.
        defaults (Dict[str, dict]): Default ordered case registry.

    Returns:
        List[str]: Normalized case names.
    """
    if len(selection) == 1 and selection[0].lower() == "all":
        return list(defaults.keys())

    normalized = [item.lower() for item in selection]
    unknown = [item for item in normalized if item not in defaults]
    if unknown:
        valid = ", ".join(defaults.keys())
        raise ValueError(f"Unknown case(s): {unknown}. Valid choices: {valid}.")
    return normalized


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the analytic benchmark suite.

    Returns:
        argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(description="SurrogateLab analytic benchmark runner.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_dir", type=str, default="./benchmark_outputs", help="Directory for JSON reports.")
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["all"],
        help="Algorithms to run: A-J or aliases all / ensemble / multifidelity / active_learning / optimization.",
    )
    parser.add_argument("--ensemble_cases", nargs="+", default=["all"], help="Ensemble benchmark cases.")
    parser.add_argument("--multifidelity_cases", nargs="+", default=["all"], help="Multi-fidelity benchmark cases.")
    parser.add_argument("--optimization_cases", nargs="+", default=["all"], help="Optimization benchmark cases.")
    parser.add_argument("--lhs_iterations", type=int, default=30, help="Maximin LHS search iterations.")

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
        help="Minimum accuracy required by each multi-fidelity surrogate case.",
    )
    parser.add_argument(
        "--active_learning_min_relative_gain",
        type=float,
        default=DEFAULT_THRESHOLD_PARAMS["active_learning_min_relative_gain"],
        help="Minimum relative accuracy gain required after infill.",
    )
    parser.add_argument("--metric_eps", type=float, default=1.0e-12, help="Stability epsilon for metrics.")

    parser.add_argument("--krg_poly", type=str, default="constant", help="Kriging regression basis.")
    parser.add_argument("--krg_kernel", type=str, default="gaussian", help="Kriging correlation kernel.")
    parser.add_argument("--krg_theta0", type=float, default=1.0, help="Initial Kriging theta.")
    parser.add_argument(
        "--krg_theta_bounds",
        type=float,
        nargs=2,
        default=[1.0e-6, 100.0],
        help="Lower and upper bounds for Kriging theta.",
    )

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
        help="Optional distance scale h for DISO infill. Defaults to the sampled nearest-distance scale.",
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
    args.ensemble_cases = _expand_case_selection(args.ensemble_cases, DEFAULT_ENSEMBLE_CASES)
    args.multifidelity_cases = _expand_case_selection(args.multifidelity_cases, DEFAULT_MULTIFIDELITY_CASES)
    args.optimization_cases = _expand_case_selection(args.optimization_cases, DEFAULT_OPTIMIZATION_CASES)

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
