# Engineering case workflow configuration for SurrogateLab
# Author: Shengning Wang

from __future__ import annotations
import argparse
from collections import OrderedDict
from typing import List

import numpy as np


ALGORITHM_ORDER = ["TAHS", "AESMSI", "MFSMLS", "MMFS", "CCAMFS", "DISO", "MIGA", "CFARSSDA"]
DEFAULT_DEMOS = list(ALGORITHM_ORDER)

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
DEFAULT_SAMPLE_COUNTS = {
    "num_train": 30,
    "num_test": 50,
    "num_lf": 30,
    "num_hf": 15,
    "num_active_initial": 3,
    "num_infill": 21,
}


def _expand_target_selection(selection: List[str]) -> List[str]:
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
    Parse command-line arguments for the engineering case workflow.

    Returns:
        argparse.Namespace: Parsed case configuration.
    """
    parser = argparse.ArgumentParser(description="SurrogateLab engineering case workflow runner.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seed_mode", type=str, default="single_seed", help="Run mode written into payload metadata.")
    parser.add_argument(
        "--demos",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=DEFAULT_DEMOS,
        help="Algorithms to run: TAHS AESMSI MFSMLS MMFS CCAMFS DISO MIGA CFARSSDA.",
    )
    parser.add_argument("--targets", nargs="+", default=["all"], help="Engineering targets: weight / stress_skin / all.")

    parser.add_argument("--num_features", type=int, default=3, help="Number of design variables.")
    parser.add_argument("--num_outputs", type=int, default=4, help="Number of simulation outputs.")
    parser.add_argument("--bounds_lower", type=float, nargs="+", default=DEFAULT_BOUNDS_LOWER, help="Lower bounds.")
    parser.add_argument("--bounds_upper", type=float, nargs="+", default=DEFAULT_BOUNDS_UPPER, help="Upper bounds.")
    parser.add_argument("--input_names", type=str, nargs="+", default=DEFAULT_INPUT_NAMES, help="Input variable names.")
    parser.add_argument("--output_names", type=str, nargs="+", default=DEFAULT_OUTPUT_NAMES, help="Output variable names.")

    parser.add_argument("--num_train", type=int, default=None, help="Number of HF training samples for ensemble and optimization.")
    parser.add_argument("--num_test", type=int, default=None, help="Number of HF test samples.")
    parser.add_argument("--num_lf", type=int, default=None, help="Number of LF samples for multi-fidelity runs.")
    parser.add_argument("--num_hf", type=int, default=None, help="Number of HF samples for multi-fidelity runs.")
    parser.add_argument("--num_active_initial", type=int, default=None, help="Number of initial HF samples for active learning.")
    parser.add_argument("--num_infill", type=int, default=None, help="Number of active-learning infill iterations.")
    parser.add_argument("--lhs_iterations", type=int, default=50, help="Maximin LHS search iterations.")

    parser.add_argument("--ensemble_threshold", type=float, default=0.5, help="Threshold used by TAHS and AESMSI.")
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

    parser.add_argument("--mf_poly_degree", type=int, default=2, help="Polynomial degree for MFSMLS.")
    parser.add_argument(
        "--mfs_mls_neighbor_factor",
        type=float,
        default=2.0,
        help="Neighborhood expansion factor used by MFSMLS local fitting.",
    )
    parser.add_argument(
        "--mfs_mls_ridge",
        type=float,
        default=1.0e-4,
        help="Ridge regularization used by MFSMLS local fitting.",
    )
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
    parser.add_argument("--df_popsize", type=int, default=DEFAULT_DRAGONFLY_PARAMS["popsize"], help="CFARSSDA population multiplier.")
    parser.add_argument("--df_maxiter", type=int, default=DEFAULT_DRAGONFLY_PARAMS["maxiter"], help="Maximum CFARSSDA iterations.")
    parser.add_argument("--opt_tol", type=float, default=1.0e-6, help="Stopping tolerance for the optimizers.")

    args = parser.parse_args()
    args.num_train = DEFAULT_SAMPLE_COUNTS["num_train"] if args.num_train is None else args.num_train
    args.num_test = DEFAULT_SAMPLE_COUNTS["num_test"] if args.num_test is None else args.num_test
    args.num_lf = DEFAULT_SAMPLE_COUNTS["num_lf"] if args.num_lf is None else args.num_lf
    args.num_hf = DEFAULT_SAMPLE_COUNTS["num_hf"] if args.num_hf is None else args.num_hf
    args.num_active_initial = DEFAULT_SAMPLE_COUNTS["num_active_initial"] if args.num_active_initial is None else args.num_active_initial
    args.num_infill = DEFAULT_SAMPLE_COUNTS["num_infill"] if args.num_infill is None else args.num_infill

    args.workflow = "case"
    args.demos = list(dict.fromkeys(args.demos))
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
