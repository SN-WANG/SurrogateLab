"""Configuration for the Abaqus benchmark platform."""

from __future__ import annotations

import argparse

import numpy as np


ALGORITHM_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

DEMO_ALIAS_MAP = {
    "all": ALGORITHM_ORDER,
    "ensemble": ["A", "B"],
    "multifidelity": ["C", "D", "E"],
    "sequential": ["F", "G", "H"],
    "optimization": ["I", "J"],
}

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


def _expand_demo_selection(selection: list[str]) -> list[str]:
    """Expand demo aliases into ordered demo labels."""
    expanded: list[str] = []
    for item in selection:
        key = item.lower()
        if key in DEMO_ALIAS_MAP:
            expanded.extend(DEMO_ALIAS_MAP[key])
        else:
            expanded.append(item.upper())

    ordered: list[str] = []
    for label in ALGORITHM_ORDER:
        if label in expanded:
            ordered.append(label)
    return ordered


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the Abaqus benchmark platform.

    Returns:
        argparse.Namespace: Parsed arguments with convenience parameter groups.
    """
    parser = argparse.ArgumentParser(
        description="Aero optimization solver benchmark platform."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory for cached data and plot outputs.",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["all"],
        help=(
            "Demos to run: A B C D E F G H I J or group aliases "
            "(all, ensemble, multifidelity, sequential, optimization)."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Enable matplotlib visualization saved to save_dir.",
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=3,
        help="Number of design variables.",
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        default=4,
        help="Number of simulation outputs.",
    )
    parser.add_argument(
        "--bounds_lower",
        type=float,
        nargs="+",
        default=DEFAULT_BOUNDS_LOWER,
        help="Lower bounds for each design variable.",
    )
    parser.add_argument(
        "--bounds_upper",
        type=float,
        nargs="+",
        default=DEFAULT_BOUNDS_UPPER,
        help="Upper bounds for each design variable.",
    )
    parser.add_argument(
        "--input_names",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_NAMES,
        help="Names of design variables.",
    )
    parser.add_argument(
        "--output_names",
        type=str,
        nargs="+",
        default=DEFAULT_OUTPUT_NAMES,
        help="Names of simulation outputs.",
    )

    parser.add_argument(
        "--num_train",
        type=int,
        default=20,
        help="Number of high-fidelity training samples.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=10,
        help="Number of test samples.",
    )
    parser.add_argument(
        "--num_lf",
        type=int,
        default=30,
        help="Number of low-fidelity samples.",
    )
    parser.add_argument(
        "--num_hf",
        type=int,
        default=10,
        help="Number of high-fidelity multi-fidelity samples.",
    )
    parser.add_argument(
        "--num_infill",
        type=int,
        default=5,
        help="Number of sequential infill iterations.",
    )
    parser.add_argument(
        "--lhs_iterations",
        type=int,
        default=50,
        help="Maximin LHS optimization iterations.",
    )

    parser.add_argument(
        "--krg_poly",
        type=str,
        default="constant",
        help="KRG regression type.",
    )
    parser.add_argument(
        "--krg_kernel",
        type=str,
        default="gaussian",
        help="KRG correlation kernel.",
    )
    parser.add_argument(
        "--krg_theta0",
        type=float,
        default=1.0,
        help="KRG initial correlation parameter.",
    )
    parser.add_argument(
        "--krg_theta_bounds",
        type=float,
        nargs=2,
        default=[1.0e-6, 100.0],
        help="KRG theta optimization bounds [lower, upper].",
    )

    parser.add_argument(
        "--ensemble_threshold",
        type=float,
        default=0.5,
        help="Filtering threshold for TAHS and AESMSI.",
    )
    parser.add_argument(
        "--mf_poly_degree",
        type=int,
        default=2,
        help="Polynomial degree for MFS-MLS.",
    )
    parser.add_argument(
        "--mf_sigma_bounds",
        type=float,
        nargs=2,
        default=[0.01, 10.0],
        help="Sigma bounds for MMFS.",
    )
    parser.add_argument(
        "--infill_criterion",
        type=str,
        default="ei",
        choices=["ei", "poi", "lcb", "mse"],
        help="Acquisition function for single-objective infill.",
    )
    parser.add_argument(
        "--mico_ratio",
        type=float,
        default=0.5,
        help="Mutual-information versus distance trade-off ratio.",
    )

    parser.add_argument(
        "--opt_single_idx",
        type=int,
        default=2,
        help="Output index for single-objective optimization.",
    )
    parser.add_argument(
        "--obj_indices",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Output indices for multi-objective optimization.",
    )
    parser.add_argument(
        "--constraint_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Output indices used as constraints.",
    )
    parser.add_argument(
        "--constraint_percentile",
        type=float,
        default=50.0,
        help="Percentile of training data used for the constraint upper bound.",
    )
    parser.add_argument(
        "--miga_popsize",
        type=int,
        default=DEFAULT_MIGA_PARAMS["popsize"],
        help="Population size multiplier for MIGA.",
    )
    parser.add_argument(
        "--miga_maxiter",
        type=int,
        default=DEFAULT_MIGA_PARAMS["maxiter"],
        help="Maximum iterations for MIGA.",
    )
    parser.add_argument(
        "--miga_num_islands",
        type=int,
        default=DEFAULT_MIGA_PARAMS["num_islands"],
        help="Number of islands used by MIGA.",
    )
    parser.add_argument(
        "--miga_migration_interval",
        type=int,
        default=DEFAULT_MIGA_PARAMS["migration_interval"],
        help="Migration interval for MIGA.",
    )
    parser.add_argument(
        "--miga_migration_size",
        type=int,
        default=DEFAULT_MIGA_PARAMS["migration_size"],
        help="Number of migrants exchanged by MIGA.",
    )
    parser.add_argument(
        "--df_popsize",
        type=int,
        default=DEFAULT_DRAGONFLY_PARAMS["popsize"],
        help="Population size multiplier for CFSSDA.",
    )
    parser.add_argument(
        "--df_maxiter",
        type=int,
        default=DEFAULT_DRAGONFLY_PARAMS["maxiter"],
        help="Maximum iterations for CFSSDA.",
    )
    parser.add_argument(
        "--opt_tol",
        type=float,
        default=1.0e-6,
        help="Convergence tolerance for the global optimizers.",
    )

    args = parser.parse_args()
    args.bounds = np.asarray(
        list(zip(args.bounds_lower, args.bounds_upper)),
        dtype=np.float64,
    )
    args.demos = _expand_demo_selection(args.demos)
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
