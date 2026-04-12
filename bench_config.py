# Analytic benchmark configuration for SurrogateLab
# Author: Shengning Wang

import argparse
from collections import OrderedDict
from typing import Dict, List


# ============================================================
# General
# ============================================================

ALGORITHM_ORDER = ["TAHS", "AESMSI", "MFSMLS", "MMFS", "CCAMFS", "DISO", "MICO", "MOBO", "MIGA", "CFARSSDA"]
SEEDS = list(range(1, 11))


# ============================================================
# Benchmark Cases
# ============================================================

ENSEMBLE_CASES = OrderedDict(
    {
        "forrester": {"num_train": 10, "num_test": 200},
        "hartman3": {"num_train": 30, "num_test": 200},
        "rosenbrock5": {"num_train": 50, "num_test": 200},
    }
)

MULTIFIDELITY_CASES = OrderedDict(
    {
        "borehole": {"num_lf": 80, "num_hf": 40, "num_test": 200},
        "currin_exponential": {"num_lf": 20, "num_hf": 10, "num_test": 200},
        "park91b": {"num_lf": 40, "num_hf": 20, "num_test": 200},
    }
)

ACTIVE_LEARNING_CASES = {
    "single_objective": {
        "name": "branin",
        "num_initial": 2,
        "num_test": 200,
        "num_infill": 14,
        "criterion": "ei",
    },
    "multi_fidelity": {
        "name": "currin_exponential",
        "num_hf_initial": 2,
        "num_lf": 20,
        "num_test": 200,
        "num_infill": 14,
        "ratio": 0.5,
    },
    "multi_objective": {
        "name": "vlmop2",
        "num_initial": 2,
        "num_test": 200,
        "num_infill": 14,
        "num_samples": 3000,
        "num_candidates": 120,
        "num_restarts": 4,
        "beta": 0.3,
    },
}

OPTIMIZATION_CASES = OrderedDict(
    {
        "branin": {"num_train": 20},
        "hartman3": {"num_train": 30},
        "rastrigin": {"num_train": 20},
    }
)


def _expand_case_selection(selection: List[str], registry: Dict[str, dict]) -> List[str]:
    """
    Expand case selection into ordered case names.

    Args:
        selection (List[str]): Requested case names.
        registry (Dict[str, dict]): Ordered case registry.

    Returns:
        List[str]: Normalized case names.
    """
    if len(selection) == 1 and selection[0].lower() == "all":
        return list(registry.keys())

    normalized = [item.lower() for item in selection]
    unknown = [item for item in normalized if item not in registry]
    if unknown:
        valid = ", ".join(registry.keys())
        raise ValueError(f"Unknown case(s): {unknown}. Valid choices: {valid}.")
    return normalized


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the analytic benchmark suite.

    Returns:
        argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(
        description="SurrogateLab analytic benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # General
    # ============================================================

    general = parser.add_argument_group("General")
    general.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seed list used by the benchmark suite.")
    general.add_argument(
        "--demos",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=list(ALGORITHM_ORDER),
        help=(
            "Algorithms to run: TAHS, AESMSI, MFSMLS, MMFS, CCAMFS, DISO, "
            "MICO, MOBO, MIGA, CFARSSDA."
        ),
    )

    # ============================================================
    # Benchmark Cases
    # ============================================================

    cases = parser.add_argument_group("Case Selection")
    cases.add_argument("--ensemble_cases", nargs="+", default=["all"], help="Ensemble benchmark cases.")
    cases.add_argument("--multifidelity_cases", nargs="+", default=["all"], help="Multi-fidelity benchmark cases.")
    cases.add_argument("--optimization_cases", nargs="+", default=["all"], help="Optimization benchmark cases.")

    # ============================================================
    # Baseline and Ensemble
    # ============================================================

    ensemble = parser.add_argument_group("Baseline and Ensemble")
    ensemble.add_argument("--ensemble_threshold", type=float, default=0.5, help="Threshold used by TAHS and AES-MSI.")
    ensemble.add_argument(
        "--ensemble_min_relative_gain",
        type=float,
        default=0.10,
        help="Minimum relative accuracy gain required by ensemble models.",
    )

    # ============================================================
    # KRG
    # ============================================================

    kriging = parser.add_argument_group("Kriging")
    kriging.add_argument("--krg_poly", type=str, default="constant", help="KRG regression basis.")
    kriging.add_argument("--krg_kernel", type=str, default="gaussian", help="KRG correlation kernel.")
    kriging.add_argument("--krg_theta0", type=float, default=0.1, help="Initial KRG theta.")
    kriging.add_argument(
        "--krg_theta_bounds",
        type=float,
        nargs=2,
        default=[1.0e-3, 10.0],
        help="Lower and upper bounds for KRG theta.",
    )

    # ============================================================
    # PRS
    # ============================================================

    prs = parser.add_argument_group("PRS")
    prs.add_argument("--prs_degree", type=int, default=5, help="Polynomial degree for PRS.")
    prs.add_argument("--prs_alpha", type=float, default=0.0, help="Ridge regularization for PRS.")

    # ============================================================
    # SVR
    # ============================================================

    svr = parser.add_argument_group("SVR")
    svr.add_argument("--svr_kernel", type=str, default="linear", choices=["rbf", "linear"], help="SVR kernel type.")
    svr.add_argument("--svr_gamma", type=float, default=None, help="SVR kernel coefficient for the rbf kernel.")
    svr.add_argument("--svr_C", type=float, default=0.1, help="SVR regularization parameter.")
    svr.add_argument("--svr_epsilon", type=float, default=2.0, help="SVR epsilon-insensitive tube width.")

    # ============================================================
    # Multi-Fidelity
    # ============================================================

    multifidelity = parser.add_argument_group("Multi-Fidelity")
    multifidelity.add_argument(
        "--mf_min_accuracy",
        type=float,
        default=90.0,
        help="Minimum accuracy required by each multi-fidelity surrogate case.",
    )
    multifidelity.add_argument(
        "--mfs_mls_neighbor_factor",
        type=float,
        default=2.0,
        help="Neighborhood expansion factor used by MFS-MLS local fitting.",
    )
    multifidelity.add_argument(
        "--mfs_mls_ridge",
        type=float,
        default=1.0e-4,
        help="Ridge regularization used by MFS-MLS local fitting.",
    )

    # ============================================================
    # Active Learning
    # ============================================================

    active = parser.add_argument_group("Active Learning")
    active.add_argument(
        "--active_learning_min_relative_gain",
        type=float,
        default=0.20,
        help="Minimum relative accuracy gain required after infill.",
    )
    active.add_argument(
        "--diso_alpha",
        type=float,
        default=4.0,
        help="Distance penalty intensity for DISO infill.",
    )
    active.add_argument(
        "--diso_min_distance",
        type=float,
        default=0.02,
        help="Minimum normalized distance to existing samples for DISO infill.",
    )
    active.add_argument(
        "--diso_distance_scale",
        type=float,
        default=None,
        help="Optional distance scale h for DISO infill. Defaults to the sampled nearest-distance scale.",
    )

    # ============================================================
    # Optimization
    # ============================================================

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--miga_popsize",
        type=int,
        default=10,
        help="MIGA population multiplier.",
    )
    optimization.add_argument(
        "--miga_maxiter",
        type=int,
        default=30,
        help="Maximum MIGA iterations.",
    )
    optimization.add_argument(
        "--miga_num_islands",
        type=int,
        default=4,
        help="Number of MIGA islands.",
    )
    optimization.add_argument(
        "--miga_migration_interval",
        type=int,
        default=10,
        help="MIGA migration interval.",
    )
    optimization.add_argument(
        "--miga_migration_size",
        type=int,
        default=2,
        help="Number of migrants exchanged by MIGA.",
    )
    optimization.add_argument(
        "--df_popsize",
        type=int,
        default=20,
        help="CFARSSDA population multiplier.",
    )
    optimization.add_argument(
        "--df_maxiter",
        type=int,
        default=50,
        help="Maximum CFARSSDA iterations.",
    )
    optimization.add_argument("--opt_tol", type=float, default=1.0e-6, help="Stopping tolerance for the optimizers.")

    # ============================================================
    # Metrics
    # ============================================================

    metrics = parser.add_argument_group("Metrics")
    metrics.add_argument("--metric_eps", type=float, default=1.0e-12, help="Stability epsilon for metrics.")

    args = parser.parse_args()
    args.demos = list(dict.fromkeys(args.demos))
    args.ensemble_cases = _expand_case_selection(args.ensemble_cases, ENSEMBLE_CASES)
    args.multifidelity_cases = _expand_case_selection(args.multifidelity_cases, MULTIFIDELITY_CASES)
    args.optimization_cases = _expand_case_selection(args.optimization_cases, OPTIMIZATION_CASES)
    args.seeds = list(args.seeds)

    args.krg_params = {
        "poly": args.krg_poly,
        "kernel": args.krg_kernel,
        "theta0": args.krg_theta0,
        "theta_bounds": tuple(args.krg_theta_bounds),
    }
    args.prs_params = {
        "degree": args.prs_degree,
        "alpha": args.prs_alpha,
    }
    args.svr_params = {
        "kernel": args.svr_kernel,
        "gamma": args.svr_gamma,
        "C": args.svr_C,
        "epsilon": args.svr_epsilon,
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
