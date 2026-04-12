# Engineering case workflow configuration for SurrogateLab
# Author: Shengning Wang

import argparse
from collections import OrderedDict
from typing import List

import numpy as np


# ============================================================
# General
# ============================================================

ALGORITHM_ORDER = ["TAHS", "AESMSI", "MFSMLS", "MMFS", "CCAMFS", "DISO", "MIGA", "CFARSSDA"]


# ============================================================
# Case Registry
# ============================================================

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


# ============================================================
# Design Space
# ============================================================

DESIGN_SPACE = {
    "num_features": 3,
    "num_outputs": 4,
    "bounds_lower": [4.0, 4.0, 4.0],
    "bounds_upper": [10.0, 10.0, 10.0],
    "input_names": ["thick1", "thick2", "thick3"],
    "output_names": ["weight", "displacement", "stress_skin", "stress_stiff"],
}


# ============================================================
# DOE
# ============================================================

SAMPLE_COUNTS = {
    "num_train": 30,
    "num_test": 50,
    "num_lf": 30,
    "num_hf": 15,
    "num_active_initial": 2,
    "num_infill": 21,
}


def _expand_target_selection(selection: List[str]) -> List[str]:
    """
    Expand the requested target selection into ordered target names.

    Args:
        selection (List[str]): Requested target names.

    Returns:
        List[str]: Normalized target names.
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
    Parse command-line arguments for the engineering case workflow.

    Returns:
        argparse.Namespace: Parsed case configuration.
    """
    parser = argparse.ArgumentParser(
        description="SurrogateLab engineering case workflow runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # General
    # ============================================================

    general = parser.add_argument_group("General")
    general.add_argument("--seed", type=int, default=7, help="Random seed.")
    general.add_argument("--seed_mode", type=str, default="single_seed", help="Run mode written into payload metadata.")
    general.add_argument(
        "--demos",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=list(ALGORITHM_ORDER),
        help="Algorithms to run: TAHS AESMSI MFSMLS MMFS CCAMFS DISO MIGA CFARSSDA.",
    )
    general.add_argument("--targets", nargs="+", default=["all"], help="Engineering targets: weight / stress_skin / all.")

    # ============================================================
    # Design Space
    # ============================================================

    design = parser.add_argument_group("Design Space")
    design.add_argument("--num_features", type=int, default=DESIGN_SPACE["num_features"], help="Number of design variables.")
    design.add_argument("--num_outputs", type=int, default=DESIGN_SPACE["num_outputs"], help="Number of simulation outputs.")
    design.add_argument("--bounds_lower", type=float, nargs="+", default=DESIGN_SPACE["bounds_lower"], help="Lower bounds.")
    design.add_argument("--bounds_upper", type=float, nargs="+", default=DESIGN_SPACE["bounds_upper"], help="Upper bounds.")
    design.add_argument("--input_names", type=str, nargs="+", default=DESIGN_SPACE["input_names"], help="Input variable names.")
    design.add_argument("--output_names", type=str, nargs="+", default=DESIGN_SPACE["output_names"], help="Output variable names.")

    # ============================================================
    # DOE
    # ============================================================

    doe = parser.add_argument_group("DOE")
    doe.add_argument(
        "--num_train",
        type=int,
        default=SAMPLE_COUNTS["num_train"],
        help="Number of HF training samples for ensemble and optimization.",
    )
    doe.add_argument("--num_test", type=int, default=SAMPLE_COUNTS["num_test"], help="Number of HF test samples.")
    doe.add_argument("--num_lf", type=int, default=SAMPLE_COUNTS["num_lf"], help="Number of LF samples for multi-fidelity runs.")
    doe.add_argument("--num_hf", type=int, default=SAMPLE_COUNTS["num_hf"], help="Number of HF samples for multi-fidelity runs.")
    doe.add_argument(
        "--num_active_initial",
        type=int,
        default=SAMPLE_COUNTS["num_active_initial"],
        help="Number of initial HF samples for active learning.",
    )
    doe.add_argument("--num_infill", type=int, default=SAMPLE_COUNTS["num_infill"], help="Number of active-learning infill iterations.")
    doe.add_argument("--doe_seed", type=int, default=42, help="Dedicated random seed used only by the engineering DOE generator.")

    # ============================================================
    # Baseline and Ensemble
    # ============================================================

    ensemble = parser.add_argument_group("Baseline and Ensemble")
    ensemble.add_argument("--ensemble_threshold", type=float, default=0.5, help="Threshold used by TAHS and AESMSI.")
    ensemble.add_argument(
        "--ensemble_min_relative_gain",
        type=float,
        default=0.10,
        help="Minimum relative accuracy gain required by ensemble models.",
    )
    ensemble.add_argument("--metric_eps", type=float, default=1.0e-12, help="Stability epsilon for metrics.")

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
        help="Minimum accuracy required by multi-fidelity models.",
    )
    multifidelity.add_argument("--mf_poly_degree", type=int, default=2, help="Polynomial degree for MFSMLS.")
    multifidelity.add_argument(
        "--mfs_mls_neighbor_factor",
        type=float,
        default=2.0,
        help="Neighborhood expansion factor used by MFSMLS local fitting.",
    )
    multifidelity.add_argument(
        "--mfs_mls_ridge",
        type=float,
        default=1.0e-4,
        help="Ridge regularization used by MFSMLS local fitting.",
    )
    multifidelity.add_argument(
        "--mf_sigma_bounds",
        type=float,
        nargs=2,
        default=[0.01, 10.0],
        help="Sigma bounds for MMFS.",
    )

    # ============================================================
    # Active Learning
    # ============================================================

    active = parser.add_argument_group("Active Learning")
    active.add_argument(
        "--active_learning_min_relative_gain",
        type=float,
        default=0.20,
        help="Minimum relative accuracy gain required after DISO infill.",
    )
    active.add_argument(
        "--infill_criterion",
        type=str,
        default="ei",
        choices=["ei", "poi", "lcb", "mse"],
        help="Base DISO acquisition.",
    )
    active.add_argument("--diso_alpha", type=float, default=4.0, help="Distance penalty intensity for DISO infill.")
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
        help="Optional characteristic distance h for DISO infill.",
    )

    # ============================================================
    # Optimization
    # ============================================================

    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--opt_target",
        type=str,
        default="stress_skin",
        choices=list(TARGET_SPECS.keys()),
        help="Optimization objective output.",
    )
    optim.add_argument(
        "--opt_constraint_target",
        type=str,
        default="weight",
        choices=list(TARGET_SPECS.keys()),
        help="Constraint output.",
    )
    optim.add_argument(
        "--opt_constraint_ub",
        type=float,
        default=0.31,
        help="Upper bound for the optimization constraint.",
    )
    optim.add_argument("--miga_popsize", type=int, default=10, help="MIGA population multiplier.")
    optim.add_argument("--miga_maxiter", type=int, default=50, help="Maximum MIGA iterations.")
    optim.add_argument("--miga_num_islands", type=int, default=4, help="Number of MIGA islands.")
    optim.add_argument(
        "--miga_migration_interval",
        type=int,
        default=10,
        help="MIGA migration interval.",
    )
    optim.add_argument(
        "--miga_migration_size",
        type=int,
        default=2,
        help="Number of migrants exchanged by MIGA.",
    )
    optim.add_argument("--df_popsize", type=int, default=20, help="CFARSSDA population multiplier.")
    optim.add_argument("--df_maxiter", type=int, default=120, help="Maximum CFARSSDA iterations.")
    optim.add_argument("--opt_tol", type=float, default=1.0e-6, help="Stopping tolerance for the optimizers.")

    args = parser.parse_args()
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
