# Unified workflow entry for SurrogateLab quality evaluation
# Author: Shengning Wang

import argparse
import json
import shlex
from pathlib import Path
from typing import Optional, Sequence

from utils.module_quality_gate import (
    DEFAULT_COVERAGE_DATA_FILE,
    DEFAULT_SOURCE_ROOTS,
    PROJECT_ROOT,
    Thresholds,
    build_result_payload,
    entry_passed,
    evaluate_dynamic_sequence,
    print_entry_coverage_report,
    print_entry_details,
    print_entry_metrics,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the dynamic quality gate.

    Args:
        argv (Optional[Sequence[str]]): Optional CLI argument list.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run case_main.py and bench_main.py sequentially, collect dynamic coverage with "
            "coverage.py, then print the three metrics for each workflow."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Defaults to the parent of utils/.",
    )
    parser.add_argument(
        "--source-roots",
        nargs="+",
        default=list(DEFAULT_SOURCE_ROOTS),
        help="Source roots used to discover target modules.",
    )
    parser.add_argument(
        "--case-args",
        type=str,
        default="",
        help="Extra command-line arguments passed to case_main.py.",
    )
    parser.add_argument(
        "--bench-args",
        type=str,
        default="",
        help="Extra command-line arguments passed to bench_main.py.",
    )
    parser.add_argument(
        "--coverage-data-file",
        type=Path,
        default=Path(DEFAULT_COVERAGE_DATA_FILE),
        help="coverage.py data file, equivalent to --data-file for coverage run/report.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum dynamic coverage percentage.",
    )
    parser.add_argument(
        "--comment-threshold",
        type=float,
        default=30.0,
        help="Minimum comment-rate percentage.",
    )
    parser.add_argument(
        "--complexity-threshold",
        type=float,
        default=25.0,
        help="Maximum average cyclomatic complexity.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also print per-module details for each workflow.",
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Also print the raw coverage.py text report equivalent to coverage report -m.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON payload after the workflow outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the sequential dynamic-quality workflow from the command line.

    Args:
        argv (Optional[Sequence[str]]): Optional CLI argument list.

    Returns:
        int: Shell-style process exit code.
    """
    args = parse_args(argv)
    thresholds = Thresholds(
        coverage=args.coverage_threshold,
        comment_rate=args.comment_threshold,
        average_complexity=args.complexity_threshold,
    )
    case_args = shlex.split(args.case_args)
    bench_args = shlex.split(args.bench_args)

    results = evaluate_dynamic_sequence(
        case_args=case_args,
        bench_args=bench_args,
        project_root=args.project_root.resolve(),
        source_roots=args.source_roots,
        coverage_data_file=args.coverage_data_file,
    )

    if args.json:
        print(json.dumps(build_result_payload(results, thresholds), indent=2))
    else:
        for result in results:
            print_entry_metrics(result, thresholds)
            if args.coverage_report:
                print_entry_coverage_report(result)
        if args.details:
            for result in results:
                print_entry_details(result, project_root=args.project_root.resolve())

    passed = all(entry_passed(result, thresholds) for result in results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
