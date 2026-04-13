# Dynamic quality gate for SurrogateLab workflows
# Author: Shengning Wang

import argparse
import ast
import json
import os
import runpy
import shlex
import sys
import tokenize
import trace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hue_logger import hue


DEFAULT_SOURCE_ROOTS = ("models", "sampling")
KEEP_MODULE_NAMES = {"__main__"}
KEEP_MODULE_PREFIXES = ("utils.module_quality_gate",)


@dataclass
class Thresholds:
    """
    Thresholds used by the dynamic quality gate.
    """

    coverage: float = 80.0
    comment_rate: float = 30.0
    average_complexity: float = 25.0


@dataclass
class ModuleMetrics:
    """
    Dynamic coverage and static quality metrics for one module.
    """

    path: Path
    executable_lines: int
    covered_lines: int
    comment_lines: int
    relevant_lines: int
    block_count: int
    total_complexity: int

    @property
    def coverage_rate(self) -> float:
        if self.executable_lines == 0:
            return 100.0
        return 100.0 * self.covered_lines / self.executable_lines

    @property
    def comment_rate(self) -> float:
        if self.relevant_lines == 0:
            return 100.0
        return 100.0 * self.comment_lines / self.relevant_lines

    @property
    def average_complexity(self) -> float:
        if self.block_count == 0:
            return 1.0
        return self.total_complexity / self.block_count


@dataclass
class EntryMetrics:
    """
    Aggregated metrics for one entry workflow.
    """

    entry_file: str
    return_code: int
    module_count: int
    executable_lines: int
    covered_lines: int
    comment_lines: int
    relevant_lines: int
    block_count: int
    total_complexity: int
    module_metrics: List[ModuleMetrics]

    @property
    def coverage_rate(self) -> float:
        if self.executable_lines == 0:
            return 100.0
        return 100.0 * self.covered_lines / self.executable_lines

    @property
    def comment_rate(self) -> float:
        if self.relevant_lines == 0:
            return 100.0
        return 100.0 * self.comment_lines / self.relevant_lines

    @property
    def average_complexity(self) -> float:
        if self.block_count == 0:
            return 1.0
        return self.total_complexity / self.block_count


# ============================================================
# Discovery
# ============================================================

def iter_python_files(root: Path) -> Iterable[Path]:
    """
    Iterate over Python files under a directory.

    Args:
        root (Path): Directory to scan.

    Yields:
        Path: Python file path.
    """
    if not root.exists():
        return

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        yield path.resolve()


def resolve_local_module_path(module_name: str, project_root: Path) -> Optional[Path]:
    """
    Resolve a dotted module name to a local Python file.

    Args:
        module_name (str): Dotted module path.
        project_root (Path): Repository root.

    Returns:
        Optional[Path]: Resolved local module path.
    """
    parts = module_name.split(".")
    module_path = project_root.joinpath(*parts).with_suffix(".py")
    if module_path.exists():
        return module_path.resolve()

    package_init = project_root.joinpath(*parts, "__init__.py")
    if package_init.exists():
        return package_init.resolve()

    return None


def read_ast(path: Path) -> ast.Module:
    """
    Parse a Python file into an AST module.

    Args:
        path (Path): Python source path.

    Returns:
        ast.Module: Parsed module AST.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def resolve_relative_module(
    current_path: Path,
    project_root: Path,
    module: Optional[str],
    level: int,
) -> Optional[str]:
    """
    Resolve a relative import to a dotted local module name.

    Args:
        current_path (Path): Importing file path.
        project_root (Path): Repository root.
        module (Optional[str]): Relative module payload.
        level (int): Relative import level.

    Returns:
        Optional[str]: Resolved dotted module name.
    """
    current_parts = list(current_path.resolve().relative_to(project_root.resolve()).with_suffix("").parts)
    if len(current_parts) < level:
        return None

    anchor_parts = current_parts[:-level]
    module_parts = [] if module is None else module.split(".")
    resolved_parts = anchor_parts + module_parts
    if not resolved_parts:
        return None
    return ".".join(resolved_parts)


def collect_local_imports(path: Path, project_root: Path) -> Set[Path]:
    """
    Collect local module dependencies imported by one Python file.

    Args:
        path (Path): Source module path.
        project_root (Path): Repository root.

    Returns:
        Set[Path]: Imported local module files.
    """
    tree = read_ast(path)
    imported_paths: Set[Path] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_path = resolve_local_module_path(alias.name, project_root)
                if module_path is not None:
                    imported_paths.add(module_path)
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level > 0:
            base_name = resolve_relative_module(path, project_root, node.module, node.level)
        else:
            base_name = node.module

        if not base_name:
            continue

        base_path = resolve_local_module_path(base_name, project_root)
        if base_path is not None:
            imported_paths.add(base_path)

        for alias in node.names:
            if alias.name == "*":
                continue
            child_name = f"{base_name}.{alias.name}"
            child_path = resolve_local_module_path(child_name, project_root)
            if child_path is not None:
                imported_paths.add(child_path)

    return imported_paths


def discover_target_modules(project_root: Path, source_roots: Sequence[str]) -> List[Path]:
    """
    Discover target modules under the configured roots and their local dependencies.

    Args:
        project_root (Path): Repository root.
        source_roots (Sequence[str]): Root directories such as ``models`` and ``sampling``.

    Returns:
        List[Path]: Sorted module file paths participating in the gate.
    """
    target_paths: Set[Path] = set()
    pending_paths: List[Path] = []

    for root_name in source_roots:
        root_dir = project_root / root_name
        for module_path in iter_python_files(root_dir):
            if module_path in target_paths:
                continue
            target_paths.add(module_path)
            pending_paths.append(module_path)

    while pending_paths:
        current_path = pending_paths.pop()
        for import_path in collect_local_imports(current_path, project_root):
            if import_path in target_paths:
                continue
            target_paths.add(import_path)
            pending_paths.append(import_path)

    return sorted(target_paths)


def discover_entry_closure(entry_path: Path, project_root: Path) -> Set[Path]:
    """
    Discover the full local import closure of one entry file.

    Args:
        entry_path (Path): Entry file path.
        project_root (Path): Repository root.

    Returns:
        Set[Path]: Reachable local module files including the entry file.
    """
    reachable_paths: Set[Path] = {entry_path.resolve()}
    pending_paths: List[Path] = [entry_path.resolve()]

    while pending_paths:
        current_path = pending_paths.pop()
        for import_path in collect_local_imports(current_path, project_root):
            if import_path in reachable_paths:
                continue
            reachable_paths.add(import_path)
            pending_paths.append(import_path)

    return reachable_paths


# ============================================================
# Static Metrics
# ============================================================

def collect_docstring_lines(path: Path) -> Set[int]:
    """
    Collect physical lines occupied by docstrings.

    Args:
        path (Path): Python source path.

    Returns:
        Set[int]: Docstring line numbers.
    """
    docstring_lines: Set[int] = set()
    tree = read_ast(path)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            continue
        if not isinstance(first_stmt.value, ast.Constant) or not isinstance(first_stmt.value.value, str):
            continue
        start = first_stmt.lineno
        end = getattr(first_stmt, "end_lineno", first_stmt.lineno)
        docstring_lines.update(range(start, end + 1))

    return docstring_lines


def collect_hash_comment_lines(path: Path) -> Set[int]:
    """
    Collect physical lines occupied by ``#`` comments.

    Args:
        path (Path): Python source path.

    Returns:
        Set[int]: Hash-comment line numbers.
    """
    comment_lines: Set[int] = set()
    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.COMMENT:
                comment_lines.add(token.start[0])
    return comment_lines


def collect_code_lines(path: Path, docstring_lines: Set[int]) -> Set[int]:
    """
    Collect physical lines containing code tokens.

    Args:
        path (Path): Python source path.
        docstring_lines (Set[int]): Docstring line numbers.

    Returns:
        Set[int]: Code line numbers.
    """
    code_lines: Set[int] = set()
    ignored_token_types = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.ENCODING,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }

    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in ignored_token_types:
                continue
            for lineno in range(token.start[0], token.end[0] + 1):
                if lineno not in docstring_lines:
                    code_lines.add(lineno)

    return code_lines


def score_comment_rate(path: Path) -> Tuple[int, int]:
    """
    Count comment lines and relevant lines for the comment-rate metric.

    Args:
        path (Path): Python source path.

    Returns:
        Tuple[int, int]: Comment lines and relevant lines.
    """
    docstring_lines = collect_docstring_lines(path)
    hash_comment_lines = collect_hash_comment_lines(path)
    comment_lines = docstring_lines | hash_comment_lines
    code_lines = collect_code_lines(path, docstring_lines)
    relevant_lines = code_lines | comment_lines
    return len(comment_lines), len(relevant_lines)


def decision_weight(node: ast.AST) -> int:
    """
    Return the cyclomatic-complexity increment for one AST node.

    Args:
        node (ast.AST): AST node.

    Returns:
        int: Complexity increment.
    """
    if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Assert)):
        return 1
    if isinstance(node, ast.BoolOp):
        return max(len(node.values) - 1, 0)
    if isinstance(node, ast.ExceptHandler):
        return 1
    if isinstance(node, ast.Try):
        return int(bool(node.orelse)) + int(bool(node.finalbody))
    if isinstance(node, ast.comprehension):
        return len(node.ifs) + int(bool(node.is_async))
    if hasattr(ast, "Match") and isinstance(node, ast.Match):
        return len(node.cases)
    if hasattr(ast, "match_case") and isinstance(node, ast.match_case):
        return int(node.guard is not None)
    return 0


def iter_local_nodes(root: ast.AST) -> Iterable[ast.AST]:
    """
    Iterate over descendant nodes while skipping nested scopes.

    Args:
        root (ast.AST): Root node for local traversal.

    Yields:
        ast.AST: Descendant node inside the current scope.
    """
    pending = list(ast.iter_child_nodes(root))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def score_block_complexity(node: ast.AST) -> int:
    """
    Compute cyclomatic complexity for one function-like block.

    Args:
        node (ast.AST): Function or async-function node.

    Returns:
        int: Cyclomatic complexity score.
    """
    return 1 + sum(decision_weight(child) for child in iter_local_nodes(node))


class ComplexityCollector(ast.NodeVisitor):
    """
    Collect block-level complexity values from one module.
    """

    def __init__(self) -> None:
        self.blocks: List[int] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.blocks.append(score_block_complexity(node))
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(stmt)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.blocks.append(score_block_complexity(node))
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(stmt)


def score_complexity(path: Path) -> Tuple[int, int]:
    """
    Count scored blocks and their total cyclomatic complexity.

    Args:
        path (Path): Python source path.

    Returns:
        Tuple[int, int]: Block count and summed complexity.
    """
    collector = ComplexityCollector()
    collector.visit(read_ast(path))
    return len(collector.blocks), sum(collector.blocks)


# ============================================================
# Dynamic Coverage
# ============================================================

def normalize_exit_code(exit_code: object) -> int:
    """
    Normalize ``SystemExit.code`` to a shell-style integer.

    Args:
        exit_code (object): ``SystemExit.code`` payload.

    Returns:
        int: Exit status compatible with shell conventions.
    """
    if exit_code in (None, 0):
        return 0
    if isinstance(exit_code, int):
        return exit_code
    return 1


def purge_local_modules(project_root: Path) -> None:
    """
    Remove cached local project modules before a traced run.

    Args:
        project_root (Path): Repository root.
    """
    to_delete: List[str] = []

    for name, module in list(sys.modules.items()):
        if name in KEEP_MODULE_NAMES or any(name.startswith(prefix) for prefix in KEEP_MODULE_PREFIXES):
            continue

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue

        try:
            resolved_path = Path(module_file).resolve()
        except OSError:
            continue

        if resolved_path == project_root or project_root in resolved_path.parents:
            to_delete.append(name)

    for name in to_delete:
        sys.modules.pop(name, None)


def build_hit_map(raw_counts: Dict[Tuple[str, int], int]) -> Dict[Path, Set[int]]:
    """
    Convert raw trace counts into a per-file executed-line map.

    Args:
        raw_counts (Dict[Tuple[str, int], int]): Raw trace counts.

    Returns:
        Dict[Path, Set[int]]: Executed line numbers grouped by resolved path.
    """
    hit_map: Dict[Path, Set[int]] = {}

    for (filename, lineno), count in raw_counts.items():
        if count <= 0:
            continue
        resolved_path = Path(filename).resolve()
        hit_map.setdefault(resolved_path, set()).add(lineno)

    return hit_map


def executable_line_set(path: Path) -> Set[int]:
    """
    Collect executable source lines using the stdlib trace parser.

    Args:
        path (Path): Python source path.

    Returns:
        Set[int]: Executable physical line numbers.
    """
    return {lineno for lineno in trace._find_executable_linenos(str(path)).keys() if lineno > 0}


def execute_traced_entry(entry_path: Path, entry_args: Sequence[str], project_root: Path) -> Tuple[int, Dict[Path, Set[int]]]:
    """
    Execute one entry script under stdlib trace and collect executed lines.

    Args:
        entry_path (Path): Entry script path.
        entry_args (Sequence[str]): Command-line arguments for the entry script.
        project_root (Path): Repository root.

    Returns:
        Tuple[int, Dict[Path, Set[int]]]: Exit code and executed lines grouped by file.
    """
    purge_local_modules(project_root)

    ignoredirs = tuple(
        path
        for path in {
            sys.prefix,
            sys.exec_prefix,
            sys.base_prefix,
            getattr(sys, "base_exec_prefix", sys.exec_prefix),
        }
        if path
    )
    tracer = trace.Trace(count=True, trace=False, ignoredirs=ignoredirs)

    original_cwd = Path.cwd()
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)

    try:
        os.chdir(project_root)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        sys.argv = [str(entry_path)] + list(entry_args)

        try:
            tracer.runfunc(runpy.run_path, str(entry_path), run_name="__main__")
            return_code = 0
        except SystemExit as error:
            return_code = normalize_exit_code(error.code)

        return return_code, build_hit_map(tracer.results().counts)
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def score_module(path: Path, hit_map: Dict[Path, Set[int]]) -> ModuleMetrics:
    """
    Compute dynamic coverage and static quality metrics for one target module.

    Args:
        path (Path): Target module path.
        hit_map (Dict[Path, Set[int]]): Executed lines grouped by file.

    Returns:
        ModuleMetrics: Per-module metric payload.
    """
    executable_lines = executable_line_set(path)
    covered_lines = executable_lines & hit_map.get(path.resolve(), set())
    comment_lines, relevant_lines = score_comment_rate(path)
    block_count, total_complexity = score_complexity(path)

    return ModuleMetrics(
        path=path.resolve(),
        executable_lines=len(executable_lines),
        covered_lines=len(covered_lines),
        comment_lines=comment_lines,
        relevant_lines=relevant_lines,
        block_count=block_count,
        total_complexity=total_complexity,
    )


def evaluate_dynamic_entry(
    entry_file: str,
    entry_args: Optional[Sequence[str]] = None,
    project_root: Path = PROJECT_ROOT,
    source_roots: Sequence[str] = DEFAULT_SOURCE_ROOTS,
) -> EntryMetrics:
    """
    Execute one workflow entry and evaluate dynamic coverage plus static quality metrics.

    Args:
        entry_file (str): Entry script file name.
        entry_args (Optional[Sequence[str]]): Command-line arguments for the entry script.
        project_root (Path): Repository root.
        source_roots (Sequence[str]): Source roots used to discover target modules.

    Returns:
        EntryMetrics: Aggregated metrics for the entry workflow.
    """
    entry_path = (project_root / entry_file).resolve()
    target_paths = discover_target_modules(project_root, source_roots)
    entry_closure = discover_entry_closure(entry_path, project_root)
    scoped_targets = sorted(path.resolve() for path in target_paths if path.resolve() in entry_closure)

    return_code, hit_map = execute_traced_entry(entry_path, [] if entry_args is None else list(entry_args), project_root)
    module_metrics = [score_module(path, hit_map) for path in scoped_targets]

    return EntryMetrics(
        entry_file=entry_file,
        return_code=return_code,
        module_count=len(module_metrics),
        executable_lines=sum(item.executable_lines for item in module_metrics),
        covered_lines=sum(item.covered_lines for item in module_metrics),
        comment_lines=sum(item.comment_lines for item in module_metrics),
        relevant_lines=sum(item.relevant_lines for item in module_metrics),
        block_count=sum(item.block_count for item in module_metrics),
        total_complexity=sum(item.total_complexity for item in module_metrics),
        module_metrics=module_metrics,
    )


def evaluate_dynamic_sequence(
    case_args: Optional[Sequence[str]] = None,
    bench_args: Optional[Sequence[str]] = None,
    project_root: Path = PROJECT_ROOT,
    source_roots: Sequence[str] = DEFAULT_SOURCE_ROOTS,
) -> List[EntryMetrics]:
    """
    Execute the case and benchmark workflows in order and collect their metrics.

    Args:
        case_args (Optional[Sequence[str]]): Command-line arguments for ``case_main.py``.
        bench_args (Optional[Sequence[str]]): Command-line arguments for ``bench_main.py``.
        project_root (Path): Repository root.
        source_roots (Sequence[str]): Source roots used to discover target modules.

    Returns:
        List[EntryMetrics]: Entry metrics in execution order.
    """
    return [
        evaluate_dynamic_entry("case_main.py", entry_args=case_args, project_root=project_root, source_roots=source_roots),
        evaluate_dynamic_entry("bench_main.py", entry_args=bench_args, project_root=project_root, source_roots=source_roots),
    ]


# ============================================================
# Reporting
# ============================================================

def entry_passed(result: EntryMetrics, thresholds: Thresholds) -> bool:
    """
    Check whether an entry passes all configured thresholds.

    Args:
        result (EntryMetrics): Entry metrics.
        thresholds (Thresholds): Quality thresholds.

    Returns:
        bool: Whether the entry passed.
    """
    return (
        result.return_code == 0
        and result.coverage_rate >= thresholds.coverage
        and result.comment_rate >= thresholds.comment_rate
        and result.average_complexity <= thresholds.average_complexity
    )


def print_entry_metrics(result: EntryMetrics, thresholds: Thresholds) -> None:
    """
    Print the three requested metrics for one workflow entry.

    Args:
        result (EntryMetrics): Entry metrics.
        thresholds (Thresholds): Quality thresholds.
    """
    coverage_ok = result.coverage_rate >= thresholds.coverage
    comment_ok = result.comment_rate >= thresholds.comment_rate
    complexity_ok = result.average_complexity <= thresholds.average_complexity
    coverage_color = hue.g if coverage_ok else hue.r
    comment_color = hue.g if comment_ok else hue.r
    complexity_color = hue.g if complexity_ok else hue.r
    overall_color = hue.g if entry_passed(result, thresholds) else hue.r
    overall_label = "PASS" if entry_passed(result, thresholds) else "FAIL"

    print("")
    print(f"{hue.b}{result.entry_file} Metrics{hue.q}")
    print(
        f"{hue.b}Dynamic coverage{hue.q}: "
        f"{coverage_color}{result.coverage_rate:.2f}%{hue.q} "
        f"{hue.m}(threshold {thresholds.coverage:.2f}%){hue.q}"
    )
    print(
        f"{hue.b}Comment rate{hue.q}   : "
        f"{comment_color}{result.comment_rate:.2f}%{hue.q} "
        f"{hue.m}(threshold {thresholds.comment_rate:.2f}%){hue.q}"
    )
    print(
        f"{hue.b}Avg complexity{hue.q} : "
        f"{complexity_color}{result.average_complexity:.2f}{hue.q} "
        f"{hue.m}(threshold {thresholds.average_complexity:.2f}){hue.q}"
    )
    print(f"{hue.b}Overall result{hue.q} : {overall_color}{overall_label}{hue.q}")


def print_entry_details(result: EntryMetrics, project_root: Path = PROJECT_ROOT) -> None:
    """
    Print per-module details for one workflow entry.

    Args:
        result (EntryMetrics): Entry metrics.
        project_root (Path): Repository root.
    """
    print("")
    print(f"{hue.b}{result.entry_file} Details{hue.q}")
    print(f"{'module':<44} {'cov%':>8} {'cmt%':>8} {'avg_cc':>8}")
    print("-" * 72)

    for item in result.module_metrics:
        module_name = item.path.relative_to(project_root).as_posix()
        print(f"{module_name:<44} {item.coverage_rate:>7.2f}% {item.comment_rate:>7.2f}% {item.average_complexity:>8.2f}")


def build_result_payload(results: Sequence[EntryMetrics], thresholds: Thresholds) -> Dict[str, object]:
    """
    Build a JSON-serializable result payload.

    Args:
        results (Sequence[EntryMetrics]): Entry metrics.
        thresholds (Thresholds): Quality thresholds.

    Returns:
        Dict[str, object]: Result payload.
    """
    entries = []

    for result in results:
        entries.append(
            {
                "entry_file": result.entry_file,
                "return_code": result.return_code,
                "module_count": result.module_count,
                "coverage_rate": round(result.coverage_rate, 2),
                "comment_rate": round(result.comment_rate, 2),
                "average_complexity": round(result.average_complexity, 2),
                "overall_pass": entry_passed(result, thresholds),
            }
        )

    return {
        "mode": "dynamic-coverage",
        "thresholds": {
            "coverage": thresholds.coverage,
            "comment_rate": thresholds.comment_rate,
            "average_complexity": thresholds.average_complexity,
        },
        "entries": entries,
    }


# ============================================================
# CLI
# ============================================================

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
            "Run case_main.py and bench_main.py sequentially, measure dynamic coverage on "
            "their shared surrogate modules, then print the three metrics for each workflow."
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
        "--json",
        action="store_true",
        help="Print a JSON payload after the workflow outputs.",
    )
    return parser.parse_args(argv)


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
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
    )

    if args.json:
        print(json.dumps(build_result_payload(results, thresholds), indent=2))
    else:
        for result in results:
            print_entry_metrics(result, thresholds)
        if args.details:
            for result in results:
                print_entry_details(result, project_root=args.project_root.resolve())

    passed = all(entry_passed(result, thresholds) for result in results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
