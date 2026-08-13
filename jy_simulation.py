# ANSYS Workbench wrapper for the jy thermal-protection case
# Author: Shengning Wang

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


MESH_SIZE_HIGH = 50.0
MESH_SIZE_LOW = 100.0
BATCH_CHUNK_SIZE = 5

INPUT_IDS = ["P1", "P2", "P3", "P9"]
OUTPUT_IDS = ["P8", "P5", "P6", "P7"]

INPUT_NAMES = ["ti65", "aerogel", "sic", "mesh_size"]
OUTPUT_NAMES = ["mass", "total_deformation", "temperature", "equivalent_stress"]


class AnsysModel:
    """Run the ANSYS Workbench jy thermal-protection case in a local copy."""

    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize the wrapper around one Workbench project copy.

        Args:
            work_dir (Optional[Path]): Directory that will hold the copied project.
        """
        root = Path(__file__).resolve().parent
        self.project_file = root / "jy.wbpj"
        self.project_dir = root / "jy_files"
        self.work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="surrogatelab_ansys_"))
        self._prepared = False

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate one design point.

        Args:
            x (np.ndarray): Design vector [ti65, aerogel, sic, mesh_size]. (4,).

        Returns:
            np.ndarray: Response vector [mass, deformation, temperature, stress]. (4,).
        """
        return self.run_batch(np.asarray(x, dtype=np.float64).reshape(1, -1))[0]

    def run_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of design points in one Workbench session.

        Args:
            x (np.ndarray): Design matrix. (N, 4).

        Returns:
            np.ndarray: Response matrix. (N, 4).
        """
        x = np.asarray(x, dtype=np.float64)
        self._prepare_project()
        results_path = self.work_dir / "results.txt"
        results_path.unlink(missing_ok=True)
        journal_path = self.work_dir / "run_batch.wbjn"
        journal_path.write_text(self._build_journal(x, results_path), encoding="utf-8")
        self._run_workbench(journal_path)
        return self._read_results(results_path, x.shape[0])

    def probe(self) -> List[List[str]]:
        """
        Open the project and dump Workbench parameter metadata.

        Returns:
            List[List[str]]: [display text, current value] rows for every parameter.
        """
        self._prepare_project()
        probe_path = self.work_dir / "probe_results.txt"
        journal_path = self.work_dir / "probe.wbjn"
        project_path = json.dumps(str((self.work_dir / self.project_file.name).resolve()))
        probe_path_str = json.dumps(str(probe_path.resolve()))
        lines = [
            "# encoding: utf-8",
            "# 2024 R1",
            'SetScriptVersion(Version="24.1.144")',
            f"Open(FilePath={project_path})",
            "import json",
            "items = []",
            "for p in Parameters.GetAllParameters():",
            "    items.append([p.DisplayText, str(p.Value)])",
            f"out = open({probe_path_str}, 'w')",
            "out.write(json.dumps(items))",
            "out.close()",
            "Save(Overwrite=True)",
        ]
        journal_path.write_text("\n".join(lines), encoding="utf-8")
        self._run_workbench(journal_path)
        with open(probe_path, "r", encoding="utf-8") as file:
            return json.loads(file.read())

    def close(self) -> None:
        """Remove the local project copy."""
        shutil.rmtree(self.work_dir)

    def _prepare_project(self) -> None:
        """Copy the Workbench project into the working directory once."""
        if self._prepared:
            return
        shutil.copy2(self.project_file, self.work_dir / self.project_file.name)
        shutil.copytree(self.project_dir, self.work_dir / self.project_dir.name)
        self._prepared = True

    def _build_journal(self, x: np.ndarray, results_path: Path) -> str:
        """
        Build the Workbench journal for a design batch.

        Args:
            x (np.ndarray): Design matrix. (N, 4).
            results_path (Path): Result text file written by the journal.

        Returns:
            str: Journal source.
        """
        project_path = json.dumps(str((self.work_dir / self.project_file.name).resolve()))
        results_path_str = json.dumps(str(results_path.resolve()))
        rows = json.dumps(x.tolist())
        lines = [
            "# encoding: utf-8",
            "# 2024 R1",
            'SetScriptVersion(Version="24.1.144")',
            f"Open(FilePath={project_path})",
            'p1 = Parameters.GetParameter(Name="P1")',
            'p2 = Parameters.GetParameter(Name="P2")',
            'p3 = Parameters.GetParameter(Name="P3")',
            'p9 = Parameters.GetParameter(Name="P9")',
            'dp0 = Parameters.GetDesignPoint(Name="0")',
            f"rows = {rows}",
            f"out = open({results_path_str}, 'a')",
            "for row in rows:",
            '    dp0.SetParameterExpression(Parameter=p1, Expression="{0} [mm]".format(row[0]))',
            '    dp0.SetParameterExpression(Parameter=p2, Expression="{0} [mm]".format(row[1]))',
            '    dp0.SetParameterExpression(Parameter=p3, Expression="{0} [mm]".format(row[2]))',
            '    dp0.SetParameterExpression(Parameter=p9, Expression="{0} [mm]".format(row[3]))',
            "    UpdateAllDesignPoints(DesignPoints=[dp0])",
            '    out.write("{0},{1},{2},{3}\\n".format(',
            '        float(Parameters.GetParameter(Name="P8").Value.Value),',
            '        float(Parameters.GetParameter(Name="P5").Value.Value),',
            '        float(Parameters.GetParameter(Name="P6").Value.Value),',
            '        float(Parameters.GetParameter(Name="P7").Value.Value)))',
            "out.close()",
            "Save(Overwrite=True)",
        ]
        return "\n".join(lines)

    def _run_workbench(self, journal_path: Path) -> None:
        """
        Execute one Workbench journal in batch mode.

        Args:
            journal_path (Path): Journal file.

        Raises:
            RuntimeError: If the Workbench batch process fails.
        """
        command = find_runwb2()
        log_path = self.work_dir / "runwb2.log"
        journal_args = ["-B", "-R", str(journal_path)]
        command_path = Path(command)
        if os.name == "nt" and command_path.suffix.lower() in (".bat", ".cmd"):
            invocation = [os.environ.get("ComSpec", "cmd.exe"), "/c", command] + journal_args
        else:
            invocation = [command] + journal_args
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                invocation,
                cwd=self.work_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:])
            raise RuntimeError(f"runwb2 returned {completed.returncode};\n{tail}")

    def _read_results(self, results_path: Path, num_rows: int) -> np.ndarray:
        """
        Read response rows written by the journal.

        Args:
            results_path (Path): Result text file.
            num_rows (int): Expected number of rows.

        Returns:
            np.ndarray: Response matrix. (N, 4).
        """
        rows: List[List[float]] = []
        with open(results_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    rows.append([float(part) for part in parts])
        data = np.asarray(rows, dtype=np.float64)
        if data.shape[0] != num_rows:
            raise RuntimeError(f"Expected {num_rows} result rows, got {data.shape[0]}.")
        return data


def _prefer_runwb2_bat(path: Path) -> Path:
    """
    Prefer the runwb2.bat wrapper that sets the ANSYS runtime environment.

    Args:
        path (Path): Located runwb2 executable or script.

    Returns:
        Path: The runwb2.bat wrapper when it exists on Windows, otherwise the input path.
    """
    if os.name != "nt" or path.name.lower().endswith((".bat", ".cmd")):
        return path
    bat_path = path.with_name("runwb2.bat")
    if bat_path.is_file():
        return bat_path
    return path


def _project_version_key() -> str:
    """
    Read the ANSYS framework version from the jy project file.

    Returns:
        str: Version key such as "241" for framework build 24.1, empty when unknown.
    """
    project_file = Path(__file__).resolve().parent / "jy.wbpj"
    match = re.search(
        r"<framework-build-version[^>]*>(\d+)\.(\d+)\.",
        project_file.read_text(encoding="utf-8-sig", errors="ignore"),
    )
    return f"{match.group(1)}{match.group(2)}" if match else ""


def find_runwb2() -> str:
    """
    Locate the ANSYS Workbench batch launcher.

    Returns:
        str: Absolute path or command name of runwb2.

    Raises:
        RuntimeError: If ANSYS Workbench cannot be found.
    """
    env_command = os.environ.get("ANSYS_RUNWB2")
    if env_command and Path(env_command).is_file():
        return str(_prefer_runwb2_bat(Path(env_command)))

    preferred = _project_version_key()

    def version_key(path: Path) -> str:
        """Extract the ANSYS version key from an install path."""
        match = re.search(r"v(\d{3})", str(path))
        return match.group(1) if match else ""

    command = shutil.which("runwb2")
    path_command = Path(command) if command is not None else None
    if path_command is not None and (not preferred or version_key(path_command) == preferred):
        return str(_prefer_runwb2_bat(path_command))

    is_windows = os.name == "nt"
    bin_dir = "Win64" if is_windows else "Linux64"
    exe_name = "runwb2.exe" if is_windows else "runwb2"

    def locate(root: Path):
        """Locate runwb2 below one ANSYS installation root."""
        candidate = root / "Framework" / "bin" / bin_dir / exe_name
        if candidate.is_file():
            return str(_prefer_runwb2_bat(candidate))
        return None

    awp_roots = [
        (env_name, Path(env_value))
        for env_name, env_value in os.environ.items()
        if env_name.startswith("AWP_ROOT") and env_value
    ]
    for env_name, root in sorted(awp_roots, key=lambda item: (version_key(item[1]) != preferred, item[0])):
        found = locate(root)
        if found:
            return found

    program_files = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
    ]
    for base in program_files:
        ansys_dir = base / "ANSYS Inc"
        if not ansys_dir.is_dir():
            continue
        version_dirs = sorted(
            ansys_dir.glob("v*"),
            key=lambda item: (version_key(item) != preferred, item.name),
        )
        for version_dir in version_dirs:
            found = locate(version_dir)
            if found:
                return found

    if path_command is not None:
        return str(_prefer_runwb2_bat(path_command))

    raise RuntimeError(
        "External ANSYS solver 'runwb2' was not found. Install ANSYS Workbench, add it to PATH, "
        "or set ANSYS_RUNWB2 to the runwb2 executable."
    )


def require_external_solver() -> Dict[str, str]:
    """
    Require the configured external ANSYS Workbench command.

    Returns:
        Dict[str, str]: External solver metadata.

    Raises:
        RuntimeError: If the Workbench command is unavailable.
    """
    command_name = find_runwb2()
    return {"solver": "ansys", "command_name": command_name}


def run_ansys_batch(x: np.ndarray, mesh_size: float, chunk_size: int = BATCH_CHUNK_SIZE) -> np.ndarray:
    """
    Evaluate a thickness DOE batch at one mesh size.

    Args:
        x (np.ndarray): Thickness design matrix. (N, 3).
        mesh_size (float): Element size in mm.
        chunk_size (int): Design points per Workbench session to bound solver memory.

    Returns:
        np.ndarray: Response matrix. (N, 4).
    """
    x = np.asarray(x, dtype=np.float64)
    x_full = np.hstack([x, np.full((x.shape[0], 1), float(mesh_size))])
    chunks = [x_full[start : start + chunk_size] for start in range(0, x_full.shape[0], chunk_size)]
    rows = []
    for chunk in chunks:
        with tempfile.TemporaryDirectory(prefix="surrogatelab_ansys_") as tmp_dir:
            model = AnsysModel(work_dir=Path(tmp_dir))
            rows.append(model.run_batch(chunk))
    return np.vstack(rows)


def main() -> None:
    """Run one ANSYS thermal-protection simulation from the command line."""
    parser = argparse.ArgumentParser(description="Run the ANSYS Workbench jy thermal-protection case.")
    parser.add_argument("--ti65", type=float, default=8.0, help="Ti65 outer layer thickness in mm.")
    parser.add_argument("--aerogel", type=float, default=8.0, help="Aerogel middle layer thickness in mm.")
    parser.add_argument("--sic", type=float, default=4.0, help="SiC inner layer thickness in mm.")
    parser.add_argument("--mesh_size", type=float, default=MESH_SIZE_HIGH, help="Element size in mm.")
    parser.add_argument("--probe", action="store_true", help="Print Workbench parameter metadata and exit.")
    args = parser.parse_args()

    require_external_solver()
    if args.probe:
        with tempfile.TemporaryDirectory(prefix="surrogatelab_ansys_") as tmp_dir:
            items = AnsysModel(work_dir=Path(tmp_dir)).probe()
        for item in items:
            print(item)
        return

    x = np.asarray([args.ti65, args.aerogel, args.sic], dtype=np.float64).reshape(1, -1)
    y = run_ansys_batch(x, args.mesh_size)[0]
    print("Input [ti65, aerogel, sic, mesh_size]:")
    print(np.concatenate([x[0], [args.mesh_size]]))
    print("Output [mass, total_deformation, temperature, equivalent_stress]:")
    print(y)


if __name__ == "__main__":
    main()
