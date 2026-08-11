# ANSYS Workbench wrapper for the jy thermal-protection case
# Author: Shengning Wang

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


MESH_SIZE_HIGH = 50.0
MESH_SIZE_LOW = 100.0

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
            f"OpenProject(FilePath={project_path})",
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
            f"OpenProject(FilePath={project_path})",
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
            '        float(Parameters.GetParameter(Name="P8").Value),',
            '        float(Parameters.GetParameter(Name="P5").Value),',
            '        float(Parameters.GetParameter(Name="P6").Value),',
            '        float(Parameters.GetParameter(Name="P7").Value)))',
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
        command = f'runwb2 -B "{journal_path}"'
        log_path = self.work_dir / "runwb2.log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(command, shell=True, cwd=self.work_dir, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise RuntimeError(f"runwb2 returned {completed.returncode}; see {log_path}")

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


def require_external_solver() -> Dict[str, str]:
    """
    Require the configured external ANSYS Workbench command.

    Returns:
        Dict[str, str]: External solver metadata.

    Raises:
        RuntimeError: If the Workbench command is unavailable.
    """
    command_name = "runwb2"
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"External ANSYS solver '{command_name}' was not found on PATH. "
            "Install or expose ANSYS Workbench before running the engineering workflow."
        )
    return {"solver": "ansys", "command_name": command_name}


def run_ansys_batch(x: np.ndarray, mesh_size: float) -> np.ndarray:
    """
    Evaluate a thickness DOE batch at one mesh size.

    Args:
        x (np.ndarray): Thickness design matrix. (N, 3).
        mesh_size (float): Element size in mm.

    Returns:
        np.ndarray: Response matrix. (N, 4).
    """
    x = np.asarray(x, dtype=np.float64)
    x_full = np.hstack([x, np.full((x.shape[0], 1), float(mesh_size))])
    with tempfile.TemporaryDirectory(prefix="surrogatelab_ansys_") as tmp_dir:
        model = AnsysModel(work_dir=Path(tmp_dir))
        return model.run_batch(x_full)


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
