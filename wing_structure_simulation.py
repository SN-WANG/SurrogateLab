# External Abaqus interface for the SurrogateLab wing-structure case
# Author: Shengning Wang

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class AbaqusModel:
    """Run one isolated Abaqus thermal-structural wing simulation."""

    command_name = "abq2022"
    input_vars = ["thick1", "thick2", "thick3"]
    output_vars = ["weight", "displacement", "stress_skin", "stress_stiff", "inner_temperature"]
    result_files = {
        "weight": "weight.txt",
        "displacement": "Displacement.txt",
        "stress_skin": "Mises-outterFaces.txt",
        "stress_stiff": "Mises-originStiff.txt",
        "inner_temperature": "InnerTemperature.txt",
    }

    def __init__(self, fidelity: str = "high"):
        """
        Initialize the external Abaqus model.

        Args:
            fidelity (str): ``high`` uses mesh size 30 and ``low`` uses mesh size 50.
        """
        if fidelity not in {"high", "low"}:
            raise ValueError(f"Unknown Abaqus fidelity: {fidelity!r}.")

        root = Path(__file__).resolve().parent
        self.fidelity = fidelity
        self.template_file = root / "wing_structure_template.py"
        self.model_file = root / "wing_structure_model.cae"
        self.run_file = "wing_structure_runtime.py"

    def run(self, input_arr) -> np.ndarray:
        """
        Run Abaqus for one TPS thickness design.

        Args:
            input_arr: SiC, Aerogel, and Ti65 layer thicknesses. (3,).

        Returns:
            np.ndarray: Weight, displacement, skin stress, stiffener stress, and inner temperature. (5,).

        Raises:
            ValueError: If the input does not contain exactly three values.
            RuntimeError: If Abaqus cannot run or any scalar result is invalid.
        """
        x = np.asarray(input_arr, dtype=float).reshape(-1)
        if x.size != len(self.input_vars):
            raise ValueError(f"Expected {len(self.input_vars)} design variables, received {x.size}.")

        params = dict(zip(self.input_vars, x))
        params["meshSize"] = 30.0 if self.fidelity == "high" else 50.0

        with tempfile.TemporaryDirectory(prefix="surrogatelab_abaqus_") as temp_dir:
            work_dir = Path(temp_dir)
            run_file = self._prepare_workspace(work_dir, params)
            log_output = self._run_abaqus(work_dir, run_file)
            return self._read_results(work_dir, log_output)

    def _prepare_workspace(self, work_dir: Path, params: dict) -> Path:
        if not self.template_file.is_file():
            raise RuntimeError(f"Abaqus template not found: {self.template_file}")
        if not self.model_file.is_file():
            raise RuntimeError(f"Abaqus CAE model not found: {self.model_file}")

        shutil.copy2(self.model_file, work_dir / self.model_file.name)
        content = self.template_file.read_text(encoding="utf-8")
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

        for name, value in params.items():
            pattern = rf"(?m)^({re.escape(name)}\s*=\s*){number}[ \t]*$"
            content, count = re.subn(pattern, lambda match: f"{match.group(1)}{value:.4f}", content, count=1)
            if count != 1:
                raise RuntimeError(f"Abaqus template parameter not found: {name}")

        run_file = work_dir / self.run_file
        run_file.write_text(content, encoding="utf-8")
        return run_file

    def _run_abaqus(self, work_dir: Path, run_file: Path) -> str:
        command = [self.command_name, "cae", f"noGUI={run_file}"]
        try:
            completed = subprocess.run(
                command,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(f"Unable to start external Abaqus solver {self.command_name!r}: {exc}") from exc

        if completed.returncode != 0:
            message = f"Abaqus exited with status {completed.returncode}."
            raise RuntimeError(self._with_log_tail(message, completed.stdout))
        return completed.stdout

    def _read_results(self, work_dir: Path, log_output: str) -> np.ndarray:
        results = []
        for name in self.output_vars:
            result_file = work_dir / self.result_files[name]
            if not result_file.is_file():
                raise RuntimeError(self._with_log_tail(f"Abaqus result is missing: {result_file.name}.", log_output))

            try:
                value = float(result_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError) as exc:
                message = f"Abaqus result is unreadable: {result_file.name}."
                raise RuntimeError(self._with_log_tail(message, log_output)) from exc
            if not np.isfinite(value):
                message = f"Abaqus result is not finite: {result_file.name}={value}."
                raise RuntimeError(self._with_log_tail(message, log_output))
            results.append(value)

        return np.asarray(results, dtype=float)

    @staticmethod
    def _with_log_tail(message: str, log_output: str, max_lines: int = 80) -> str:
        lines = log_output.splitlines()
        if not lines:
            return message
        tail = "\n".join(lines[-max_lines:])
        return f"{message}\nAbaqus log tail:\n{tail}"
