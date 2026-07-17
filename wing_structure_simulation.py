import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class AbaqusModel:
    def __init__(self, fidelity='high'):
        """Abaqus wrapper for one TPS wing simulation.

        Parameters
        ----------
        fidelity : {'high', 'low'}
            high keeps the template meshSize=30. low changes meshSize to 50.
        """
        self.fidelity = fidelity

        # Files and command. The script path is quoted so workspaces with spaces
        # in their path still work.
        root = Path(__file__).resolve().parent
        self.template_file = root / 'wing_structure_template.py'
        self.model_file = root / 'wing_structure_model.cae'
        self.run_file = 'wing_structure_runtime.py'
        self.abaqus_cmd = 'abq2022 cae noGUI="{}"'
        self.log_file = 'abaqus_run.log'

        # Design variables: TPS layer thicknesses, inside to outside.
        self.input_vars = ['sic_thick', 'aerogel_thick', 'ti65_thick']

        # Scalar result files written by the Abaqus script.
        self.result_files = {
            'weight': 'weight.txt',
            'displacement': 'Displacement.txt',
            'stress_skin': 'Mises-outterFaces.txt',
            'stress_stiff': 'Mises-originStiff.txt',
            'inner_temperature': 'InnerTemperature.txt',
        }
        self.output_vars = [
            'weight',
            'displacement',
            'stress_skin',
            'stress_stiff',
            'inner_temperature',
        ]

    def run(self, input_arr):
        """Run Abaqus and return results in output_vars order."""
        with tempfile.TemporaryDirectory(prefix='surrogatelab_abaqus_') as temp_dir:
            work_dir = Path(temp_dir)
            shutil.copy2(self.template_file, work_dir / self.template_file.name)
            shutil.copy2(self.model_file, work_dir / self.model_file.name)

            template_file = self.template_file
            current_dir = os.getcwd()
            self.template_file = self.template_file.name
            os.chdir(work_dir)

            try:
                x = np.asarray(input_arr, dtype=float).flatten()
                if len(x) != len(self.input_vars):
                    print(f"[Error] Expected {len(self.input_vars)} variables, got {len(x)}.")
                    return np.full(len(self.output_vars), np.nan)

                params = dict(zip(self.input_vars, x))
                if self.fidelity == 'low':
                    params['meshSize'] = 50.0

                if not self._update_script(params):
                    return np.full(len(self.output_vars), np.nan)

                self._cleanup_result_files()
                if not self._run_abaqus():
                    return np.full(len(self.output_vars), np.nan)

                return self._read_results()
            finally:
                os.chdir(current_dir)
                self.template_file = template_file

    def _update_script(self, params):
        """Create the Abaqus run script from the template."""
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for key, value in params.items():
                pattern = rf'({key}\s*=\s*)[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
                replacement = rf'\g<1>{value:.4f}'
                content, count = re.subn(pattern, replacement, content, count=1)
                if count == 0:
                    print(f"[Warning] Parameter not found in template: {key}")

            with open(self.run_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[Error] Failed to update Abaqus script: {e}")
            return False

    def _cleanup_result_files(self):
        """Remove old scalar result files before starting a new run."""
        for file_name in set(self.result_files.values()):
            if os.path.exists(file_name):
                try:
                    os.remove(file_name)
                except OSError:
                    pass

    def _run_abaqus(self):
        """Run Abaqus and write all console output to abaqus_run.log."""
        current_dir = os.path.abspath(os.getcwd())
        script_path = os.path.join(current_dir, self.run_file)
        cmd = self.abaqus_cmd.format(script_path)

        print(f"[Info] Running Abaqus command: {cmd}")
        print(f"[Info] Abaqus log: {os.path.join(current_dir, self.log_file)}")

        try:
            with open(self.log_file, 'w', encoding='utf-8', errors='replace') as log:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=current_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except Exception as e:
            print(f"[Error] Failed to start Abaqus: {e}")
            return False

        if completed.returncode != 0:
            print(f"[Error] Abaqus returned non-zero code: {completed.returncode}")
            self._print_log_tail()
            return False

        return True

    def _print_log_tail(self, max_lines=80):
        """Print the tail of the Abaqus log for diagnosis."""
        if not os.path.exists(self.log_file):
            print(f"[Error] Log file not found: {self.log_file}")
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            print(f"[Info] Last {min(max_lines, len(lines))} lines of {self.log_file}:")
            for line in lines[-max_lines:]:
                print(line.rstrip())
        except Exception as e:
            print(f"[Error] Failed to read log file: {e}")

    def _read_results(self):
        """Read scalar result files in output_vars order."""
        results = []
        missing = []

        for var_name in self.output_vars:
            file_name = self.result_files[var_name]
            val = np.nan

            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as f:
                        val = float(f.read().strip())
                except Exception:
                    missing.append(file_name)
            else:
                missing.append(file_name)

            results.append(val)

        if missing:
            print(f"[Warning] Missing or unreadable result files: {missing}")
            self._print_log_tail(max_lines=40)

        return np.array(results)


def main():
    """Run one thermal-protection simulation from the command line."""
    parser = argparse.ArgumentParser(
        description='Run the Abaqus TPS wing model for one thickness design.')
    parser.add_argument('--fidelity', choices=['high', 'low'], default='high',
                        help='high keeps meshSize=30; low changes meshSize=50.')
    parser.add_argument('--sic', type=float, default=4.0,
                        help='SiC inner layer thickness.')
    parser.add_argument('--aerogel', type=float, default=8.0,
                        help='Aerogel middle layer thickness.')
    parser.add_argument('--ti65', type=float, default=8.0,
                        help='Ti65 outer layer thickness.')
    args = parser.parse_args()

    model = AbaqusModel(fidelity=args.fidelity)
    x = np.array([args.sic, args.aerogel, args.ti65], dtype=float)
    y = model.run(x)

    print('Input [sic_thick, aerogel_thick, ti65_thick]:')
    print(x)
    print('Output [weight, displacement, stress_skin, stress_stiff, inner_temperature]:')
    print(y)


if __name__ == '__main__':
    main()
