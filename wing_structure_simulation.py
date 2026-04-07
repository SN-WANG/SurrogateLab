import subprocess
import re
import os
import numpy as np


class AbaqusModel:
    def __init__(self, fidelity='high'):
        """
        :param fidelity: 'high' (默认，保持模板 meshSize=30) 或 'low' (强制修改 meshSize=50)
        """
        # 保存保真度设置
        self.fidelity = fidelity

        # ================= 配置区域 =================
        self.template_file = 'wing_structure_template.py'
        self.run_file = 'wing_structure_runtime.py'
        self.abaqus_cmd = "abq2022 cae noGUI={}"

        self.input_vars = ['thick1', 'thick2', 'thick3']

        self.result_files = {
            'weight': 'weight.txt',
            'displacement': 'Displacement.txt',
            'stress_skin': 'Mises-outterFaces.txt',
            'stress_stiff': 'Mises-originStiff.txt'
        }

        self.output_vars = ['weight', 'displacement', 'stress_skin', 'stress_stiff']
        # ===========================================

    def run(self, input_arr):
        if isinstance(input_arr, np.ndarray):
            x = input_arr.flatten()
        else:
            x = np.array(input_arr)

        if len(x) != 3:
            print(f"[Error] 输入维度错误，需要 3 个变量，实际输入: {len(x)}")
            return np.full(4, np.nan)

        # 1. 基础设计变量
        params = dict(zip(self.input_vars, x))

        # 2. 根据保真度设置处理 meshSize
        # 如果是 low，我们将 meshSize 加入待修改的参数列表，值为 50
        # 如果是 high，我们不把 meshSize 加入 params，正则替换时就会跳过它，保留模板原值(30)
        if self.fidelity == 'low':
            params['meshSize'] = 50

        if not self._update_script(params):
            return np.full(4, np.nan)

        self._run_abaqus()

        return self._read_results()

    def _update_script(self, params):
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for key, value in params.items():
                # 正则替换：匹配 key = 数字，替换为 key = 新数值
                pattern = rf'({key}\s*=\s*)[\d\.]+'
                replacement = f'{key}={value:.4f}'
                content = re.sub(pattern, replacement, content)

            with open(self.run_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[Error] 更新脚本失败: {e}")
            return False

    def _run_abaqus(self):
        current_dir = os.path.abspath(os.getcwd())
        script_path = os.path.join(current_dir, self.run_file)
        cmd = self.abaqus_cmd.format(script_path)

        try:
            subprocess.run(cmd, shell=True, cwd=current_dir, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("  [Warning] Abaqus 运行返回错误代码 (可能是许可问题或模型不收敛)")

    def _read_results(self):
        results = []
        for var_name in self.output_vars:
            file_name = self.result_files[var_name]
            val = np.nan

            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as f:
                        val = float(f.read().strip())
                except:
                    pass

            results.append(val)

        return np.array(results)