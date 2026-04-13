# Kriging (KRG, Gaussian Process Regression) Surrogate Model
# Author: Shengning Wang

from typing import Optional

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.optimize import Bounds, minimize

from utils.scaler import StandardScalerNP


def _constant_regression(x: np.ndarray) -> np.ndarray:
    return np.ones((x.shape[0], 1), dtype=x.dtype)


def _gaussian_correlation(theta: np.ndarray, d: np.ndarray) -> np.ndarray:
    d = np.atleast_2d(d)
    input_dim = d.shape[1]

    theta_vec = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_vec.size == 1:
        theta_vec = np.full(input_dim, float(theta_vec[0]), dtype=np.float64)
    elif theta_vec.size != input_dim:
        raise ValueError(f"Theta must be scalar or length {input_dim}.")

    return np.exp(-np.sum(theta_vec[np.newaxis, :] * d ** 2, axis=1, keepdims=True))


class KRG:
    """
    Kriging surrogate kept for the current SurrogateLab workflow.
    """

    def __init__(
        self,
        poly: str = "constant",
        kernel: str = "gaussian",
        theta0: float | np.ndarray = 1.0,
        theta_bounds: tuple[float, float] = (1e-6, 100.0),
    ) -> None:
        """
        Initialize the KRG model.

        Args:
            poly (str): Regression basis. Only ``"constant"`` is retained.
            kernel (str): Correlation kernel. Only ``"gaussian"`` is retained.
            theta0 (float | np.ndarray): Initial theta value.
            theta_bounds (tuple[float, float]): Optimization bounds for theta.
        """
        if poly != "constant":
            raise ValueError("Only the constant regression basis is retained in SurrogateLab KRG.")
        if kernel != "gaussian":
            raise ValueError("Only the gaussian kernel is retained in SurrogateLab KRG.")

        self.theta0 = theta0
        self.theta_bounds = theta_bounds

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.x_train_scaled: Optional[np.ndarray] = None
        self.theta: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.gamma: Optional[np.ndarray] = None
        self.sigma2: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.Ft: Optional[np.ndarray] = None

    def _fit_gls(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, d: np.ndarray):
        num_samples = x.shape[0]

        r_vec = _gaussian_correlation(theta, d)
        r_mat = np.eye(num_samples, dtype=np.float64)
        idx_upper = np.triu_indices(num_samples, k=1)
        r_mat[idx_upper] = r_vec[:, 0]
        r_mat.T[idx_upper] = r_vec[:, 0]
        r_mat += np.eye(num_samples, dtype=np.float64) * (10 + num_samples) * np.finfo(float).eps

        try:
            c_mat = cholesky(r_mat, lower=True)
            y_tilde = solve_triangular(c_mat, y, lower=True)
            f_mat = _constant_regression(x)
            f_tilde = solve_triangular(c_mat, f_mat, lower=True)
            q_mat, g_mat = qr(f_tilde, mode="economic")
            beta = solve_triangular(g_mat, q_mat.T @ y_tilde, lower=False)
        except np.linalg.LinAlgError:
            return None

        rho = y_tilde - f_tilde @ beta
        sigma2 = np.sum(rho ** 2, axis=0) / num_samples
        gamma = solve_triangular(c_mat.T, rho, lower=False)
        return c_mat, g_mat, f_tilde, beta, gamma, sigma2

    def _objective_function(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
        fit_result = self._fit_gls(x, y, theta, d)
        if fit_result is None:
            return 1.0e20

        c_mat, _, _, _, _, sigma2 = fit_result
        if np.any(sigma2 <= 0.0):
            return 1.0e20

        num_samples = x.shape[0]
        target_dim = y.shape[1]
        log_det_r = 2.0 * np.sum(np.log(np.diag(c_mat)))
        return float(num_samples * np.sum(np.log(sigma2)) + target_dim * log_det_r)

    def _build_initial_theta(self, input_dim: int) -> np.ndarray:
        theta0 = np.asarray(self.theta0, dtype=np.float64).reshape(-1)
        if theta0.size == 1:
            return np.full(input_dim, float(theta0[0]), dtype=np.float64)
        if theta0.size != input_dim:
            raise ValueError(f"Theta0 must be scalar or length {input_dim}.")
        return theta0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the KRG surrogate.

        Args:
            x_train (np.ndarray): Training inputs. (N, D).
            y_train (np.ndarray): Training targets. (N, C).
        """
        num_samples, input_dim = x_train.shape

        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)
        self.x_train_scaled = x_scaled

        idx_i, idx_j = np.triu_indices(num_samples, k=1)
        d = x_scaled[idx_i] - x_scaled[idx_j]

        theta_initial = self._build_initial_theta(input_dim)
        bounds = None
        if self.theta_bounds is not None:
            bounds = Bounds(
                np.full(input_dim, self.theta_bounds[0], dtype=np.float64),
                np.full(input_dim, self.theta_bounds[1], dtype=np.float64),
            )

        result = minimize(
            fun=self._objective_function,
            x0=theta_initial,
            args=(x_scaled, y_scaled, d),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

        self.theta = np.asarray(result.x, dtype=np.float64)
        fit_result = self._fit_gls(x_scaled, y_scaled, self.theta, d)
        if fit_result is None:
            raise ValueError("Fit failed (Cholesky decomposition error).")

        self.C, self.G, self.Ft, self.beta, self.gamma, self.sigma2 = fit_result

    def predict(self, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at new inputs.

        Args:
            x_pred (np.ndarray): Prediction inputs. (N, D).

        Returns:
            tuple[np.ndarray, np.ndarray]: Mean and variance. Both (N, C).
        """
        if self.beta is None or self.gamma is None or self.sigma2 is None:
            raise RuntimeError("Model has not been fitted.")

        num_train = self.x_train_scaled.shape[0]
        num_pred, input_dim = x_pred.shape
        x_scaled = self.scaler_x.transform(x_pred)

        d_cross = self.x_train_scaled[:, np.newaxis, :] - x_scaled[np.newaxis, :, :]
        r_cross = _gaussian_correlation(self.theta, d_cross.reshape(-1, input_dim)).reshape(num_train, num_pred)
        f_test = _constant_regression(x_scaled)

        y_pred_scaled = f_test @ self.beta + r_cross.T @ self.gamma

        rt = solve_triangular(self.C, r_cross, lower=True)
        u = self.Ft.T @ rt - f_test.T
        v = solve_triangular(self.G.T, u, lower=True)
        var_factor = np.maximum(1.0 + np.sum(v ** 2, axis=0) - np.sum(rt ** 2, axis=0), 0.0)
        var_pred_scaled = np.outer(var_factor, self.sigma2)

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        var_pred = var_pred_scaled * (self.scaler_y.std ** 2)
        return y_pred, var_pred
