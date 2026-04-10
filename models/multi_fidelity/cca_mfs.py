# CCA-MFS: Multi-Fidelity Surrogate Model Based on Canonical Correlation Analysis and Least Squares
# Paper reference: https://doi.org/10.1115/1.4047686
# Paper author: Liye Lv, Chaoyang Zong, Chao Zhang, Xueguan Song, Wei Sun
# Code author: Shengning Wang

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import inv, sqrtm

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class CCAMFS:
    """
    Multi-fidelity surrogate model based on canonical correlation analysis.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None, residual_ridge: float = 1.0) -> None:
        """
        Initialize the CCA-MFS surrogate.

        Args:
            lf_model_params (Optional[Dict]): Parameters for internal RBF models.
            residual_ridge (float): Ridge factor for residual correction.
        """
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)
        self.hf_rbf_model_ = RBF(**params)
        self.lf_rbf_model_ = RBF(**params)
        self.residual_ridge = residual_ridge

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.U_: Optional[np.ndarray] = None
        self.V_: Optional[np.ndarray] = None
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.y_lf_at_hf_: Optional[np.ndarray] = None
        self.Ph_transformed_: Optional[np.ndarray] = None
        self.Pl_transformed_: Optional[np.ndarray] = None
        self.Rh_: Optional[np.ndarray] = None
        self.Rhl_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.W1_: Optional[np.ndarray] = None
        self.W2_: Optional[np.ndarray] = None
        self.is_fitted = False

    def _compute_covariance_matrices(self, Ph: np.ndarray, Pl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute paired covariance matrices for CCA.

        Args:
            Ph (np.ndarray): HF paired samples. (N, D).
            Pl (np.ndarray): LF paired samples. (N, D).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: S11, S22, S12.
        """
        ph_centered = Ph - np.mean(Ph, axis=0, keepdims=True)
        pl_centered = Pl - np.mean(Pl, axis=0, keepdims=True)
        scale = Ph.shape[0] - 1
        S11 = (ph_centered.T @ ph_centered) / scale
        S22 = (pl_centered.T @ pl_centered) / scale
        S12 = (ph_centered.T @ pl_centered) / scale
        return S11, S22, S12

    def _compute_cca_transition_matrices(
        self,
        S11: np.ndarray,
        S22: np.ndarray,
        S12: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CCA transition matrices.

        Args:
            S11 (np.ndarray): HF covariance. (D, D).
            S22 (np.ndarray): LF covariance. (D, D).
            S12 (np.ndarray): Cross covariance. (D, D).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transition matrices U and V.
        """
        S11_inv_sqrt = inv(sqrtm(S11).astype(np.float64))
        S22_inv_sqrt = inv(sqrtm(S22).astype(np.float64))
        C = S11_inv_sqrt @ S12 @ S22_inv_sqrt
        L, _, R_t = np.linalg.svd(C, full_matrices=True)
        U = S11_inv_sqrt @ L
        V = S22_inv_sqrt @ R_t.T
        return U, V

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.

        Args:
            x (np.ndarray): Query points. (N, D).
            c (np.ndarray): Reference points. (M, D).

        Returns:
            np.ndarray: Distances. (N, M).
        """
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        dists_sq = x_norm_sq + c_norm_sq - 2.0 * (x @ c.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return np.sqrt(dists_sq)

    def _build_rbf_correlation_from_model(self, x_query: np.ndarray, model: RBF) -> np.ndarray:
        """
        Build an RBF correlation matrix from a trained RBF model.

        Args:
            x_query (np.ndarray): Query points. (N, D).
            model (RBF): Trained RBF model.

        Returns:
            np.ndarray: Correlation matrix. (N, M).
        """
        dists = self._compute_dists(x_query, model.centers)
        return np.exp(-model.gamma * (dists ** 2))

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Fit the CCA-MFS surrogate.

        Args:
            x_lf (np.ndarray): LF inputs. (N_L, D).
            y_lf (np.ndarray): LF targets. (N_L, C).
            x_hf (np.ndarray): HF inputs. (N_H, D).
            y_hf (np.ndarray): HF targets. (N_H, C).
        """
        num_hf, _ = x_hf.shape
        num_lf = x_lf.shape[0]
        target_dim = y_hf.shape[1]

        self.lf_model.fit(x_lf, y_lf)

        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):
            y_lf_at_hf = y_lf_at_hf[0]
        self.y_lf_at_hf_ = self.scaler_y.transform(y_lf_at_hf)

        x_lf_scaled = self.scaler_x.transform(x_lf)
        y_lf_scaled = self.scaler_y.transform(y_lf)

        Ph = np.concatenate([self.x_hf_train_, self.y_hf_train_], axis=1)
        Pl = np.concatenate([x_lf_scaled, y_lf_scaled], axis=1)
        Pl_cca = np.concatenate([self.x_hf_train_, self.y_lf_at_hf_], axis=1)

        S11, S22, S12 = self._compute_covariance_matrices(Ph, Pl_cca)
        self.U_, self.V_ = self._compute_cca_transition_matrices(S11, S22, S12)

        self.Ph_transformed_ = Ph @ self.U_
        self.Pl_transformed_ = Pl @ self.V_

        self.hf_rbf_model_.fit(self.Ph_transformed_, self.y_hf_train_)
        self.lf_rbf_model_.fit(self.Pl_transformed_, y_lf_scaled)

        self.Rh_ = self._build_rbf_correlation_from_model(self.Ph_transformed_, self.hf_rbf_model_)
        self.Rhl_ = self._build_rbf_correlation_from_model(self.Ph_transformed_, self.lf_rbf_model_)

        correction_matrix = np.concatenate([self.Rh_, self.Rhl_], axis=1)
        gram = correction_matrix.T @ correction_matrix
        if self.residual_ridge > 0.0:
            gram = gram + self.residual_ridge * np.eye(gram.shape[0], dtype=np.float64)

        self.bias_ = np.zeros(target_dim, dtype=np.float64)
        self.rho_ = np.zeros(target_dim, dtype=np.float64)
        self.W1_ = np.zeros((num_hf, target_dim), dtype=np.float64)
        self.W2_ = np.zeros((num_lf, target_dim), dtype=np.float64)

        ones = np.ones((num_hf, 1), dtype=np.float64)
        for m in range(target_dim):
            affine_matrix = np.concatenate([ones, self.y_lf_at_hf_[:, m:m + 1]], axis=1)
            affine_target = self.y_hf_train_[:, m:m + 1]

            try:
                affine_theta, _, _, _ = np.linalg.lstsq(affine_matrix, affine_target, rcond=None)
            except np.linalg.LinAlgError:
                affine_theta = np.linalg.pinv(affine_matrix) @ affine_target

            self.bias_[m] = float(affine_theta[0, 0])
            self.rho_[m] = float(affine_theta[1, 0])

            residual_target = affine_target - (self.bias_[m] + self.rho_[m] * self.y_lf_at_hf_[:, m:m + 1])
            rhs = correction_matrix.T @ residual_target
            try:
                correction_theta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                correction_theta = np.linalg.pinv(gram) @ rhs

            self.W1_[:, m] = correction_theta[:num_hf, 0]
            self.W2_[:, m] = correction_theta[num_hf:, 0]

        self.is_fitted = True

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new inputs.

        Args:
            x_pred (np.ndarray): Prediction inputs. (N, D).

        Returns:
            np.ndarray: Predicted targets. (N, C).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        x_pred_scaled = self.scaler_x.transform(x_pred)
        y_lf_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_pred, tuple):
            y_lf_pred = y_lf_pred[0]
        y_lf_pred_scaled = self.scaler_y.transform(y_lf_pred)

        P_test = np.concatenate([x_pred_scaled, y_lf_pred_scaled], axis=1)
        P_test_U = P_test @ self.U_
        P_test_V = P_test @ self.V_

        Rh_ts = self._build_rbf_correlation_from_model(P_test_U, self.hf_rbf_model_)
        Rl_ts = self._build_rbf_correlation_from_model(P_test_V, self.lf_rbf_model_)

        y_pred_scaled = self.bias_[None, :] + self.rho_[None, :] * y_lf_pred_scaled + Rh_ts @ self.W1_ + Rl_ts @ self.W2_
        return self.scaler_y.inverse_transform(y_pred_scaled)
