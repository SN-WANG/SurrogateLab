# MFS-MLS: Multi-Fidelity Surrogate Model based on Moving Least Squares
# Paper reference: https://doi.org/10.1007/s00158-021-03044-5
# Paper author: Shuo Wang, Yin Liu, Qi Zhou, Yongliang Yuan, Liye Lv, Xueguan Song
# Code author: Shengning Wang

from itertools import combinations_with_replacement
from typing import Dict, Optional

import numpy as np

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class MFSMLS:
    """
    Multi-fidelity surrogate model based on moving least squares.
    """

    def __init__(
        self,
        lf_model_params: Optional[Dict] = None,
        poly_degree: int = 2,
        neighbor_factor: float = 1.0,
        ridge: float = 1.0e-8,
    ) -> None:
        """
        Initialize the MFS-MLS surrogate.

        Args:
            lf_model_params (Optional[Dict]): Parameters for the LF RBF model.
            poly_degree (int): Polynomial basis degree.
            neighbor_factor (float): Expansion factor for required HF neighbors.
            ridge (float): Ridge factor for local weighted least squares.
        """
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)
        self.poly_degree = poly_degree
        self.neighbor_factor = neighbor_factor
        self.ridge = ridge

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.p_train_: Optional[np.ndarray] = None
        self.required_hf_samples_: int = 0
        self.is_fitted = False

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.

        Args:
            x (np.ndarray): Query points. (N, D).
            c (np.ndarray): Reference points. (M, D).

        Returns:
            np.ndarray: Distance matrix. (N, M).
        """
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        dists_sq = x_norm_sq + c_norm_sq - 2.0 * (x @ c.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return np.sqrt(dists_sq)

    def _generate_polynomial_powers(self, input_dim: int) -> np.ndarray:
        """
        Generate monomial exponent vectors.

        Args:
            input_dim (int): Input dimension.

        Returns:
            np.ndarray: Exponent matrix. (P, D).
        """
        powers = []
        for degree in range(self.poly_degree + 1):
            for combo in combinations_with_replacement(range(input_dim), degree):
                power = np.zeros(input_dim, dtype=np.int64)
                for idx in combo:
                    power[idx] += 1
                powers.append(power)
        return np.stack(powers, axis=0)

    def _build_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis features.

        Args:
            x (np.ndarray): Input points. (N, D).

        Returns:
            np.ndarray: Polynomial features. (N, P).
        """
        powers = self._generate_polynomial_powers(x.shape[1])
        phi = np.ones((x.shape[0], powers.shape[0]), dtype=x.dtype)
        for dim in range(x.shape[1]):
            exp_dim = powers[:, dim]
            mask = exp_dim > 0
            if np.any(mask):
                phi[:, mask] *= np.power(x[:, dim:dim + 1], exp_dim[mask])
        return phi

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Fit the MFS-MLS surrogate.

        Args:
            x_lf (np.ndarray): LF inputs. (N_L, D).
            y_lf (np.ndarray): LF targets. (N_L, C).
            x_hf (np.ndarray): HF inputs. (N_H, D).
            y_hf (np.ndarray): HF targets. (N_H, C).
        """
        self.lf_model.fit(x_lf, y_lf)

        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):
            y_lf_at_hf = y_lf_at_hf[0]

        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf)
        poly_basis = self._build_polynomial_features(self.x_hf_train_)
        self.p_train_ = np.concatenate([y_lf_at_hf_scaled, poly_basis], axis=1)

        min_required = self.p_train_.shape[1]
        expanded_required = int(np.ceil(self.neighbor_factor * min_required))
        self.required_hf_samples_ = min(max(min_required, expanded_required), self.x_hf_train_.shape[0])
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
        y_lf_at_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_at_pred, tuple):
            y_lf_at_pred = y_lf_at_pred[0]

        y_lf_at_pred_scaled = self.scaler_y.transform(y_lf_at_pred)
        poly_basis_pred = self._build_polynomial_features(x_pred_scaled)
        p_pred = np.concatenate([y_lf_at_pred_scaled, poly_basis_pred], axis=1)

        dists = self._compute_dists(x_pred_scaled, self.x_hf_train_)
        num_samples = x_pred.shape[0]
        num_basis = self.p_train_.shape[1]
        target_dim = self.y_hf_train_.shape[1]
        y_pred_scaled = np.zeros((num_samples, target_dim), dtype=np.float64)

        for i in range(num_samples):
            sorted_dists = np.sort(dists[i])
            influence_radius = max(float(sorted_dists[self.required_hf_samples_ - 1]), 1.0e-12)
            di = dists[i] / influence_radius

            wi = np.zeros(self.x_hf_train_.shape[0], dtype=np.float64)
            mask = di <= 1.0
            wi[mask] = np.exp(-4.0 * di[mask] ** 2)

            W = np.diag(wi)
            lhs = self.p_train_.T @ W @ self.p_train_
            rhs = self.p_train_.T @ W @ self.y_hf_train_

            try:
                coeffs = np.linalg.solve(lhs + self.ridge * np.eye(num_basis), rhs)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(lhs) @ rhs

            y_pred_scaled[i] = p_pred[i] @ coeffs

        return self.scaler_y.inverse_transform(y_pred_scaled)
