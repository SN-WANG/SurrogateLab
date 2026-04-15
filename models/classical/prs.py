# Polynomial Regression Surface (PRS) Surrogate Model
# Author: Shengning Wang

import numpy as np
from typing import Optional
from itertools import combinations_with_replacement

from utils.scaler import StandardScalerNP


class PRS:
    """
    Polynomial Regression Surface (PRS) using regularized least squares on polynomial features.
    """

    def __init__(self, degree: int = 3, alpha: float = 0.0):
        """
        Args:
            degree (int): Polynomial degree.
            alpha (float): Ridge regularization strength.
        """
        # parameters
        self.degree = degree
        self.alpha = alpha

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.powers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    # ============================================================
    # Polynomial Index Enumeration
    # ============================================================
    # ------------------------------------------------------------------

    def _generate_powers(self, input_dim: int) -> np.ndarray:
        """
        Generates exponent combinations for full polynomial basis.

        Args:
            input_dim (int): Dimension of input.

        Returns:
            np.ndarray: Exponent matrix. Shape: (num_terms, input_dim), dtype: int64.
        """
        powers = []
        for d in range(self.degree + 1):
            # Enumerate all monomials x_1^{a_1} ... x_D^{a_D} with total degree sum(a_k) = d.
            for comb in combinations_with_replacement(range(input_dim), d):
                # Each combination maps to one exponent vector
                # alpha = [a_1, ..., a_D] with sum(alpha) = d.
                power = np.zeros(input_dim, dtype=np.int64)
                for idx in comb:
                    power[idx] += 1
                powers.append(power)

        # Stack all exponent vectors into
        # A = [alpha_1; ...; alpha_P].
        return np.stack(powers, axis=0)

    # ============================================================
    # Design Matrix Construction
    # ============================================================
    # ------------------------------------------------------------------

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """
        Constructs polynomial feature matrix.

        Args:
            x (np.ndarray): Inputs. Shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Feature matrix. Shape: (num_samples, num_terms), dtype: float64.
        """
        num_samples, input_dim = x.shape
        num_terms = self.powers.shape[0]

        # Polynomial design matrix:
        # Phi_{i, j} = prod_k x_{ik}^{power_{j, k}}.
        phi = np.ones((num_samples, num_terms), dtype=x.dtype)
        # The constant column corresponds to alpha = 0 and remains equal to 1.

        for d in range(input_dim):
            exp_d = self.powers[:, d]
            mask = exp_d > 0
            if not np.any(mask):
                continue
            # Multiply the monomials by x_d^{a_d} only where the exponent is nonzero.
            phi[:, mask] *= (x[:, d:d+1] ** exp_d[mask])

        return phi

    # ============================================================
    # Ridge-Regularized Least Squares
    # ============================================================
    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform model training.

        Args:
            x_train (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y_train (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.
        """
        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)
        # The polynomial basis is built in normalized coordinates to keep
        # high-order monomials numerically comparable.

        input_dim = x_scaled.shape[1]
        self.powers = self._generate_powers(input_dim)
        phi = self._build_features(x_scaled)
        # Phi has shape (N, P), where P is the number of retained monomials.

        # Normal equations for ridge-regularized least squares:
        # (Phi^T Phi + alpha I) W = Phi^T Y.
        xtx = phi.T @ phi
        xty = phi.T @ y_scaled

        if self.alpha > 0.0:
            # Ridge regularization stabilizes the Gram matrix when the monomial
            # basis becomes nearly collinear.
            np.fill_diagonal(xtx, xtx.diagonal() + self.alpha)

        # Solve for the coefficient matrix W = [w_1, ..., w_C].
        self.weights = np.linalg.solve(xtx, xty)

    # ============================================================
    # Forward Evaluation
    # ============================================================
    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape: (num_samples, target_dim), dtype: float64.
        """
        if self.weights is None or self.powers is None:
            raise RuntimeError("Model has not been fitted.")

        x_scaled = self.scaler_x.transform(x_pred)
        # Rebuild the same monomial basis Phi(x_pred) used in training.
        phi = self._build_features(x_scaled)

        # PRS prediction is the polynomial expansion y_hat = Phi(x) W.
        y_scaled = phi @ self.weights
        # Inverse scaling maps the polynomial output back to the physical target space.
        y_pred = self.scaler_y.inverse_transform(y_scaled)

        return y_pred
