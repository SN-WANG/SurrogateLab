# Radial Basis Function (RBF) surrogate model
# Author: Shengning Wang

from typing import Optional

import numpy as np

from utils.scaler import StandardScalerNP


class RBF:
    """
    Classical Gaussian RBF surrogate model.
    """

    def __init__(self, num_centers: int = 20, gamma: Optional[float] = None, alpha: float = 0.0, max_iter: int = 500):
        """
        Initialize the RBF surrogate.

        Args:
            num_centers (int): Kept for backward compatibility.
            gamma (Optional[float]): Gaussian shape parameter.
            alpha (float): Diagonal regularization strength.
            max_iter (int): Kept for backward compatibility.
        """
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances.

        Args:
            x (np.ndarray): Query points. (N, D).
            c (np.ndarray): Reference points. (M, D).

        Returns:
            np.ndarray: Squared distances. (N, M).
        """
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        dists_sq = x_norm_sq + c_norm_sq - 2.0 * (x @ c.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return dists_sq

    def _estimate_gamma(self, x: np.ndarray) -> float:
        """
        Estimate the Gaussian shape parameter from training samples.

        Args:
            x (np.ndarray): Training inputs. (N, D).

        Returns:
            float: Estimated Gaussian shape parameter.
        """
        dists_sq = self._compute_dists(x, x)
        nonzero = dists_sq[dists_sq > 1.0e-12]
        if nonzero.size == 0:
            return 1.0

        median_dist_sq = float(np.median(nonzero))
        if median_dist_sq <= 1.0e-12:
            return 1.0

        return 1.0 / median_dist_sq

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """
        Build the Gaussian RBF correlation matrix.

        Args:
            x (np.ndarray): Query points. (N, D).

        Returns:
            np.ndarray: Correlation matrix. (N, M).
        """
        dists_sq = self._compute_dists(x, self.centers)
        return np.exp(-self.gamma * dists_sq)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the RBF surrogate.

        Args:
            x_train (np.ndarray): Training inputs. (N, D).
            y_train (np.ndarray): Training targets. (N, C).
        """
        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)

        self.centers = x_scaled.copy()
        self.num_centers = x_scaled.shape[0]

        if self.gamma is None:
            self.gamma = self._estimate_gamma(x_scaled)

        phi = self._build_features(x_scaled)
        if self.alpha > 0.0:
            phi = phi.copy()
            np.fill_diagonal(phi, phi.diagonal() + self.alpha)

        try:
            self.weights = np.linalg.solve(phi, y_scaled)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(phi) @ y_scaled

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new inputs.

        Args:
            x_pred (np.ndarray): Prediction inputs. (N, D).

        Returns:
            np.ndarray: Predicted targets. (N, C).
        """
        if self.centers is None or self.weights is None:
            raise RuntimeError("Model has not been fitted.")

        x_scaled = self.scaler_x.transform(x_pred)
        phi = self._build_features(x_scaled)
        y_scaled = phi @ self.weights
        return self.scaler_y.inverse_transform(y_scaled)
