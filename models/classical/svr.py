# Support Vector Regression (SVR) Surrogate Model
# Author: Shengning Wang

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Tuple, Optional, Literal

from utils.scaler import StandardScalerNP


class SVR:
    """
    Support Vector Regression (SVR) using Dual Optimization with epsilon-insensitive loss.
    """

    def __init__(self, kernel: Literal["rbf", "linear"] = "rbf", gamma: Optional[float] = None,
                 C: float = 1.0, epsilon: float = 0.1):
        """
        Initializes the SVR configuration.

        Args:
            kernel (str): Kernel type ("rbf" or "linear").
            gamma (Optional[float]): Kernel coefficient for rbf.
            C (float): Regularization parameter (penalty).
            epsilon (float): Epsilon-tube width (tolerance margin).
        """
        # parameters
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.support_vectors_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted = False

    # ============================================================
    # Kernel Matrix
    # ============================================================
    # ------------------------------------------------------------------

    def _build_features(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Constructs SVR feature matrix.

        Args:
            x1 (np.ndarray): Shape (num_samples_1, input_dim), dtype: float64.
            x2 (np.ndarray): Shape (num_samples_2, input_dim), dtype: float64.

        Returns:
            np.ndarray: Feature matrix. Shape: (num_samples_1, num_samples_2), dtype: float64.
        """
        if self.kernel == "linear":
            # Linear kernel:
            # K(x_i, x_j) = x_i^T x_j.
            # This corresponds to a primal hyperplane in the original feature space.
            return x1 @ x2.T
        elif self.kernel == "rbf":
            # RBF kernel:
            # K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2).
            # The squared distance is expanded in matrix form for vectorized evaluation.
            dists = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2.0 * (x1 @ x2.T)
            return np.exp(-self.gamma * dists)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    # ============================================================
    # Dual Quadratic Program
    # ============================================================
    # ------------------------------------------------------------------

    def _solve_dual(self, phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solves the dual QP problem for a single output dimension.

        Minimize: 0.5 * beta^T @ phi @ beta - beta^T @ y + epsilon * |beta|
        Subject to: sum(beta) = 0, -C <= beta <= C
        Note: beta = alpha - alpha*

        Args:
            phi (np.ndarray): Feature matrix of shape (num_samples, num_samples), dtype: float64.
            y (np.ndarray): Target vector of shape (num_samples,), dtype: float64.

        Returns:
            Tuple[np.ndarray, float]: Dual coefficients (beta) and intercept (bias).
                beta shape: (num_samples,), dtype: float64.
                bias dtype: float.
        """
        num_samples = y.shape[0]

        # Optimize the standard epsilon-SVR dual variables
        # x = [alpha; alpha*], with beta = alpha - alpha*.
        # The dual objective is:
        # 0.5 * beta^T K beta - y^T beta + epsilon * sum(alpha + alpha*).
        # The epsilon tube enforces zero loss whenever |y - f(x)| <= epsilon.

        def objective(x):
            alpha, alpha_star = x[:num_samples], x[num_samples:]
            beta = alpha - alpha_star

            # quadratic term: 0.5 * beta.T @ phi @ beta
            term1 = 0.5 * beta @ phi @ beta

            # linear term: epsilon * sum(alpha + alpha*) - beta.T @ y
            term2 = self.epsilon * np.sum(alpha + alpha_star) - y @ beta
            # The dual objective is convex because the kernel Gram matrix K is PSD.

            return term1 + term2

        # manually providing jacobian to speed up SLSQP by avoiding finite difference
        def jacobian(x):
            alpha, alpha_star = x[:num_samples], x[num_samples:]
            beta = alpha - alpha_star
            # grad_beta = K beta - y, then map to alpha and alpha* blocks.
            grad_base = phi @ beta - y
            grad_alpha = grad_base + self.epsilon
            grad_alpha_star = -grad_base + self.epsilon
            return np.concatenate([grad_alpha, grad_alpha_star])

        # Equality constraint:
        # sum_i (alpha_i - alpha_i*) = sum_i beta_i = 0.
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:num_samples] - x[num_samples:])}]
        # This zero-sum constraint is what allows the intercept b to appear in the predictor.

        # Box constraints:
        # 0 <= alpha_i, alpha_i* <= C.
        bounds = Bounds(np.zeros(2 * num_samples), np.full(2 * num_samples, self.C))
        # Larger C penalizes tube violations more strongly and makes the fit less regularized.

        # initial guess
        x0 = np.zeros(2 * num_samples)

        # solve QP
        res = minimize(fun=objective, x0=x0, method="SLSQP", jac=jacobian, bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-6, "maxiter": 200})

        if not res.success:
            # warning: optimization might fail on hard problems, fallback or log could be added
            pass

        alpha_final, alpha_star_final = np.split(res.x, 2)
        beta = alpha_final - alpha_star_final
        # beta_i > 0 and beta_i < 0 correspond to opposite sides of the epsilon tube.

        # Small |beta_i| values are pruned to recover sparse support vectors.
        beta[np.abs(beta) < 1e-5] = 0.0

        # Free support vectors satisfy 0 < |beta_i| < C and can be used to recover
        # the bias from y_i = sum_j beta_j K_ij + b +/- epsilon.
        sv_mask = (np.abs(beta) > 1e-5) & (np.abs(beta) < self.C - 1e-5)

        if np.any(sv_mask):
            # b = y - phi @ beta - sign(beta) * epsilon
            biases = y[sv_mask] - phi[sv_mask] @ beta - np.sign(beta[sv_mask]) * self.epsilon

            # take mean for numerical stability
            intercept = np.mean(biases)
        else:
            # fallback if no free support vectors are found
            intercept = 0.0

        # The returned pair (beta, b) defines f(x) = sum_i beta_i K(x, x_i) + b.
        return beta, intercept

    # ============================================================
    # Multi-Output Training
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
        # Training is carried out in standardized coordinates so the kernel
        # scale and epsilon tube are not dominated by raw engineering units.

        if self.gamma is None:
            x_var = x_scaled.var()
            # Common kernel-width heuristic:
            # gamma = 1 / (D * Var(X)).
            self.gamma = 1.0 / (x_scaled.shape[1] * x_var) if x_var > 0 else 1.0

        phi = self._build_features(x_scaled, x_scaled)
        # phi is the Gram matrix K(X, X) used by every output dimension.

        # storage for multi-target
        num_samples, target_dim = y_scaled.shape
        self.dual_coef_ = np.zeros((num_samples, target_dim))
        self.intercept_ = np.zeros(target_dim)
        self.support_vectors_ = x_scaled  # store scaled support vectors
        # The same support-vector matrix X is reused across all output channels.

        # fit per target dimension
        for d in range(target_dim):
            # Each output solves an independent epsilon-SVR dual problem.
            beta, bias = self._solve_dual(phi, y_scaled[:, d])
            self.dual_coef_[:, d] = beta
            self.intercept_[d] = bias

        self.fitted = True

    # ============================================================
    # Kernel Prediction
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
        if not self.fitted:
            raise RuntimeError("Model has not been fitted.")

        x_scaled = self.scaler_x.transform(x_pred)
        # Cross-kernel evaluation K(X_pred, X_sv) transfers the support-vector
        # representation from the training set to the prediction points.
        phi = self._build_features(x_scaled, self.support_vectors_)

        # Dual-form predictor:
        # y_hat(x) = sum_i beta_i K(x, x_i) + b.
        # Matrix form:
        # Y_hat = K(X_pred, X_sv) B + b.
        y_scaled = phi @ self.dual_coef_ + self.intercept_
        y_pred = self.scaler_y.inverse_transform(y_scaled)

        return y_pred
