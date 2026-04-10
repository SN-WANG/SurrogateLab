# Multi-Fidelity Infill via MICO (Mutual Information and Correlation Criterion)
# Paper reference: https://doi.org/10.1007/s00366-023-01858-z
# Paper author: Shuo Wang, Xiaonan Lai, Xiwang He, Kunpeng Li, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import List, Optional, Tuple

from sampling.base_infill import BaseInfill
from utils.scaler import MinMaxScalerNP


class MultiFidelityInfill(BaseInfill):
    """
    Greedy multi-fidelity infill based on the MICO score.

    The LF pool is treated as a discrete candidate set. evaluate() maps
    continuous queries to the nearest normalised LF node, while propose()
    greedily returns the remaining LF node with the largest MICO score.
    """

    def __init__(
        self,
        model,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        target_idx: int = 0,
        ratio: float = 0.5,
        theta_v: Optional[np.ndarray] = None,
        theta_d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the MICO-based multi-fidelity infill strategy.

        Args:
            model: Fitted surrogate model passed through the base contract.
            x_hf (np.ndarray): HF training locations. (num_hf, input_dim).
            y_hf (np.ndarray): HF training responses. (num_hf, output_dim).
            x_lf (np.ndarray): LF candidate pool. (num_lf, input_dim).
            y_lf (np.ndarray): LF responses on the pool. (num_lf, output_dim).
            target_idx (int): Output dimension used for scoring. Default 0.
            ratio (float): Kept for compatibility, but not used in scoring.
            theta_v (Optional[np.ndarray]): LF correlation lengths. If None,
                estimate them from the normalised LF point set.
            theta_d (Optional[np.ndarray]): Discrepancy correlation lengths.
                If None, copy theta_v.
        """
        super().__init__(model, bounds=None, target_idx=target_idx, num_restarts=0)

        self.x_hf = np.asarray(x_hf, dtype=np.float64)
        self.y_hf = np.asarray(y_hf, dtype=np.float64)
        self.x_lf = np.asarray(x_lf, dtype=np.float64)
        self.y_lf = np.asarray(y_lf, dtype=np.float64)

        self.num_lf, self.input_dim = self.x_lf.shape
        self.num_hf = self.x_hf.shape[0]
        self.output_dim = self.y_lf.shape[1]
        self.ratio = float(ratio)

        self._scaler_x = MinMaxScalerNP(norm_range="unit")
        self.x_lf_norm = self._scaler_x.fit(self.x_lf, channel_dim=1).transform(self.x_lf)
        self.x_hf_norm = self._scaler_x.transform(self.x_hf)

        self.hf_idxs = self._map_hf_to_lf()
        self.selected_idxs = np.unique(self.hf_idxs).astype(np.int64)
        self.y_lf_at_hf = self.y_lf[self.hf_idxs]

        self.theta_v = self._init_theta(theta_v)
        self.theta_d = self._init_theta(theta_d) if theta_d is not None else self.theta_v.copy()
        self.rho = self._estimate_rho()
        self.sigma_sq_v = self._estimate_sigma_sq_v()
        self.sigma_sq_d = self._estimate_sigma_sq_d()

    def _init_theta(self, theta: Optional[np.ndarray]) -> np.ndarray:
        if theta is not None:
            return np.asarray(theta, dtype=np.float64).flatten()
        return self._estimate_theta_from_points(self.x_lf_norm)

    def _estimate_theta_from_points(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 2:
            return np.ones(self.input_dim, dtype=np.float64)

        theta = np.ones(self.input_dim, dtype=np.float64)
        for dim in range(self.input_dim):
            diffs = x[:, dim][:, None] - x[:, dim][None, :]
            sq = diffs ** 2
            nz = sq[sq > 1e-12]
            if nz.size > 0:
                theta[dim] = 1.0 / max(float(np.median(nz)), 1e-12)
        return theta

    def _estimate_rho(self) -> np.ndarray:
        rho = np.ones(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            yl = self.y_lf_at_hf[:, d]
            yh = self.y_hf[:, d]
            denom = float(yl @ yl)
            if denom > 1e-12:
                rho[d] = float((yl @ yh) / denom)
        return rho

    def _estimate_sigma_sq_v(self) -> np.ndarray:
        ddof = 1 if self.y_lf.shape[0] > 1 else 0
        sigma_sq_v = np.var(self.y_lf, axis=0, ddof=ddof).astype(np.float64)
        return np.maximum(sigma_sq_v, 1e-12)

    def _estimate_sigma_sq_d(self) -> np.ndarray:
        ddof = 1 if self.y_hf.shape[0] > 1 else 0
        sigma_sq_d = np.zeros(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            discrepancy = self.y_hf[:, d] - self.rho[d] * self.y_lf_at_hf[:, d]
            sigma_sq_d[d] = max(float(np.var(discrepancy, ddof=ddof)), 1e-12)
        return sigma_sq_d

    def _map_hf_to_lf(self) -> np.ndarray:
        dists = self._compute_sq_dists(self.x_hf_norm, self.x_lf_norm)
        return np.argmin(dists, axis=1).astype(np.int64)

    def _remaining_candidate_indices(self, selected_idxs: np.ndarray) -> np.ndarray:
        candidate_mask = np.ones(self.num_lf, dtype=bool)
        candidate_mask[np.asarray(selected_idxs, dtype=np.int64)] = False
        return np.where(candidate_mask)[0].astype(np.int64)

    def _compute_sq_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_sq = np.sum(c ** 2, axis=1)
        d_sq = x_sq + c_sq - 2.0 * (x @ c.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return d_sq

    def _correlation_matrix(self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x1s = x1 * np.sqrt(theta)
        x2s = x2 * np.sqrt(theta)
        x1_sq = np.sum(x1s ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(x2s ** 2, axis=1)
        d_sq = x1_sq + x2_sq - 2.0 * (x1s @ x2s.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return np.exp(-d_sq)

    def _mf_covariance(self, x1: np.ndarray, x2: np.ndarray, out_idx: int) -> np.ndarray:
        rho_d = self.rho[out_idx]
        sigma_v = self.sigma_sq_v[out_idx]
        sigma_d = self.sigma_sq_d[out_idx]
        psi_v = self._correlation_matrix(x1, x2, self.theta_v)
        psi_d = self._correlation_matrix(x1, x2, self.theta_d)
        return (rho_d ** 2 * sigma_v) * psi_v + sigma_d * psi_d

    def _compute_mico_scores(
        self,
        candidate_idxs: np.ndarray,
        selected_idxs: List[int],
        out_idx: int,
    ) -> np.ndarray:
        candidate_idxs = np.asarray(candidate_idxs, dtype=np.int64)
        selected_idxs = np.unique(np.asarray(selected_idxs, dtype=np.int64))
        num_cands = candidate_idxs.size
        if num_cands == 0:
            return np.zeros(0, dtype=np.float64)

        x_candidates = self.x_lf_norm[candidate_idxs]
        if selected_idxs.size == 0:
            c_vv = self._mf_covariance(x_candidates, x_candidates, out_idx)
            c_vv += np.eye(num_cands, dtype=np.float64) * 1e-6
            icov_vv = np.linalg.pinv(c_vv)
            delta_n = np.maximum(np.diag(c_vv), 1e-12)
            delta_d = np.maximum(np.diag(icov_vv), 1e-12)
            return delta_n * delta_d

        x_selected = self.x_lf_norm[selected_idxs]
        c_aa = self._mf_covariance(x_selected, x_selected, out_idx)
        c_aa += np.eye(selected_idxs.size, dtype=np.float64) * 1e-6
        icov_a = np.linalg.pinv(c_aa)

        c_ya = self._mf_covariance(x_candidates, x_selected, out_idx)
        c_vv = self._mf_covariance(x_candidates, x_candidates, out_idx)
        c_vv += np.eye(num_cands, dtype=np.float64) * 1e-6
        icov_vv = np.linalg.pinv(c_vv)

        temp = c_ya @ icov_a @ c_ya.T
        delta_n = np.maximum(np.diag(c_vv - temp), 1e-12)
        delta_d = np.maximum(np.diag(icov_vv), 1e-12)
        return delta_n * delta_d

    def _compute_full_pool_scores(self, out_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        candidates = self._remaining_candidate_indices(self.selected_idxs)
        scores = self._compute_mico_scores(candidates, self.selected_idxs.tolist(), out_idx)
        return candidates, scores

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the MICO score at continuous coordinates.

        Args:
            x (np.ndarray): Query coordinates. (N, input_dim).

        Returns:
            np.ndarray: MICO scores. (N, 1).
        """
        x = np.asarray(x, dtype=np.float64)
        x_norm = self._scaler_x.transform(x)
        dists = self._compute_sq_dists(x_norm, self.x_lf_norm)
        idxs = np.argmin(dists, axis=1).astype(np.int64)

        candidates, scores = self._compute_full_pool_scores(self.target_idx)
        score_map = {int(idx): float(score) for idx, score in zip(candidates, scores)}
        values = np.array([score_map.get(int(idx), 0.0) for idx in idxs], dtype=np.float64)
        return values.reshape(-1, 1)

    def propose(self) -> np.ndarray:
        """
        Greedily return the remaining LF node with the largest MICO score.

        Returns:
            np.ndarray: Proposed design point. (1, input_dim).
        """
        candidates, scores = self._compute_full_pool_scores(self.target_idx)
        if candidates.size == 0:
            fallback = int(np.random.randint(self.num_lf))
            return self.x_lf[fallback].reshape(1, -1)

        best_idx = int(candidates[int(np.argmax(scores))])
        return self.x_lf[best_idx].reshape(1, -1)
