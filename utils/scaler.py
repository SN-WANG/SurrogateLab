# Scalers for Standardization and Normalization
# Author: Shengning Wang

from typing import Literal, Optional

import numpy as np


def _reduce_dims(x: np.ndarray, channel_dim: int) -> tuple[int, ...]:
    channel_axis = channel_dim % x.ndim
    return tuple(dim for dim in range(x.ndim) if dim != channel_axis)


class StandardScalerNP:
    """
    NumPy implementation of channel-wise standardization.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        """
        Initialize the scaler.

        Args:
            eps (float): Lower bound used for near-zero standard deviations.
        """
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "StandardScalerNP":
        """
        Fit channel-wise mean and standard deviation.

        Args:
            x (np.ndarray): Input data. (N, ..., C).
            channel_dim (int): Channel dimension.

        Returns:
            StandardScalerNP: Fitted scaler.
        """
        reduce_dims = _reduce_dims(x, channel_dim)
        self.mean = np.mean(x, axis=reduce_dims, keepdims=True)
        self.std = np.std(x, axis=reduce_dims, keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize input data.

        Args:
            x (np.ndarray): Input data. (N, ..., C).

        Returns:
            np.ndarray: Standardized data. (N, ..., C).
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restore standardized data to the original scale.

        Args:
            x (np.ndarray): Standardized data. (N, ..., C).

        Returns:
            np.ndarray: Restored data. (N, ..., C).
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return x * self.std + self.mean


class MinMaxScalerNP:
    """
    NumPy implementation of channel-wise min-max normalization.
    """

    def __init__(self, norm_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7) -> None:
        """
        Initialize the scaler.

        Args:
            norm_range (Literal["unit", "bipolar"]): Target range.
            eps (float): Lower bound used for near-zero ranges.
        """
        if norm_range == "unit":
            self.a, self.b = 0.0, 1.0
        elif norm_range == "bipolar":
            self.a, self.b = -1.0, 1.0
        else:
            raise ValueError("Invalid norm_range.")

        self.norm_range = norm_range
        self.eps = eps
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "MinMaxScalerNP":
        """
        Fit channel-wise min and max values.

        Args:
            x (np.ndarray): Input data. (N, ..., C).
            channel_dim (int): Channel dimension.

        Returns:
            MinMaxScalerNP: Fitted scaler.
        """
        reduce_dims = _reduce_dims(x, channel_dim)
        self.data_min = np.min(x, axis=reduce_dims, keepdims=True)
        self.data_max = np.max(x, axis=reduce_dims, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input data.

        Args:
            x (np.ndarray): Input data. (N, ..., C).

        Returns:
            np.ndarray: Normalized data. (N, ..., C).
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")
        scale = np.where(self.data_max - self.data_min < self.eps, 1.0, self.data_max - self.data_min)
        return self.a + (x - self.data_min) * (self.b - self.a) / scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restore normalized data to the original scale.

        Args:
            x (np.ndarray): Normalized data. (N, ..., C).

        Returns:
            np.ndarray: Restored data. (N, ..., C).
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")
        scale = np.where(self.data_max - self.data_min < self.eps, 1.0, self.data_max - self.data_min)
        return (x - self.a) * scale / (self.b - self.a) + self.data_min
