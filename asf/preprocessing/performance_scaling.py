"""
Normalization techniques for algorithm performance data.

This module provides various scaling and transformation methods, primarily
adapted to handle runtime data in algorithm selection.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


class AbstractNormalization(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Abstract base class for normalization techniques.

    All normalization classes should inherit from this class and implement
    the `transform` and `inverse_transform` methods.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> AbstractNormalization:
        """
        Fit the normalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        AbstractNormalization
            The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the input data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data.

        Returns
        -------
        np.ndarray
            Original data.
        """
        raise NotImplementedError


class MinMaxNormalization(AbstractNormalization):
    """
    Normalization using Min-Max scaling.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        """
        Initialize MinMaxNormalization.

        Parameters
        ----------
        feature_range : tuple[float, float], default=(0, 1)
            Desired range of transformed data.
        """
        super().__init__()
        self.feature_range = feature_range

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> MinMaxNormalization:
        """
        Fit the Min-Max scaler to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        MinMaxNormalization
            The fitted normalization instance.
        """
        self.min_max_scale = MinMaxScaler(feature_range=self.feature_range)
        self.min_max_scale.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using Min-Max scaling.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Parameters
        ----------
        X : np.ndarray
            Transformed data.

        Returns
        -------
        np.ndarray
            Original data.
        """
        return self.min_max_scale.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class ZScoreNormalization(AbstractNormalization):
    """
    Normalization using Z-Score scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> ZScoreNormalization:
        """
        Fit the Z-Score scaler to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        ZScoreNormalization
            The fitted normalization instance.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using Z-Score scaling.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.scaler.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Parameters
        ----------
        X : np.ndarray
            Transformed data.

        Returns
        -------
        np.ndarray
            Original data.
        """
        return self.scaler.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class LogNormalization(AbstractNormalization):
    """
    Normalization using logarithmic scaling.
    """

    def __init__(self, base: float = 10.0, eps: float = 1e-6) -> None:
        """
        Initialize LogNormalization.

        Parameters
        ----------
        base : float, default=10.0
            Base of the logarithm.
        eps : float, default=1e-6
            Small constant to avoid log(0).
        """
        super().__init__()
        self.base = base
        self.eps = eps

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LogNormalization:
        """
        Fit the LogNormalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        LogNormalization
            The fitted normalization instance.
        """
        x_min = np.min(np.asarray(X))
        if x_min <= 0:
            self.min_val = x_min
        else:
            self.min_val = 0.0
            self.eps = 0.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data using logarithmic scaling.

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        X_shifted = X - self.min_val + self.eps
        return np.log(X_shifted) / np.log(self.base)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data back to the original scale.

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        X_orig = np.power(self.base, X)
        if self.min_val != 0:
            X_orig = X_orig + self.min_val - self.eps
        return X_orig


class SqrtNormalization(AbstractNormalization):
    """
    Normalization using square root scaling.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Initialize SqrtNormalization.

        Parameters
        ----------
        eps : float, default=1e-6
            Small constant to avoid sqrt(0).
        """
        super().__init__()
        self.eps = eps

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> SqrtNormalization:
        """
        Fit the SqrtNormalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        SqrtNormalization
            The fitted normalization instance.
        """
        x_min = np.min(np.asarray(X))
        if x_min < 0:
            self.min_val = x_min
        else:
            self.min_val = 0.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data using square root scaling.

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        X_shifted = X + self.min_val + self.eps
        return np.sqrt(X_shifted)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data back to the original scale.

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        X_orig = np.power(X, 2)
        if self.min_val != 0:
            X_orig = X_orig - self.min_val - self.eps
        return X_orig


class InvSigmoidNormalization(AbstractNormalization):
    """
    Normalization using inverse sigmoid scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> InvSigmoidNormalization:
        """
        Fit the InvSigmoidNormalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        InvSigmoidNormalization
            The fitted normalization instance.
        """
        self.min_max_scale = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
        self.min_max_scale.fit(np.asarray(X).reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data using inverse sigmoid scaling.

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        X_scaled = self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)
        return np.log(X_scaled / (1 - X_scaled))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data back to the original scale.

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        X_logit = scipy.special.expit(X)
        return self.min_max_scale.inverse_transform(X_logit.reshape(-1, 1)).reshape(-1)


class NegExpNormalization(AbstractNormalization):
    """
    Normalization using negative exponential scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> NegExpNormalization:
        """
        Fit the NegExpNormalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        NegExpNormalization
            The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data using negative exponential scaling.

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        return np.exp(-X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data back to the original scale.

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        return -np.log(X)


class DummyNormalization(AbstractNormalization):
    """
    Normalization that does not change the data.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> DummyNormalization:
        """
        Fit the DummyNormalization model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        DummyNormalization
            The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data (no change).

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data (no change).

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        return X


class BoxCoxNormalization(AbstractNormalization):
    """
    Normalization using Box-Cox transformation (Yeo-Johnson variant).
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BoxCoxNormalization:
        """
        Fit the Box-Cox transformer to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray or None, default=None
            Target values.
        sample_weight : np.ndarray or None, default=None
            Sample weights.

        Returns
        -------
        BoxCoxNormalization
            The fitted normalization instance.
        """
        self.box_cox = PowerTransformer(method="yeo-johnson")
        self.box_cox.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Transform the input data using Box-Cox transformation.

                Parameters
                ----------
                X : np.ndarray
                    Input data.

                Returns
        -------
                np.ndarray
                    Transformed data.
        """
        return self.box_cox.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
                Inverse transform the data back to the original scale.

                Parameters
                ----------
                X : np.ndarray
                    Transformed data.

                Returns
        -------
                np.ndarray
                    Original data.
        """
        X_orig = self.box_cox.inverse_transform(X.reshape(-1, 1)).reshape(-1)
        return X_orig
