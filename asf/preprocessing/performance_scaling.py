"""
Normalization techniques for algorithm performance data.

This module provides various scaling and transformation methods, primarily
adapted to handle runtime data in algorithm selection.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats
from typing import Any
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


from asf.utils.configurable import ConfigurableMixin


class AbstractNormalization(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator, ConfigurableMixin
):
    """
    Abstract base class for normalization techniques.

    All normalization classes should inherit from this class and implement
    the `transform` and `inverse_transform` methods.
    """

    def __init__(self, **kwargs: Any) -> None:
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

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for sklearn scalers (n_samples, 1)."""
        return np.asarray(X).reshape(-1, 1)

    def _reshape_output(self, X: np.ndarray) -> np.ndarray:
        """Reshape output from sklearn scalers back to 1D."""
        return np.asarray(X).reshape(-1)


class MinMaxNormalization(AbstractNormalization):
    """
    Normalization using Min-Max scaling.
    """

    PREFIX = "min_max"

    def __init__(
        self, feature_range: tuple[float, float] = (0, 1), **kwargs: Any
    ) -> None:
        """
        Initialize MinMaxNormalization.

        Parameters
        ----------
        feature_range : tuple[float, float], default=(0, 1)
            Desired range of transformed data.
        """
        super().__init__(**kwargs)
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
        self.min_max_scale.fit(self._reshape_input(X))
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
        return self._reshape_output(
            self.min_max_scale.transform(self._reshape_input(X))
        )

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
        return self._reshape_output(
            self.min_max_scale.inverse_transform(self._reshape_input(X))
        )


class ZScoreNormalization(AbstractNormalization):
    """
    Normalization using Z-Score scaling.
    """

    PREFIX = "z_score"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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
        self.scaler.fit(self._reshape_input(X))
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
        return self._reshape_output(self.scaler.transform(self._reshape_input(X)))

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
        return self._reshape_output(
            self.scaler.inverse_transform(self._reshape_input(X))
        )


class LogNormalization(AbstractNormalization):
    """
    Normalization using logarithmic scaling.
    """

    PREFIX = "log"

    def __init__(self, base: float = 10.0, eps: float = 1e-6, **kwargs: Any) -> None:
        """
        Initialize LogNormalization.

        Parameters
        ----------
        base : float, default=10.0
            Base of the logarithm.
        eps : float, default=1e-6
            Small constant to avoid log(0).
        """
        super().__init__(**kwargs)
        self.base = float(base)
        self.eps = float(eps)
        self.min_val: float = 0.0

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
        X_shifted = np.asarray(X) - self.min_val + self.eps
        # Clip to avoid non-positive values for log
        X_shifted = np.clip(X_shifted, self.eps, None)
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

    PREFIX = "sqrt"

    def __init__(self, eps: float = 1e-6, **kwargs: Any) -> None:
        """
        Initialize SqrtNormalization.

        Parameters
        ----------
        eps : float, default=1e-6
            Small constant to avoid sqrt(0).
        """
        super().__init__(**kwargs)
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
        X_shifted = np.asarray(X) - self.min_val + self.eps
        # Clip to avoid negative values for sqrt
        X_shifted = np.clip(X_shifted, 0, None)
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
            X_orig = X_orig + self.min_val - self.eps
        return X_orig


class InvSigmoidNormalization(AbstractNormalization):
    """
    Normalization using inverse sigmoid scaling.
    """

    PREFIX = "inv_sigmoid"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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
        self.min_max_scale.fit(self._reshape_input(X))
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
        X_scaled = self._reshape_output(
            self.min_max_scale.transform(self._reshape_input(X))
        )
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
        return self._reshape_output(
            self.min_max_scale.inverse_transform(self._reshape_input(X_logit))
        )


class NegExpNormalization(AbstractNormalization):
    """
    Normalization using negative exponential scaling.
    """

    PREFIX = "neg_exp"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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

    PREFIX = "dummy"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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

    PREFIX = "box_cox"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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
        self.box_cox.fit(self._reshape_input(X))
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
        return self._reshape_output(self.box_cox.transform(self._reshape_input(X)))

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
        X_orig = self._reshape_output(
            self.box_cox.inverse_transform(self._reshape_input(X))
        )
        return X_orig
