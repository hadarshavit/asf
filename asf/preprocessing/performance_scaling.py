import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import scipy.stats
import scipy.special


class AbstractNormalization(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Abstract base class for normalization techniques. All normalization classes
    should inherit from this class and implement the `transform` and `inverse_transform` methods.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "AbstractNormalization":
        """
        Fit the normalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            AbstractNormalization: The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the input data.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        raise NotImplementedError


class MinMaxNormalization(AbstractNormalization):
    """
    Normalization using Min-Max scaling.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        """
        Initialize MinMaxNormalization.

        Args:
            feature_range (tuple[float, float], optional): Desired range of transformed data. Defaults to (0, 1).
        """
        super().__init__()
        self.feature_range = feature_range

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "MinMaxNormalization":
        """
        Fit the Min-Max scaler to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            MinMaxNormalization: The fitted normalization instance.
        """
        self.min_max_scale = MinMaxScaler(feature_range=self.feature_range)
        self.min_max_scale.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using Min-Max scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        return self.min_max_scale.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class ZScoreNormalization(AbstractNormalization):
    """
    Normalization using Z-Score scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "ZScoreNormalization":
        """
        Fit the Z-Score scaler to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            ZScoreNormalization: The fitted normalization instance.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using Z-Score scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.scaler.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        return self.scaler.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class LogNormalization(AbstractNormalization):
    """
    Normalization using logarithmic scaling.
    """

    def __init__(self, base: float = 10, eps: float = 0.0) -> None:
        """
        Initialize LogNormalization.

        Args:
            base (float, optional): Base of the logarithm. Defaults to 10.
            eps (float, optional): Small constant to avoid log(0). Defaults to 1e-6.
        """
        super().__init__()
        self.base = base
        self.eps = eps

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "LogNormalization":
        """
        Fit the LogNormalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            LogNormalization: The fitted normalization instance.
        """
        if X.min() <= 0:
            self.min_val = X.min()
        else:
            self.min_val = 0
            self.eps = 0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using logarithmic scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        X = X - self.min_val + self.eps
        return np.log(X) / np.log(self.base)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        X = np.power(self.base, X)
        if self.min_val != 0:
            X = X + self.min_val - self.eps
        return X


class SqrtNormalization(AbstractNormalization):
    """
    Normalization using square root scaling.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Initialize SqrtNormalization.

        Args:
            eps (float, optional): Small constant to avoid sqrt(0). Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "SqrtNormalization":
        """
        Fit the SqrtNormalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            SqrtNormalization: The fitted normalization instance.
        """
        if X.min() < 0:
            self.min_val = X.min()
            X = X + self.min_val + self.eps
        else:
            self.min_val = 0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using square root scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        X = X + self.min_val + self.eps
        return np.sqrt(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        X = np.power(X, 2)
        if self.min_val != 0:
            X = X - self.min_val - self.eps
        return X


class InvSigmoidNormalization(AbstractNormalization):
    """
    Normalization using inverse sigmoid scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "InvSigmoidNormalization":
        """
        Fit the InvSigmoidNormalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            InvSigmoidNormalization: The fitted normalization instance.
        """
        self.min_max_scale = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
        self.min_max_scale.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using inverse sigmoid scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        X = self.min_max_scale.transform(X.reshape(-1, 1)).reshape(-1)
        return np.log(X / (1 - X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        X = scipy.special.expit(X)
        return self.min_max_scale.inverse_transform(X.reshape(-1, 1)).reshape(-1)


class NegExpNormalization(AbstractNormalization):
    """
    Normalization using negative exponential scaling.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "NegExpNormalization":
        """
        Fit the NegExpNormalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            NegExpNormalization: The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using negative exponential scaling.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return np.exp(-X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        return -np.log(X)


class DummyNormalization(AbstractNormalization):
    """
    Normalization that does not change the data.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "DummyNormalization":
        """
        Fit the DummyNormalization model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            DummyNormalization: The fitted normalization instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data (no change).

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data (no change).

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        return X


class BoxCoxNormalization(AbstractNormalization):
    """
    Normalization using Box-Cox transformation.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray = None, sample_weight: np.ndarray = None
    ) -> "BoxCoxNormalization":
        """
        Fit the Box-Cox transformer to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target values. Defaults to None.
            sample_weight (np.ndarray, optional): Sample weights. Defaults to None.

        Returns:
            BoxCoxNormalization: The fitted normalization instance.
        """
        self.box_cox = PowerTransformer(method="yeo-johnson")
        self.box_cox.fit(X.reshape(-1, 1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using Box-Cox transformation.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.box_cox.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original scale.

        Args:
            X (np.ndarray): Transformed data.

        Returns:
            np.ndarray: Original data.
        """
        X = self.box_cox.inverse_transform(X.reshape(-1, 1)).reshape(-1)
        return X
