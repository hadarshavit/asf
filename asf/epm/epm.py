import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor


class AbstractEPM:
    """
    The EPM (Empirical Performance Model) class is a wrapper for machine learning models
    that includes preprocessing, normalization, and optional inverse transformation of predictions.

    Attributes:
        predictor_class (type[AbstractPredictor] | type[RegressorMixin]): The class of the predictor to use.
        normalization_class (type[AbstractNormalization]): The normalization class to apply to the target variable.
        transform_back (bool): Whether to apply inverse transformation to predictions.
        features_preprocessing (str | TransformerMixin): Preprocessing pipeline for features.
        predictor_config (dict | None): Configuration for the predictor.
        predictor_kwargs (dict | None): Additional keyword arguments for the predictor.
    """

    def __init__(
        self,
        normalization_class: type[AbstractNormalization] = LogNormalization,
        transform_back: bool = True,
        features_preprocessing: str | TransformerMixin = "default",
        categorical_features: list | None = None,
        numerical_features: list | None = None,
        predictor_config: dict | None = None,
        predictor_kwargs: dict | None = None,
        imputer: callable = None,
    ):
        """
        Initialize the EPM model.

        Parameters:
            predictor_class (type[AbstractPredictor] | type[RegressorMixin]): The class of the predictor to use.
            normalization_class (type[AbstractNormalization]): The normalization class to apply to the target variable.
            transform_back (bool): Whether to apply inverse transformation to predictions.
            features_preprocessing (str | TransformerMixin): Preprocessing pipeline for features.
            categorical_features (list | None): List of categorical feature names.
            numerical_features (list | None): List of numerical feature names.
            predictor_config (dict | None): Configuration for the predictor.
            predictor_kwargs (dict | None): Additional keyword arguments for the predictor.
        """
        self.normalization_class = normalization_class
        self.transform_back = transform_back
        self.predictor_config = predictor_config
        self.predictor_kwargs = predictor_kwargs or {}
        self.imputer = imputer
        self.numpy = False

        if features_preprocessing == "default":
            self.features_preprocessing = get_default_preprocessor(
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
        else:
            self.features_preprocessing = features_preprocessing

    def fit(
        self,
        X: pd.DataFrame | pd.Series | list,
        y: pd.Series | list,
        sample_weight: list | None = None,
    ) -> "AbstractEPM":
        """
        Fit the EPM model to the data.

        Parameters:
            X (pd.DataFrame | pd.Series | list): Features.
            y (pd.Series | list): Target variable.
            sample_weight (list | None): Sample weights (optional).

        Returns:
            EPM: The fitted EPM model.
        """
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X = pd.DataFrame(
                X,
                index=range(len(X)),
                columns=[f"f_{i}" for i in range(X.shape[1])],
            )
            y = pd.Series(
                y,
                index=range(len(y)),
            )
            self.numpy = True

        if self.features_preprocessing is not None:
            X = self.features_preprocessing.fit_transform(X)

        self.normalization = self.normalization_class()
        self.normalization.fit(y)
        y = self.normalization.transform(y)

        if self.imputer is not None:
            y = self.imputer(y, X)

        self._fit(X, y, sample_weight)

        return self

    def _fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: list | None):
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, X: pd.DataFrame | pd.Series | list) -> list:
        """
        Predict using the fitted EPM model.

        Parameters:
            X (pd.DataFrame | pd.Series | list): Features.

        Returns:
            list: Predicted values.
        """
        if self.numpy:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(
                    X,
                    index=range(len(X)),
                    columns=[f"f_{i}" for i in range(X.shape[1])],
                )

        if self.features_preprocessing is not None:
            X = self.features_preprocessing.transform(X)

        y_pred = self._predict(X)

        if self.transform_back:
            y_pred = self.normalization.inverse_transform(y_pred)

        return y_pred

    def _predict(self, X: pd.DataFrame) -> list:
        raise NotImplementedError("Subclasses must implement this method.")
