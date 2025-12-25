"""
EPM (Empirical Performance Model) - A wrapper for machine learning models with preprocessing.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, TransformerMixin

from asf.predictors import SklearnWrapper
from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.random_forest import RandomForestRegressorWrapper
from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor


class EPM:
    """
    Empirical Performance Model wrapper.

    The EPM (Empirical Performance Model) class is a wrapper for machine learning models
    that includes preprocessing, normalization, and optional inverse transformation of predictions.

    Parameters
    ----------
    predictor_class : type[AbstractPredictor] or type[RegressorMixin], default=RandomForestRegressorWrapper
        The class of the predictor to use.
    normalization_class : type[AbstractNormalization], default=LogNormalization
        The normalization class to apply to the target variable.
    transform_back : bool, default=True
        Whether to apply inverse transformation to predictions.
    features_preprocessing : str or TransformerMixin, default="default"
        Preprocessing pipeline for features.
    categorical_features : list or None, default=None
        List of categorical feature names.
    numerical_features : list or None, default=None
        List of numerical feature names.
    predictor_config : dict or None, default=None
        Configuration for the predictor.
    predictor_kwargs : dict or None, default=None
        Additional keyword arguments for the predictor.
    imputer : Callable or None, default=None
        Optional imputer function for target variables.
    """

    def __init__(
        self,
        predictor_class: type[AbstractPredictor]
        | type[RegressorMixin] = RandomForestRegressorWrapper,
        normalization_class: type[AbstractNormalization] = LogNormalization,
        transform_back: bool = True,
        features_preprocessing: str | TransformerMixin | None = "default",
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        predictor_config: dict[str, Any] | None = None,
        predictor_kwargs: dict[str, Any] | None = None,
        imputer: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None,
    ) -> None:
        if isinstance(predictor_class, type) and issubclass(
            predictor_class, RegressorMixin
        ):
            self.model_class: Any = partial(SklearnWrapper, predictor_class)
        else:
            self.model_class = predictor_class

        self.predictor_class = predictor_class
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
        X: pd.DataFrame | pd.Series | np.ndarray | list[Any],
        y: pd.Series | np.ndarray | list[Any],
        sample_weight: list[float] | np.ndarray | None = None,
    ) -> EPM:
        """
        Fit the EPM model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, np.ndarray, or list
            Input features.
        y : pd.Series, np.ndarray, or list
            Target values.
        sample_weight : list, np.ndarray, or None, default=None
            Sample weights.

        Returns
        -------
        EPM
            The fitted model.
        """
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X_df = pd.DataFrame(
                X,
                index=range(len(X)),
                columns=pd.Index([f"f_{i}" for i in range(X.shape[1])]),
            )
            y_ser = pd.Series(
                y,
                index=range(len(y)),
            )
            self.numpy = True
        else:
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y

        if self.features_preprocessing is not None and not isinstance(
            self.features_preprocessing, str
        ):
            X_df = self.features_preprocessing.fit_transform(X_df)  # type: ignore

        self.normalization = self.normalization_class()
        self.normalization.fit(np.asarray(y_ser))
        y_ser_scaled = self.normalization.transform(np.asarray(y_ser))
        y_ser = pd.Series(y_ser_scaled, index=y_ser.index)

        if self.imputer is not None:
            y_ser = self.imputer(y_ser, X_df)

        self.predictor = self._get_predictor()

        self.predictor.fit(X_df, y_ser, sample_weight=sample_weight)
        return self

    def _get_predictor(self) -> AbstractPredictor:
        """Get the predictor instance."""
        if self.predictor_config is None:
            predictor = self.predictor_class(**self.predictor_kwargs)
        else:
            # Assume get_from_configuration returns a partial or a class
            predictor_factory = self.predictor_class.get_from_configuration(  # type: ignore
                self.predictor_config, **self.predictor_kwargs
            )
            if callable(predictor_factory):
                predictor = predictor_factory()
            else:
                predictor = predictor_factory

        if not isinstance(predictor, AbstractPredictor):
            raise TypeError(f"Predictor {predictor} is not an AbstractPredictor")
        return predictor

    def predict(
        self, X: pd.DataFrame | pd.Series | np.ndarray | list[Any]
    ) -> np.ndarray:
        """
        Predict targets.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, np.ndarray, or list
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.numpy:
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(
                    X,
                    index=range(len(X)),
                    columns=pd.Index([f"f_{i}" for i in range(X.shape[1])]),
                )
            else:
                X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        else:
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.features_preprocessing is not None and not isinstance(
            self.features_preprocessing, str
        ):
            X_df = self.features_preprocessing.transform(X_df)  # type: ignore

        y_pred = self.predictor.predict(X_df)

        if self.transform_back:
            y_pred = self.normalization.inverse_transform(y_pred)

        return np.asarray(y_pred)
