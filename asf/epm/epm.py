from functools import partial
from typing import Type, Union, Optional

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin

from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.predictors import SklearnWrapper
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor
from sklearn.base import TransformerMixin
from asf.predictors.abstract_predictor import AbstractPredictor


class EPM:
    """
    The EPM (Empirical Performance Model) class is a wrapper for machine learning models
    that includes preprocessing, normalization, and optional inverse transformation of predictions.

    Attributes:
        predictor_class (Type[AbstractPredictor] | Type[RegressorMixin]): The class of the predictor to use.
        normalization_class (Type[AbstractNormalization]): The normalization class to apply to the target variable.
        transform_back (bool): Whether to apply inverse transformation to predictions.
        features_preprocessing (Union[str, TransformerMixin]): Preprocessing pipeline for features.
        predictor_config (Optional[dict]): Configuration for the predictor.
        predictor_kwargs (Optional[dict]): Additional keyword arguments for the predictor.
    """

    def __init__(
        self,
        predictor_class: Union[Type[AbstractPredictor], Type[RegressorMixin]],
        normalization_class: Type[AbstractNormalization] = LogNormalization,
        transform_back: bool = True,
        features_preprocessing: Union[str, TransformerMixin] = "default",
        categorical_features: Optional[list] = None,
        numerical_features: Optional[list] = None,
        predictor_config: Optional[dict] = None,
        predictor_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the EPM model.

        Parameters:
            predictor_class (Type[AbstractPredictor] | Type[RegressorMixin]): The class of the predictor to use.
            normalization_class (Type[AbstractNormalization]): The normalization class to apply to the target variable.
            transform_back (bool): Whether to apply inverse transformation to predictions.
            features_preprocessing (Union[str, TransformerMixin]): Preprocessing pipeline for features.
            categorical_features (Optional[list]): List of categorical feature names.
            numerical_features (Optional[list]): List of numerical feature names.
            predictor_config (Optional[dict]): Configuration for the predictor.
            predictor_kwargs (Optional[dict]): Additional keyword arguments for the predictor.
        """
        if isinstance(predictor_class, type) and issubclass(
            predictor_class, (RegressorMixin)
        ):
            self.model_class = partial(SklearnWrapper, predictor_class)
        else:
            self.model_class = predictor_class

        self.predictor_class = predictor_class
        self.normalization_class = normalization_class
        self.transform_back = transform_back
        self.predictor_config = predictor_config
        self.predictor_kwargs = predictor_kwargs or {}
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
        X: Union[pd.DataFrame, pd.Series, list],
        y: Union[pd.Series, list],
        sample_weight: Optional[list] = None,
    ) -> "EPM":
        """
        Fit the EPM model to the data.

        Parameters:
            X (Union[pd.DataFrame, pd.Series, list]): Features.
            y (Union[pd.Series, list]): Target variable.
            sample_weight (Optional[list]): Sample weights (optional).

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

        if self.predictor_config is None:
            self.predictor = self.predictor_class(**self.predictor_kwargs)
        else:
            self.predictor = self.predictor_class.get_from_configuration(
                self.predictor_config, **self.predictor_kwargs
            )()

        self.predictor.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series, list]) -> list:
        """
        Predict using the fitted EPM model.

        Parameters:
            X (Union[pd.DataFrame, pd.Series, list]): Features.

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

        y_pred = self.predictor.predict(X)

        if self.transform_back:
            y_pred = self.normalization.inverse_transform(y_pred)

        return y_pred
