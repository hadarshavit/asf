from __future__ import annotations

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Constant,
        Float,
        Integer,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from typing import Any
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor, XGBClassifier, XGBRanker

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin


class XGBoostClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for the XGBoost classifier to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_classifier"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the XGBoostClassifierWrapper.

        Parameters
        ----------
        init_params : dict or None
            Initialization parameters for the XGBoost classifier (backward compatibility).
        **kwargs : Any
            Initialization parameters for the XGBoost classifier.
        """
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it using pip install asf-lib[xgb]."
            )
        super().__init__(XGBClassifier, init_params=init_params, **kwargs)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        Y : np.ndarray
            Target values of shape (n_samples,).
        sample_weight : np.ndarray, optional
            Sample weights of shape (n_samples,) (default is None).
        **kwargs : Any
            Additional keyword arguments for the scikit-learn model's `fit` method.
        """
        X_in = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        Y_in = Y.to_numpy() if isinstance(Y, (pd.DataFrame, pd.Series)) else Y
        Y_in = np.asarray(Y_in)
        if Y_in.ndim > 1:
            Y_in = Y_in.reshape(-1)

        if sample_weight is not None and isinstance(
            sample_weight, (pd.DataFrame, pd.Series)
        ):
            sample_weight = sample_weight.to_numpy()
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).reshape(-1)

        if Y_in.dtype == bool:
            self.bool_labels = True
        else:
            self.bool_labels = False

        self.model_class.fit(X_in, Y_in, sample_weight=sample_weight, **kwargs)

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments for the scikit-learn model's `predict` method.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        X_in = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        if self.bool_labels:
            return self.model_class.predict(X_in, **kwargs).astype(bool)
        return self.model_class.predict(X_in, **kwargs)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for XGBoost classifier."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Constant("booster", "gbtree"),
            Constant("n_estimators", 2000),
            Integer("max_depth", (1, 11), log=False, default=8),
            Integer("min_child_weight", (1, 100), log=True, default=39),
            Float(
                "colsample_bytree", (0.0, 1.0), log=False, default=0.2545374925231651
            ),
            Float(
                "colsample_bylevel", (0.0, 1.0), log=False, default=0.6909224923784677
            ),
            Float("lambda", (0.001, 1000), log=True, default=31.393252465064943),
            Float("alpha", (0.001, 1000), log=True, default=0.24167936088332426),
            Float(
                "learning_rate", (0.001, 0.1), log=True, default=0.008237525103357958
            ),
        ]
        return hyperparameters, [], []


class XGBoostRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for the XGBoost regressor to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the XGBoostRegressorWrapper.

        Parameters
        ----------
        init_params : dict or None
            Initialization parameters for the XGBoost regressor (backward compatibility).
        **kwargs : Any
            Initialization parameters for the XGBoost regressor.
        """
        super().__init__(XGBRegressor, init_params=init_params, **kwargs)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for XGBoost regressor."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Constant("booster", "gbtree"),
            Constant("n_estimators", 2000),
            Integer("max_depth", (1, 11), log=False, default=8),
            Integer("min_child_weight", (1, 100), log=True, default=39),
            Float(
                "colsample_bytree", (0.0, 1.0), log=False, default=0.2545374925231651
            ),
            Float(
                "colsample_bylevel", (0.0, 1.0), log=False, default=0.6909224923784677
            ),
            Float("lambda", (0.001, 1000), log=True, default=31.393252465064943),
            Float("alpha", (0.001, 1000), log=True, default=0.24167936088332426),
            Float(
                "learning_rate", (0.001, 0.1), log=True, default=0.008237525103357958
            ),
        ]
        return hyperparameters, [], []


class XGBoostRankerWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for the XGBoost ranker to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_ranker"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the XGBoostRankerWrapper.

        Parameters
        ----------
        init_params : dict or None
            Initialization parameters for the XGBoost ranker (backward compatibility).
        **kwargs : Any
            Initialization parameters for the XGBoost ranker.
        """
        super().__init__(XGBRanker, init_params=init_params, **kwargs)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for XGBoost ranker."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Constant("booster", "gbtree"),
            Integer("max_depth", (1, 20), log=False, default=13),
            Integer("min_child_weight", (1, 100), log=True, default=39),
            Float(
                "colsample_bytree", (0.0, 1.0), log=False, default=0.2545374925231651
            ),
            Float(
                "colsample_bylevel", (0.0, 1.0), log=False, default=0.6909224923784677
            ),
            Float("lambda", (0.001, 1000), log=True, default=31.393252465064943),
            Float("alpha", (0.001, 1000), log=True, default=0.24167936088332426),
            Float(
                "learning_rate", (0.001, 0.1), log=True, default=0.008237525103357958
            ),
        ]
        return hyperparameters, [], []
