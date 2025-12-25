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
from functools import partial
import numpy as np

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

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the XGBoostClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost classifier.
        """
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it using pip install asf-lib[xgb]."
            )
        super().__init__(XGBClassifier, init_params or {})

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
        if Y.dtype == bool:
            self.bool_labels = True
        else:
            self.bool_labels = False

        self.model_class.fit(X, Y, sample_weight=sample_weight, **kwargs)  # type: ignore[attr-defined]

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
        if self.bool_labels:
            return self.model_class.predict(X, **kwargs).astype(bool)  # type: ignore[attr-defined]
        return self.model_class.predict(X, **kwargs)  # type: ignore[attr-defined]

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

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        xgb_params = clean_config.copy()
        xgb_params.update(kwargs)

        return partial(XGBoostClassifierWrapper, init_params=xgb_params)


class XGBoostRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for the XGBoost regressor to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the XGBoostRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost regressor.
        """
        super().__init__(XGBRegressor, init_params or {})

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

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        xgb_params = clean_config.copy()
        xgb_params.update(kwargs)

        return partial(XGBoostRegressorWrapper, init_params=xgb_params)


class XGBoostRankerWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for the XGBoost ranker to integrate with the ASF framework.
    """

    PREFIX: str = "xgb_ranker"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the XGBoostRankerWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initialization parameters for the XGBoost ranker.
        """
        super().__init__(XGBRanker, init_params or {})

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

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        xgb_params = clean_config.copy()
        xgb_params.update(kwargs)

        return partial(XGBoostRankerWrapper, init_params=xgb_params)
