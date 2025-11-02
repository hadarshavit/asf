from __future__ import annotations

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Constant,
        Float,
        Integer,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from typing import Any, Callable
from functools import partial
import numpy as np

try:
    from xgboost import XGBRegressor, XGBClassifier, XGBRanker

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from asf.predictors.sklearn_wrapper import SklearnWrapper


class XGBoostClassifierWrapper(SklearnWrapper):
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
        sample_weight: np.ndarray = None,
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

        self.model_class.fit(X, Y, sample_weight=sample_weight, **kwargs)

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
            return self.model_class.predict(X, **kwargs).astype(bool)
        return self.model_class.predict(X, **kwargs)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost classifier.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if cs is None:
            cs = ConfigurationSpace(name="XGBoost")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostClassifierWrapper.PREFIX}"
        else:
            prefix = XGBoostClassifierWrapper.PREFIX

        booster = Constant(f"{prefix}:booster", "gbtree")
        max_depth = Integer(
            f"{prefix}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{prefix}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{prefix}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{prefix}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{prefix}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{prefix}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )

        params = [
            booster,
            max_depth,
            min_child_weight,
            colsample_bytree,
            colsample_bylevel,
            lambda_param,
            alpha,
            learning_rate,
        ]

        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        **kwargs: Any,
    ) -> Callable[..., "XGBoostClassifierWrapper"]:
        """
        Create an XGBoostClassifierWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.
        additional_params : dict, optional
            Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostClassifierWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostClassifierWrapper.PREFIX}"
        else:
            prefix = XGBoostClassifierWrapper.PREFIX

        xgb_params = {
            "booster": configuration[f"{prefix}:booster"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_child_weight": configuration[f"{prefix}:min_child_weight"],
            "colsample_bytree": configuration[f"{prefix}:colsample_bytree"],
            "colsample_bylevel": configuration[f"{prefix}:colsample_bylevel"],
            "lambda": configuration[f"{prefix}:lambda"],
            "alpha": configuration[f"{prefix}:alpha"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            **kwargs,
        }

        return partial(XGBoostClassifierWrapper, init_params=xgb_params)


class XGBoostRegressorWrapper(SklearnWrapper):
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
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if cs is None:
            cs = ConfigurationSpace(name="XGBoostRegressor")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostRegressorWrapper.PREFIX}"
        else:
            prefix = XGBoostRegressorWrapper.PREFIX

        booster = Constant(f"{prefix}:booster", "gbtree")
        max_depth = Integer(
            f"{prefix}:max_depth",
            (1, 15),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{prefix}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{prefix}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{prefix}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{prefix}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{prefix}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )

        params = [
            booster,
            max_depth,
            min_child_weight,
            colsample_bytree,
            colsample_bylevel,
            lambda_param,
            alpha,
            learning_rate,
        ]
        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        **kwargs,
    ) -> Callable[..., "XGBoostRegressorWrapper"]:
        """
        Create an XGBoostRegressorWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.
        additional_params : dict, optional
            Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostRegressorWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostRegressorWrapper.PREFIX}"
        else:
            prefix = XGBoostRegressorWrapper.PREFIX

        xgb_params = {
            "booster": configuration[f"{prefix}:booster"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_child_weight": configuration[f"{prefix}:min_child_weight"],
            "colsample_bytree": configuration[f"{prefix}:colsample_bytree"],
            "colsample_bylevel": configuration[f"{prefix}:colsample_bylevel"],
            "lambda": configuration[f"{prefix}:lambda"],
            "alpha": configuration[f"{prefix}:alpha"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            **kwargs,
        }

        return partial(XGBoostRegressorWrapper, init_params=xgb_params)


class XGBoostRankerWrapper(SklearnWrapper):
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
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost ranker.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space with the XGBoost parameters.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if cs is None:
            cs = ConfigurationSpace(name="XGBoostRanker")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostRankerWrapper.PREFIX}"
        else:
            prefix = XGBoostRankerWrapper.PREFIX

        booster = Constant(f"{prefix}:booster", "gbtree")
        max_depth = Integer(
            f"{prefix}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{prefix}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{prefix}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{prefix}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{prefix}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{prefix}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )
        params = [
            booster,
            max_depth,
            min_child_weight,
            colsample_bytree,
            colsample_bylevel,
            lambda_param,
            alpha,
            learning_rate,
        ]

        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)
        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        **kwargs,
    ) -> Callable[..., "XGBoostRankerWrapper"]:
        """
        Create an XGBoostRankerWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration dictionary.

        Returns
        -------
        Callable[..., XGBoostRankerWrapper]
            A callable that initializes the wrapper with the given configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBoostRankerWrapper.PREFIX}"
        else:
            prefix = XGBoostRankerWrapper.PREFIX

        xgb_params = {
            "booster": configuration[f"{prefix}:booster"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_child_weight": configuration[f"{prefix}:min_child_weight"],
            "colsample_bytree": configuration[f"{prefix}:colsample_bytree"],
            "colsample_bylevel": configuration[f"{prefix}:colsample_bylevel"],
            "lambda": configuration[f"{prefix}:lambda"],
            "alpha": configuration[f"{prefix}:alpha"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            **kwargs,
        }

        return partial(XGBoostRankerWrapper, init_params=xgb_params)
