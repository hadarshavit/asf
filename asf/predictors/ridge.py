from asf.predictors.sklearn_wrapper import SklearnWrapper
from typing import Any
from sklearn.linear_model import Ridge, RidgeClassifier

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Float,
        Categorical,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from functools import partial


class RidgeRegressorWrapper(SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.Ridge
    """

    PREFIX = "ridge_regressor"

    def __init__(self, init_params: dict[str, Any] = {}):
        super().__init__(Ridge, init_params)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
        ) -> ConfigurationSpace:
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RidgeRegressorWrapper.PREFIX}:"
            else:
                prefix = RidgeRegressorWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="RidgeRegressor")

            alpha = Float(
                f"{prefix}alpha",
                (1e-6, 100.0),
                log=True,
                default=1.0,
            )
            fit_intercept = Categorical(
                f"{prefix}fit_intercept",
                [True, False],
                default=True,
            )
            solver = Categorical(
                f"{prefix}solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                default="auto",
            )

            params = [alpha, fit_intercept, solver]
            if parent_param is not None:
                from ConfigSpace.conditions import EqualsCondition

                conditions = [
                    EqualsCondition(
                        child=param, parent=parent_param, value=parent_value
                    )
                    for param in params
                ]
            else:
                conditions = []

            cs.add(params + conditions)
            return cs

        @staticmethod
        def get_from_configuration(
            configuration: dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RidgeRegressorWrapper.PREFIX}:"
            else:
                prefix = RidgeRegressorWrapper.PREFIX

            params = {
                "alpha": configuration[f"{prefix}alpha"],
                "fit_intercept": configuration[f"{prefix}fit_intercept"],
                "solver": configuration[f"{prefix}solver"],
                **kwargs,
            }
            return partial(RidgeRegressorWrapper, init_params=params)


class RidgeClassifierWrapper(SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.RidgeClassifier
    """

    PREFIX = "ridge_classifier"

    def __init__(self, init_params: dict[str, Any] = {}):
        super().__init__(RidgeClassifier, init_params)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
        ) -> ConfigurationSpace:
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RidgeClassifierWrapper.PREFIX}:"
            else:
                prefix = RidgeClassifierWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="RidgeClassifier")

            alpha = Float(
                f"{prefix}alpha",
                (1e-6, 100.0),
                log=True,
                default=1.0,
            )
            solver = Categorical(
                f"{prefix}solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                default="auto",
            )

            params = [alpha, solver]
            if parent_param is not None:
                from ConfigSpace.conditions import EqualsCondition

                conditions = [
                    EqualsCondition(
                        child=param, parent=parent_param, value=parent_value
                    )
                    for param in params
                ]
            else:
                conditions = []

            cs.add(params + conditions)
            return cs

        @staticmethod
        def get_from_configuration(
            configuration: dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RidgeClassifierWrapper.PREFIX}:"
            else:
                prefix = RidgeClassifierWrapper.PREFIX

            params = {
                "alpha": configuration[f"{prefix}alpha"],
                "solver": configuration[f"{prefix}solver"],
                **kwargs,
            }
            return partial(RidgeClassifierWrapper, init_params=params)
