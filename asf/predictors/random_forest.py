from asf.predictors.sklearn_wrapper import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Integer,
        Float,
        Categorical,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Any


class RandomForestClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the RandomForestClassifier from scikit-learn, providing
    additional functionality for configuration space management.
    """

    PREFIX = "rf_classifier"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the RandomForestClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the RandomForestClassifier.
        """
        super().__init__(RandomForestClassifier, init_params)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the Random Forest Classifier.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the Random Forest Classifier parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RandomForestClassifierWrapper.PREFIX}:"
            else:
                prefix = RandomForestClassifierWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="RandomForest")

            n_estimators = Integer(
                f"{prefix}n_estimators",
                (16, 128),
                log=True,
                default=116,
            )
            min_samples_split = Integer(
                f"{prefix}min_samples_split",
                (2, 20),
                log=False,
                default=2,
            )
            min_samples_leaf = Integer(
                f"{prefix}min_samples_leaf",
                (1, 20),
                log=False,
                default=2,
            )
            max_features = Float(
                f"{prefix}max_features",
                (0.1, 1.0),
                log=False,
                default=0.17055852159745608,
            )
            bootstrap = Categorical(
                f"{prefix}bootstrap",
                items=[True, False],
                default=False,
            )
            max_depth = Integer(
                f"{prefix}max_depth",
                (1, 100),
                log=False,
                default=13,
            )

            params = [
                n_estimators,
                min_samples_split,
                min_samples_leaf,
                max_features,
                bootstrap,
                max_depth
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
            configuration: dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create a RandomForestClassifierWrapper instance from a configuration.

            Parameters
            ----------
            configuration : dict
                A dictionary containing the configuration parameters.
            additional_params : dict, optional
                Additional parameters to override or extend the configuration.

            Returns
            -------
            partial
                A partial function to create a RandomForestClassifierWrapper instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RandomForestClassifierWrapper.PREFIX}:"
            else:
                prefix = RandomForestClassifierWrapper.PREFIX

            rf_params = {
                "n_estimators": configuration[f"{prefix}n_estimators"],
                "min_samples_split": configuration[f"{prefix}min_samples_split"],
                "min_samples_leaf": configuration[f"{prefix}min_samples_leaf"],
                "max_features": configuration[f"{prefix}max_features"],
                "bootstrap": configuration[f"{prefix}bootstrap"],
                "max_depth": configuration[f"{prefix}max_depth"]
                **kwargs,
            }

            return partial(RandomForestClassifierWrapper, init_params=rf_params)


class RandomForestRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the RandomForestRegressor from scikit-learn, providing
    additional functionality for configuration space management.
    """

    PREFIX = "rf_regressor"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the RandomForestRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the RandomForestRegressor.
        """
        super().__init__(RandomForestRegressor, init_params)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the Random Forest Regressor.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the Random Forest Regressor parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RandomForestRegressorWrapper.PREFIX}:"
            else:
                prefix = RandomForestRegressorWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="RandomForestRegressor")

            n_estimators = Integer(
                f"{prefix}n_estimators",
                (16, 128),
                log=True,
                default=116,
            )
            min_samples_split = Integer(
                f"{prefix}min_samples_split",
                (2, 20),
                log=False,
                default=2,
            )
            min_samples_leaf = Integer(
                f"{prefix}min_samples_leaf",
                (1, 20),
                log=False,
                default=2,
            )
            max_features = Float(
                f"{prefix}max_features",
                (0.1, 1.0),
                log=False,
                default=0.17055852159745608,
            )
            bootstrap = Categorical(
                f"{prefix}bootstrap",
                items=[True, False],
                default=False,
            )
            max_depth = Integer(
                f"{prefix}max_depth",
                (1, 100),
                log=False,
                default=13,
            )
            
            params = [
                n_estimators,
                min_samples_split,
                min_samples_leaf,
                max_features,
                bootstrap,
                max_depth
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
            configuration: dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create a RandomForestRegressorWrapper instance from a configuration.

            Parameters
            ----------
            configuration : dict
                A dictionary containing the configuration parameters.
            additional_params : dict, optional
                Additional parameters to override or extend the configuration.

            Returns
            -------
            partial
                A partial function to create a RandomForestRegressorWrapper instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{RandomForestRegressorWrapper.PREFIX}:"
            else:
                prefix = RandomForestRegressorWrapper.PREFIX

            rf_params = {
                "n_estimators": configuration[f"{prefix}n_estimators"],
                "min_samples_split": configuration[f"{prefix}min_samples_split"],
                "min_samples_leaf": configuration[f"{prefix}min_samples_leaf"],
                "max_features": configuration[f"{prefix}max_features"],
                "bootstrap": configuration[f"{prefix}bootstrap"],
                "max_depth": configuration[f"{prefix}max_depth"]
                **kwargs,
            }

            return partial(RandomForestRegressorWrapper, init_params=rf_params)
