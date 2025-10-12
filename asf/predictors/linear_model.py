try:
    from ConfigSpace import ConfigurationSpace, Float, EqualsCondition
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge

from asf.predictors.sklearn_wrapper import SklearnWrapper

from functools import partial
from typing import Optional, Dict, Any


class LinearClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the SGDClassifier from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_classifier"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinearClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDClassifier.
        """
        super().__init__(SGDClassifier, init_params or {})

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the Linear Classifier.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the Linear Classifier parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{LinearClassifierWrapper.PREFIX}"
            else:
                prefix = LinearClassifierWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="Linear Classifier")

            alpha = Float(
                f"{prefix}:alpha",
                (1e-5, 1),
                log=True,
                default=1e-3,
            )
            eta0 = Float(
                f"{prefix}:eta0",
                (1e-5, 1),
                log=True,
                default=1e-2,
            )

            params = [
                alpha,
                eta0,
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
            configuration: Dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create a partial function to initialize LinearClassifierWrapper with parameters from a configuration.

            Parameters
            ----------
            configuration : dict
                A dictionary containing the configuration parameters.
            additional_params : dict, optional
                Additional parameters to include in the initialization.

            Returns
            -------
            partial
                A partial function to initialize LinearClassifierWrapper.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{LinearClassifierWrapper.PREFIX}"
            else:
                prefix = LinearClassifierWrapper.PREFIX

            linear_classifier_params = {
                "alpha": configuration[f"{prefix}:alpha"],
                "eta0": configuration[f"{prefix}:eta0"],
                **kwargs,
            }

            return partial(
                LinearClassifierWrapper, init_params=linear_classifier_params
            )


class LinearRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the SGDRegressor from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LinearRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDRegressor.
        """
        super().__init__(SGDRegressor, init_params or {})

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the Linear Regressor.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the Linear Regressor parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{LinearRegressorWrapper.PREFIX}"
            else:
                prefix = LinearRegressorWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="Linear Regressor")

            alpha = Float(
                f"{prefix}:alpha",
                (1e-5, 1),
                log=True,
                default=1e-3,
            )
            eta0 = Float(
                f"{prefix}:eta0",
                (1e-5, 1),
                log=True,
                default=1e-2,
            )

            params = [alpha, eta0]

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
            configuration: Dict[str, Any], pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create a partial function to initialize LinearRegressorWrapper with parameters from a configuration.

            Parameters
            ----------
            configuration : dict
                A dictionary containing the configuration parameters.
            additional_params : dict, optional
                Additional parameters to include in the initialization.

            Returns
            -------
            partial
                A partial function to initialize LinearRegressorWrapper.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{LinearRegressorWrapper.PREFIX}"
            else:
                prefix = LinearRegressorWrapper.PREFIX

            linear_regressor_params = {
                "alpha": configuration[f"{prefix}:alpha"],
                "eta0": configuration[f"{prefix}:eta0"],
                **kwargs,
            }

            return partial(LinearRegressorWrapper, init_params=linear_regressor_params)


class RidgeRegressorWrapper(SklearnWrapper):
    """Wrapper around scikit-learn's Ridge regressor for ASF predictors."""

    PREFIX = "ridge_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        super().__init__(Ridge, init_params or {})
