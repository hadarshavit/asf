try:
    from ConfigSpace import ConfigurationSpace, Float, Integer, EqualsCondition
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from sklearn.neural_network import MLPClassifier, MLPRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper

from typing import Optional, Dict, Any

from functools import partial


class MLPClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the MLPClassifier from scikit-learn, providing additional functionality
    for configuration space and parameter handling.
    """

    PREFIX = "mlp_classifier"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MLPClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initial parameters for the MLPClassifier.
        """
        super().__init__(MLPClassifier, init_params or {})

    def fit(
        self, X: Any, Y: Any, sample_weight: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights. Not supported for MLPClassifier.
        kwargs : dict
            Additional arguments for the fit method.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPClassifier"
        )
        self.model_class.fit(X, Y, **kwargs)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the MLP Classifier.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the MLP Classifier parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{MLPClassifierWrapper.PREFIX}"
            else:
                prefix = MLPClassifierWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="MLP Classifier")

            depth = Integer(f"{prefix}:depth", (1, 3), default=3, log=False)

            width = Integer(f"{prefix}:width", (16, 1024), default=64, log=True)

            batch_size = Integer(
                f"{prefix}:batch_size",
                (256, 1024),
                default=256,
                log=True,
            )  # MODIFIED from HPOBENCH

            alpha = Float(
                f"{prefix}:alpha",
                (10**-8, 1),
                default=10**-3,
                log=True,
            )

            learning_rate_init = Float(
                f"{prefix}:learning_rate_init",
                (10**-5, 1),
                default=10**-3,
                log=True,
            )

            params = [depth, width, batch_size, alpha, learning_rate_init]
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
            configuration: ConfigurationSpace, pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create an MLPClassifierWrapper instance from a configuration.

            Parameters
            ----------
            configuration : ConfigurationSpace
                The configuration containing the parameters.
            additional_params : dict, optional
                Additional parameters to override the default configuration.

            Returns
            -------
            partial
                A partial function to create an MLPClassifierWrapper instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{MLPClassifierWrapper.PREFIX}"
            else:
                prefix = MLPClassifierWrapper.PREFIX

            hidden_layers = [configuration[f"{prefix}:width"]] * configuration[
                f"{prefix}:depth"
            ]

            if "activation" not in kwargs:
                kwargs["activation"] = "relu"
            if "solver" not in kwargs:
                kwargs["solver"] = "adam"

            mlp_params = {
                "hidden_layer_sizes": tuple(hidden_layers),
                "batch_size": configuration[f"{prefix}:batch_size"],
                "alpha": configuration[f"{prefix}:alpha"],
                "learning_rate_init": configuration[f"{prefix}:learning_rate_init"],
                **kwargs,
            }

            return partial(MLPClassifierWrapper, init_params=mlp_params)


class MLPRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the MLPRegressor from scikit-learn, providing additional functionality
    for configuration space and parameter handling.
    """

    PREFIX = "mlp_regressor"

    def __init__(self, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MLPRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Initial parameters for the MLPRegressor.
        """
        super().__init__(MLPRegressor, init_params or {})

    def fit(
        self, X: Any, Y: Any, sample_weight: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights. Not supported for MLPRegressor.
        kwargs : dict
            Additional arguments for the fit method.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPRegressor"
        )
        self.model_class.fit(X, Y, **kwargs)

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the MLP Regressor.

            Parameters
            ----------
            cs : ConfigurationSpace, optional
                The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

            Returns
            -------
            ConfigurationSpace
                The configuration space with the MLP Regressor parameters.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{MLPRegressorWrapper.PREFIX}"
            else:
                prefix = MLPRegressorWrapper.PREFIX

            if cs is None:
                cs = ConfigurationSpace(name="MLP Regressor")

            depth = Integer(f"{prefix}:depth", (1, 3), default=3, log=False)

            width = Integer(f"{prefix}:width", (16, 1024), default=64, log=True)

            batch_size = Integer(
                f"{prefix}:batch_size",
                (256, 1024),
                default=256,
                log=True,
            )

            alpha = Float(
                f"{prefix}:alpha",
                (10**-8, 1),
                default=10**-3,
                log=True,
            )

            learning_rate_init = Float(
                f"{prefix}:learning_rate_init",
                (10**-5, 1),
                default=10**-3,
                log=True,
            )

            params = [depth, width, batch_size, alpha, learning_rate_init]
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
            configuration: ConfigurationSpace, pre_prefix: str = "", **kwargs
        ) -> partial:
            """
            Create an MLPRegressorWrapper instance from a configuration.

            Parameters
            ----------
            configuration : ConfigurationSpace
                The configuration containing the parameters.
            additional_params : dict, optional
                Additional parameters to override the default configuration.

            Returns
            -------
            partial
                A partial function to create an MLPRegressorWrapper instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{MLPRegressorWrapper.PREFIX}"
            else:
                prefix = MLPRegressorWrapper.PREFIX

            hidden_layers = [configuration[f"{prefix}:width"]] * configuration[
                f"{prefix}:depth"
            ]

            if "activation" not in kwargs:
                kwargs["activation"] = "relu"
            if "solver" not in kwargs:
                kwargs["solver"] = "adam"

            mlp_params = {
                "hidden_layer_sizes": tuple(hidden_layers),
                "batch_size": configuration[f"{prefix}:batch_size"],
                "alpha": configuration[f"{prefix}:alpha"],
                "learning_rate_init": configuration[f"{prefix}:learning_rate_init"],
                **kwargs,
            }

            return partial(MLPRegressorWrapper, init_params=mlp_params)
