from __future__ import annotations

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from sklearn.svm import SVC, SVR
from functools import partial
from typing import Any

from asf.predictors.sklearn_wrapper import SklearnWrapper


class SVMClassifierWrapper(SklearnWrapper):
    """
    A wrapper for the Scikit-learn SVC (Support Vector Classifier) model.
    Provides methods to define a configuration space and create an instance
    of the classifier from a configuration.

    Attributes
    ----------
    PREFIX : str
        Prefix used for parameter names in the configuration space.
    """

    PREFIX = "svm_classifier"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the SVMClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVC model.
        """
        super().__init__(SVC, init_params)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Define the configuration space for the SVM classifier.

        Returns
        -------
        ConfigurationSpace
            The configuration space containing hyperparameters for the SVM classifier.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if cs is None:
            cs = ConfigurationSpace(name="SVM")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{SVMClassifierWrapper.PREFIX}"
        else:
            prefix = SVMClassifierWrapper.PREFIX

        kernel = Categorical(
            f"{prefix}:kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer(f"{prefix}:degree", (1, 128), log=True, default=1)
        coef0 = Float(
            f"{prefix}:coef0",
            (-0.5, 0.5),
            log=False,
            default=0.49070634552851977,
        )
        tol = Float(
            f"{prefix}:tol",
            (1e-4, 1e-2),
            log=True,
            default=0.0002154969698207585,
        )
        gamma = Categorical(
            f"{prefix}:gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float(
            f"{prefix}:C",
            (1.0, 20),
            log=True,
            default=3.2333262862494365,
        )
        shrinking = Categorical(
            f"{prefix}:shrinking",
            items=[True, False],
            default=True,
        )

        params = [kernel, degree, coef0, tol, gamma, C, shrinking]

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
        Create an SVMClassifierWrapper instance from a configuration.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the model initialization.

        Returns
        -------
        partial
            A partial function to create an SVMClassifierWrapper instance.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{SVMClassifierWrapper.PREFIX}"
        else:
            prefix = SVMClassifierWrapper.PREFIX

        svm_params = {
            "kernel": configuration[f"{prefix}:kernel"],
            "degree": configuration[f"{prefix}:degree"],
            "coef0": configuration[f"{prefix}:coef0"],
            "tol": configuration[f"{prefix}:tol"],
            "gamma": configuration[f"{prefix}:gamma"],
            "C": configuration[f"{prefix}:C"],
            "shrinking": configuration[f"{prefix}:shrinking"],
            **kwargs,
        }

        return partial(SVMClassifierWrapper, init_params=svm_params)


class SVMRegressorWrapper(SklearnWrapper):
    """
    A wrapper for the Scikit-learn SVR (Support Vector Regressor) model.
    Provides methods to define a configuration space and create an instance
    of the regressor from a configuration.

    Attributes
    ----------
    PREFIX : str
        Prefix used for parameter names in the configuration space.
    """

    PREFIX = "svm_regressor"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the SVMRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVR model.
        """
        super().__init__(SVR, init_params)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Define the configuration space for the SVM regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to add the parameters to. If None, a new
            ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space containing hyperparameters for the SVM regressor.
        """

        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{SVMRegressorWrapper.PREFIX}"
        else:
            prefix = SVMRegressorWrapper.PREFIX

        if cs is None:
            cs = ConfigurationSpace(name="SVM Regressor")

        kernel = Categorical(
            f"{prefix}:kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer(f"{prefix}:degree", (1, 128), log=True, default=1)
        coef0 = Float(
            f"{prefix}:coef0",
            (-0.5, 0.5),
            log=False,
            default=0.0,
        )
        tol = Float(
            f"{prefix}:tol",
            (1e-4, 1e-2),
            log=True,
            default=0.001,
        )
        gamma = Categorical(
            f"{prefix}:gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float(f"{prefix}:C", (1.0, 20), log=True, default=1.0)
        shrinking = Categorical(
            f"{prefix}:shrinking",
            items=[True, False],
            default=True,
        )
        params = [kernel, degree, coef0, tol, gamma, C, shrinking]
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
        Create an SVMRegressorWrapper instance from a configuration.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the configuration parameters.
        additional_params : dict, optional
            Additional parameters to include in the model initialization.

        Returns
        -------
        partial
            A partial function to create an SVMRegressorWrapper instance.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{SVMRegressorWrapper.PREFIX}"
        else:
            prefix = SVMRegressorWrapper.PREFIX

        svr_params = {
            "kernel": configuration[f"{prefix}:kernel"],
            "degree": configuration[f"{prefix}:degree"],
            "coef0": configuration[f"{prefix}:coef0"],
            "tol": configuration[f"{prefix}:tol"],
            "gamma": configuration[f"{prefix}:gamma"],
            "C": configuration[f"{prefix}:C"],
            "shrinking": configuration[f"{prefix}:shrinking"],
            **kwargs,
        }

        return partial(SVMRegressorWrapper, init_params=svr_params)
