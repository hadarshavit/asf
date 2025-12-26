from __future__ import annotations

try:
    from ConfigSpace import (
        Categorical,
        Constant,
        Float,
        InCondition,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from typing import Any

from sklearn.svm import SVC, SVR

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin


class SVMClassifierWrapper(ConfigurableMixin, SklearnWrapper):
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

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the SVMClassifierWrapper.

        Parameters
        ----------
        init_params : dict or None
            Parameters for the SVC model (backward compatibility).
        **kwargs : Any
            Parameters for the SVC model.
        """
        super().__init__(SVC, init_params=init_params, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define the configuration space for the SVM classifier.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        # ConfigurableMixin handles prefixes
        max_iter = Constant("max_iter", 20000)
        kernel = Categorical(
            "kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer("degree", (1, 128), log=True, default=1)
        coef0 = Float(
            "coef0",
            (-0.5, 0.5),
            log=False,
            default=0.49070634552851977,
        )
        tol = Float(
            "tol",
            (1e-4, 1e-2),
            log=True,
            default=0.0002154969698207585,
        )
        gamma = Categorical(
            "gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float(
            "C",
            (1.0, 20),
            log=True,
            default=1.0,
        )
        shrinking = Categorical(
            "shrinking",
            items=[True, False],
            default=True,
        )

        params = [kernel, degree, coef0, tol, gamma, C, shrinking, max_iter]

        gamma_cond = InCondition(
            child=gamma,
            parent=kernel,
            values=["rbf", "poly", "sigmoid"],
        )
        degree_cond = InCondition(
            child=degree,
            parent=kernel,
            values=["poly"],
        )
        conditions = [gamma_cond, degree_cond]

        return params, conditions, []


class SVMRegressorWrapper(ConfigurableMixin, SklearnWrapper):
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

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the SVMRegressorWrapper.

        Parameters
        ----------
        init_params : dict or None
            Parameters for the SVR model (backward compatibility).
        **kwargs : Any
            Parameters for the SVR model.
        """
        super().__init__(SVR, init_params=init_params, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define the configuration space for the SVM regressor.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        # ConfigurableMixin handles prefixes
        max_iter = Constant("max_iter", 20000)
        kernel = Categorical(
            "kernel",
            items=["linear", "rbf", "poly", "sigmoid"],
            default="rbf",
        )
        degree = Integer("degree", (1, 128), log=True, default=1)
        coef0 = Float(
            "coef0",
            (-0.5, 0.5),
            log=False,
            default=0.0,
        )
        tol = Float(
            "tol",
            (1e-4, 1e-2),
            log=True,
            default=0.001,
        )
        gamma = Categorical(
            "gamma",
            items=["scale", "auto"],
            default="scale",
        )
        C = Float("C", (1.0, 20), log=True, default=1.0)
        shrinking = Categorical(
            "shrinking",
            items=[True, False],
            default=True,
        )
        epsilon = Float(
            "epsilon",
            (0.01, 0.99),
            log=True,
            default=0.0251,
        )

        params = [kernel, degree, coef0, tol, gamma, C, shrinking, epsilon, max_iter]

        gamma_cond = InCondition(
            child=gamma,
            parent=kernel,
            values=["rbf", "poly", "sigmoid"],
        )
        degree_cond = InCondition(
            child=degree,
            parent=kernel,
            values=["poly"],
        )
        conditions = [gamma_cond, degree_cond]

        return params, conditions, []
