from __future__ import annotations

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        Constant,
        EqualsCondition,
        Float,
        InCondition,
        Integer,
        AndConjunction,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from functools import partial
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

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the SVMClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVC model.
        """
        super().__init__(SVC, init_params)

    @classmethod
    def get_configuration_space(
        cls,
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
        **kwargs,
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

        prefix = cls.PREFIX
        max_iter = Constant(
            f"{prefix}:max_iter",
            20000,
        )
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
            default=1.0,
        )
        shrinking = Categorical(
            f"{prefix}:shrinking",
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
        cur_conds = [gamma_cond, degree_cond]

        if parent_param is not None:
            simple_params = [p for p in params if p not in (gamma, degree)]
            simple_equals = [
                EqualsCondition(child=param, parent=parent_param, value=parent_value)
                for param in simple_params
            ]

            gamma_eq = EqualsCondition(
                child=gamma, parent=parent_param, value=parent_value
            )
            degree_eq = EqualsCondition(
                child=degree, parent=parent_param, value=parent_value
            )

            # AndConjunction expects variadic condition arguments, not a list
            gamma_and = AndConjunction(gamma_eq, gamma_cond)
            degree_and = AndConjunction(degree_eq, degree_cond)

            conditions = simple_equals + [gamma_and, degree_and]

            cs.add(params + conditions)
        else:
            conditions = []
            cs.add(params + conditions + cur_conds)

        return cs

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        # We need to manually handle 'kernel' logic because ConfigSpace doesn't
        # automatically filter inactive conditionals from the dictionary
        # if the input dictionary has them (which depends on how SMAC behaves).
        # Assuming clean_config has valid active parameters.

        # SklearnWrapper expects init_params dict, not kwargs
        svm_params = clean_config.copy()
        svm_params.update(kwargs)

        return partial(SVMClassifierWrapper, init_params=svm_params)


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

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the SVMRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            Dictionary of parameters to initialize the SVR model.
        """
        super().__init__(SVR, init_params)

    @classmethod
    def get_configuration_space(
        cls,
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
        **kwargs,
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

        prefix = cls.PREFIX

        if cs is None:
            cs = ConfigurationSpace(name="SVM Regressor")

        max_iter = Constant(
            f"{prefix}:max_iter",
            20000,
        )
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
        epsilon = Float(
            f"{prefix}:epsilon",
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
        cur_conds = [gamma_cond, degree_cond]

        if parent_param is not None:
            simple_params = [p for p in params if p not in (gamma, degree)]
            simple_equals = [
                EqualsCondition(child=param, parent=parent_param, value=parent_value)
                for param in simple_params
            ]

            gamma_eq = EqualsCondition(
                child=gamma, parent=parent_param, value=parent_value
            )
            degree_eq = EqualsCondition(
                child=degree, parent=parent_param, value=parent_value
            )

            # AndConjunction expects variadic condition arguments, not a list
            gamma_and = AndConjunction(gamma_eq, gamma_cond)
            degree_and = AndConjunction(degree_eq, degree_cond)

            conditions = simple_equals + [gamma_and, degree_and]

            cs.add(params + conditions)
        else:
            conditions = []
            cs.add(params + conditions + cur_conds)

        return cs

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        svm_params = clean_config.copy()
        svm_params.update(kwargs)

        return partial(SVMRegressorWrapper, init_params=svm_params)
