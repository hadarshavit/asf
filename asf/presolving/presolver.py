import pandas as pd
from abc import abstractmethod

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Configuration,
        UniformFloatHyperparameter,
        EqualsCondition,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractPresolver:
    def __init__(
        self,
        budget: float,
        maximize: bool = False,
    ):
        self.budget = budget
        self.maximize = maximize

    @abstractmethod
    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self) -> list[tuple[str, float]]:
        pass

    @staticmethod
    def get_configuration_space(
        cs: "ConfigurationSpace | None" = None,
        cs_transform: dict | None = None,
        parent_param: None = None,
        parent_value: str | None = None,
        total_budget: float | None = None,
        **kwargs,
    ) -> "tuple[ConfigurationSpace, dict]":
        """
        Get the configuration space for the presolver.

        Parameters
        ----------
        cs : ConfigurationSpace or None, optional
            The configuration space to use. If None, a new one will be created.
        cs_transform : dict or None, optional
            A dictionary for transforming configuration space values.
        parent_param : Hyperparameter or None, optional
            Parent parameter for conditional configuration.
        parent_value : str or None, optional
            Value of parent parameter that activates these parameters.
        total_budget : float or None, optional
            Total budget available (used to set upper bound for presolver_budget).
        **kwargs : dict
            Additional keyword arguments for configuration space creation.

        Returns
        -------
        tuple[ConfigurationSpace, dict]
            The configuration space and transformation dictionary.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        NotImplementedError
            If the method is not implemented in a subclass (for presolvers without configuration).
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        # Default implementation for presolvers without configuration space
        # Subclasses can override this to add their own hyperparameters
        if cs is None:
            cs = ConfigurationSpace()
        if cs_transform is None:
            cs_transform = {}

        # Add presolver_budget hyperparameter if parent_param is specified
        if parent_param is not None and parent_value is not None:
            # Budget for presolver (fraction of total budget)
            if total_budget is not None:
                # Ensure upper > lower (upper must be strictly greater than 1)
                upper_budget = max(1.1, 0.1 * total_budget)
                presolver_budget_param = UniformFloatHyperparameter(
                    name=f"{parent_value}:presolver_budget",
                    lower=1,
                    upper=upper_budget,
                    default_value=min(10, upper_budget),
                    log=True,
                )
                cs.add(presolver_budget_param)

                # Make it conditional on the presolver being selected
                condition = EqualsCondition(
                    presolver_budget_param, parent_param, parent_value
                )
                cs.add(condition)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: "Configuration | dict",
        cs_transform: dict,
        budget: float | None = None,
        maximize: bool = False,
        presolver_name: str | None = None,
        **kwargs,
    ) -> "AbstractPresolver":
        """
        Create a presolver instance from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration object or dictionary.
        cs_transform : dict
            The transformation dictionary for the configuration space.
        budget : float or None, optional
            Budget for the presolver. If None, will try to extract from configuration
            using the presolver_budget parameter. Defaults to None.
        maximize : bool, optional
            Whether to maximize the metric. Defaults to False.
        presolver_name : str or None, optional
            Name of the presolver (used to find budget in configuration). If None,
            will use the class name. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        AbstractPresolver
            The presolver instance.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        # Extract budget from configuration if not provided
        if budget is None and presolver_name is not None:
            budget_key = f"{presolver_name}:presolver_budget"
            if budget_key in configuration:
                budget = configuration[budget_key]

        raise NotImplementedError(
            "get_from_configuration() is not implemented for this presolver"
        )
