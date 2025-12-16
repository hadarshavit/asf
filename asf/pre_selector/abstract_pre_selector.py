import pandas as pd
import numpy as np

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Configuration,
        UniformIntegerHyperparameter,
        EqualsCondition,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractPreSelector:
    """
    Abstract class for pre-selectors.
    """

    def __init__(self, n_algorithms: int | None = None):
        """
        Initialize the pre-selector with the given configuration.

        Args:
            config (dict): Configuration for the pre-selector.
        """
        self.n_algorithms = n_algorithms

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Fit the pre-selector to the performance data and transform it.
        Args:
            performance (pd.DataFrame | np.ndarray): Performance data to fit and transform.
        Returns:
            pd.DataFrame | np.ndarray: Transformed performance data.
        """
        raise NotImplementedError(
            "fit_transform method must be implemented in subclasses."
        )

    @staticmethod
    def get_configuration_space(
        cs: "ConfigurationSpace | None" = None,
        cs_transform: dict | None = None,
        parent_param: "None" = None,
        parent_value: str | None = None,
        n_algorithms_max: int | None = None,
        **kwargs,
    ) -> "tuple[ConfigurationSpace, dict]":
        """
        Get the configuration space for the algorithm pre-selector.

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
        n_algorithms_max : int or None, optional
            Maximum number of algorithms that can be pre-selected.
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
            If the method is not implemented in a subclass (for pre-selectors without configuration).
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        # Default implementation for pre-selectors without configuration space
        # Subclasses can override this to add their own hyperparameters
        if cs is None:
            cs = ConfigurationSpace()
        if cs_transform is None:
            cs_transform = {}

        # Add n_algorithms hyperparameter if needed
        if parent_param is not None and parent_value is not None:
            # Number of algorithms to pre-select
            if n_algorithms_max is not None:
                n_algos_param = UniformIntegerHyperparameter(
                    name=f"{parent_value}:n_algorithms",
                    lower=2,
                    upper=n_algorithms_max,
                )
                cs.add(n_algos_param)

                # Make it conditional on the pre-selector being selected
                condition = EqualsCondition(n_algos_param, parent_param, parent_value)
                cs.add(condition)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: "Configuration | dict",
        cs_transform: dict,
        maximize: bool = False,
        pre_selector_name: str | None = None,
        **kwargs,
    ) -> "AbstractPreSelector":
        """
        Create a pre-selector instance from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration object or dictionary.
        cs_transform : dict
            The transformation dictionary for the configuration space.
        maximize : bool, optional
            Whether to maximize the metric. Defaults to False.
        pre_selector_name : str or None, optional
            Name of the pre-selector (used to find n_algorithms in configuration). If None,
            will use the class name. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        AbstractPreSelector
            The pre-selector instance.

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

        # Extract n_algorithms from configuration if provided
        n_algorithms = None
        if pre_selector_name is not None:
            n_algos_key = f"{pre_selector_name}:n_algorithms"
            if n_algos_key in configuration:
                n_algorithms = configuration[n_algos_key]

        return n_algorithms
