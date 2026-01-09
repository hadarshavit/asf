"""
Abstract base class for algorithm pre-selectors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from ConfigSpace import (
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
        UniformIntegerHyperparameter,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractPreSelector:
    """
    Abstract base class for algorithm pre-selectors.

    Parameters
    ----------
    n_algorithms : int or None, default=None
        The number of algorithms to pre-select.
    """

    def __init__(self, n_algorithms: int | None = None) -> None:
        """
        Initialize the pre-selector.

        Parameters
        ----------
        n_algorithms : int or None, default=None
            The number of algorithms to pre-select.
        """
        self.n_algorithms = n_algorithms

    def fit_transform(
        self,
        performance: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """
        Fit the pre-selector to the performance data and transform it.

        Parameters
        ----------
        performance : pd.DataFrame or np.ndarray
            Performance data to fit and transform.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Transformed performance data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "fit_transform method must be implemented in subclasses."
        )

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, Any] | None = None,
        parent_param: Any | None = None,
        parent_value: Any | None = None,
        n_algorithms_max: int | None = None,
        **kwargs: Any,
    ) -> tuple[ConfigurationSpace, dict[str, Any]]:
        """
        Get the configuration space for the algorithm pre-selector.

        Parameters
        ----------
        cs : ConfigurationSpace or None, default=None
            The configuration space to use. If None, a new one will be created.
        cs_transform : dict[str, Any] or None, default=None
            A dictionary for transforming configuration space values.
        parent_param : Any or None, default=None
            Parent parameter for conditional configuration.
        parent_value : Any or None, default=None
            Value of parent parameter that activates these parameters.
        n_algorithms_max : int or None, default=None
            Maximum number of algorithms that can be pre-selected.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[ConfigurationSpace, dict[str, Any]]
            The configuration space and transformation dictionary.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if cs is None:
            cs = ConfigurationSpace()
        if cs_transform is None:
            cs_transform = {}

        if parent_param is not None and parent_value is not None:
            if n_algorithms_max is not None:
                n_algos_param = UniformIntegerHyperparameter(
                    name=f"{parent_value}:n_algorithms",
                    lower=2,
                    upper=n_algorithms_max,
                )
                cs.add(n_algos_param)

                condition = EqualsCondition(n_algos_param, parent_param, parent_value)
                cs.add(condition)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any],
        maximize: bool = False,
        pre_selector_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a pre-selector instance from a configuration.

        In the abstract class, this method extracts `n_algorithms` from the configuration.

        Parameters
        ----------
        configuration : Configuration or dict[str, Any]
            The configuration object or dictionary.
        cs_transform : dict[str, Any]
            The transformation dictionary.
        maximize : bool, default=False
            Whether to maximize the metric.
        pre_selector_name : str or None, default=None
            Name of the pre-selector.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The extracted n_algorithms or a pre-selector instance in subclasses.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        n_algorithms = None
        if pre_selector_name is not None:
            n_algos_key = f"{pre_selector_name}:n_algorithms"
            if n_algos_key in configuration:
                n_algorithms = configuration[n_algos_key]

        return n_algorithms
