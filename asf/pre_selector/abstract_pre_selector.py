import pandas as pd
import numpy as np


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
