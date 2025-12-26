from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from asf.selectors.abstract_selector import AbstractSelector


class AbstractEPMBasedSelector(AbstractSelector):
    """
    Abstract base class for selectors that utilize an Empirical Performance Model (EPM).

    This class provides functionality to initialize with EPM parameters,
    save the selector to a file, and load it back.

    Attributes
    ----------
    epm_kwargs : dict[str, Any]
        Keyword arguments for the EPM.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the AbstractEPMBasedSelector.
        """
        self.epm_kwargs = {}
        # Pull em_ prefixed arguments into epm_kwargs
        to_del = []
        for k, v in kwargs.items():
            if k.startswith("em_"):
                self.epm_kwargs[k] = v
                to_del.append(k)

        # Pull other EPM related arguments if they exist
        for k in ["use_log10"]:
            if k in kwargs:
                self.epm_kwargs[k] = kwargs[k]
                to_del.append(k)

        for k in to_del:
            del kwargs[k]

        super().__init__(**kwargs)

    def save(self, path: str | Path) -> None:
        """
        Save the selector instance to the specified file path.

        Parameters
        ----------
        path : str or Path
            The file path to save the selector.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> AbstractEPMBasedSelector:
        """
                Load a selector instance from the specified file path.

                Parameters
                ----------
                path : str or Path
                    The file path to load the selector from.

                Returns
        -------
                AbstractEPMBasedSelector
                    The loaded selector instance.
        """
        return joblib.load(path)
