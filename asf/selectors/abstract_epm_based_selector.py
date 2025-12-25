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

    def __init__(self, epm_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Initialize the AbstractEPMBasedSelector.

        Parameters
        ----------
        epm_kwargs : dict[str, Any] or None, default=None
            Keyword arguments for the EPM.
        **kwargs : Any
            Additional keyword arguments passed to the parent class initializer.
        """
        super().__init__(**kwargs)
        self.epm_kwargs = epm_kwargs or {}

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
