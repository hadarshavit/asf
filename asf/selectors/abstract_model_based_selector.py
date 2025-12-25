from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable

import joblib
from sklearn.base import ClassifierMixin, RegressorMixin

from asf.predictors import SklearnWrapper
from asf.predictors.abstract_predictor import AbstractPredictor
from asf.selectors.abstract_selector import AbstractSelector


class AbstractModelBasedSelector(AbstractSelector):
    """
    Abstract base class for selectors that utilize a machine learning model.

    This class provides functionality to initialize with a model class,
    save the selector to a file, and load it back.

    Attributes
    ----------
    model_class : Callable
        A callable that represents the model class to be used.
    """

    def __init__(
        self,
        model_class: type[AbstractPredictor] | Callable[..., Any],
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AbstractModelBasedSelector.

        Parameters
        ----------
        model_class : type[AbstractPredictor] or Callable
            The model class or a callable that returns a model instance.
            If a scikit-learn compatible class is provided, it's wrapped with SklearnWrapper.
        **kwargs : Any
            Additional keyword arguments passed to the parent class initializer.
        """
        super().__init__(**kwargs)

        if isinstance(model_class, type) and issubclass(
            model_class, (ClassifierMixin, RegressorMixin)
        ):
            self.model_class: Callable[..., Any] = partial(SklearnWrapper, model_class)
        else:
            self.model_class = model_class

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
    def load(cls, path: str | Path) -> AbstractModelBasedSelector:
        """
                Load a selector instance from the specified file path.

                Parameters
                ----------
                path : str or Path
                    The file path to load the selector from.

                Returns
        -------
                AbstractModelBasedSelector
                    The loaded selector instance.
        """
        return joblib.load(path)
