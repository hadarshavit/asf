"""
Empirical Performance Model (EPM) based on Extra Trees.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble._forest import ExtraTreesRegressor

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (
        Categorical,
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class EPMExtraTrees(ExtraTreesRegressor, AbstractPredictor, ConfigurableMixin):
    """
    Implementation of Extra Trees as an Empirical Performance Model (EPM).

    This model follows the approach described in the paper:
    "Algorithm runtime prediction: Methods & evaluation" by Hutter, Xu, Hoos, and Leyton-Brown (2014).

    Parameters
    ----------
    log : bool, default=False
        Whether to apply logarithmic transformation to tree values during training.
    **kwargs : Any
        Additional keyword arguments passed to the ExtraTreesRegressor.
    """

    PREFIX: str = "epm_extra_trees"

    def __init__(
        self,
        *,
        log: bool = False,
        **kwargs: Any,
    ) -> None:
        # Separate args for EPMExtraTrees and ExtraTreesRegressor
        self.log = log
        # Pass remaining kwargs to ExtraTreesRegressor
        super().__init__(**kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for EPMExtraTrees.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            (hyperparameters, conditions, forbiddens)
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("n_estimators", (16, 128), log=True, default=100),
            Integer("min_samples_split", (2, 20), log=False, default=2),
            Integer("min_samples_leaf", (1, 20), log=False, default=1),
            Float("max_features", (0.1, 1.0), log=False, default=1.0),
            Categorical("bootstrap", items=[True, False], default=False),
            Categorical("log", items=[True, False], default=False),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(EPMExtraTrees, **config)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        sample_weight : np.ndarray or None, default=None
            Sample weights. Currently not supported.

        Raises
        ------
        AssertionError
            If sample weights are provided.
        """
        assert sample_weight is None, "Sample weights are not supported"
        super().fit(X=X, y=Y, sample_weight=sample_weight)

        if self.log:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = Y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(self, X: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the model and return means and variances.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on of shape (n_samples, n_features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (means, variances) where means is shape (n_samples, 1)
            and variances is shape (n_samples, 1).
        """
        preds = []
        for tree in self.estimators_:
            preds.append(tree.predict(X))
        preds_arr = np.array(preds).T

        means = preds_arr.mean(axis=1)
        vars_arr = preds_arr.var(axis=1)

        return means.reshape(-1, 1), vars_arr.reshape(-1, 1)

    def save(self, file_path: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> EPMExtraTrees:
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        EPMExtraTrees
            The loaded model.
        """
        return joblib.load(file_path)
