"""
Empirical Performance Model (EPM) based on Random Forest.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor

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


class EPMRandomForest(ForestRegressor, AbstractPredictor, ConfigurableMixin):
    """
    Implementation of Random Forest as an Empirical Performance Model (EPM).

    This model follows the approach described in the paper:
    "Algorithm runtime prediction: Methods & evaluation" by Hutter, Xu, Hoos, and Leyton-Brown (2014).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    log : bool, default=False
        Whether to apply logarithmic transformation to the tree values.
    return_var : bool, default=False
        Whether to compute variance across trees.
    criterion : str, default="squared_error"
        The function to measure the quality of a split.
    splitter : str, default="random"
        The strategy used to choose the split at each node.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.
    max_features : float, default=1.0
        The number of features to consider when looking for the best split.
    max_leaf_nodes : int or None, default=None
        Grow trees with max_leaf_nodes in best-first fashion.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity.
    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel.
    random_state : int or None, default=None
        Controls the randomness of the estimator.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit.
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples : int, float or None, default=None
        The number of samples to draw from X to train each base estimator.
    monotonic_cst : np.ndarray or None, default=None
        Constraints for monotonicity of features.
    """

    PREFIX: str = "epm_random_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        log: bool = False,
        return_var: bool = False,
        criterion: str = "squared_error",
        splitter: str = "random",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,
        monotonic_cst: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            DecisionTreeRegressor(),
            n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.splitter = splitter
        self.log = log
        self.return_var = return_var

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for EPMRandomForest.

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
        return partial(EPMRandomForest, **config)

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

        self.trainX = X
        self.trainY = Y
        if self.log:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = Y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(
        self, X: np.ndarray, **kwargs: Any
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Predicted means, or a tuple of (means, variances) if return_var is True.
        """
        preds = []
        for tree in self.estimators_:
            preds.append(tree.predict(X))
        preds_arr = np.array(preds).T

        means = preds_arr.mean(axis=1)
        vars_arr = preds_arr.var(axis=1)

        if self.return_var:
            return means, vars_arr
        else:
            return means

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
    def load(cls, file_path: str) -> EPMRandomForest:
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        EPMRandomForest
            The loaded model.
        """
        return joblib.load(file_path)
