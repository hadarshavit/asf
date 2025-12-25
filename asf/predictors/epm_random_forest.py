from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from asf.predictors import AbstractPredictor


from asf.utils.configurable import ConfigurableMixin
from functools import partial
from typing import Any

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
        Float,
        Categorical,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class EPMRandomForest(ForestRegressor, AbstractPredictor, ConfigurableMixin):
    """
    Implementation of random forest as described in the paper
    "Algorithm runtime prediction: Methods & evaluation" by Hutter, Xu, Hoos, and Leyton-Brown (2014).

    This class extends `ForestRegressor`, `AbstractPredictor`, and `ConfigurableMixin` to provide
    a random forest implementation with additional functionality for runtime prediction.

    Parameters
    ----------
    n_estimators : int, optional
        The number of trees in the forest. Default is 100.
    log : bool, optional
        Whether to apply logarithmic transformation to the tree values. Default is False.
    cross_trees_variance : bool, optional
        Whether to compute variance across trees. Default is False.
    criterion : str, optional
        The function to measure the quality of a split. Default is "squared_error".
    splitter : str, optional
        The strategy used to choose the split at each node. Default is "random".
    max_depth : int, optional
        The maximum depth of the tree. Default is None.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node. Default is 2.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node. Default is 1.
    min_weight_fraction_leaf : float, optional
        The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
    max_features : float, optional
        The number of features to consider when looking for the best split. Default is 1.0.
    max_leaf_nodes : int, optional
        Grow trees with max_leaf_nodes in best-first fashion. Default is None.
    min_impurity_decrease : float, optional
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
    bootstrap : bool, optional
        Whether bootstrap samples are used when building trees. Default is False.
    oob_score : bool, optional
        Whether to use out-of-bag samples to estimate the generalization score. Default is False.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is None.
    random_state : int, optional
        Controls the randomness of the estimator. Default is None.
    verbose : int, optional
        Controls the verbosity when fitting and predicting. Default is 0.
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble. Default is False.
    ccp_alpha : float, optional
        Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.
    max_samples : int or float, optional
        If bootstrap is True, the number of samples to draw from X to train each base estimator. Default is None.
    monotonic_cst : array-like, optional
        Constraints for monotonicity of features. Default is None.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        log: bool = False,
        return_var: bool = False,
        criterion: str = "squared_error",
        splitter: str = "random",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: int | float = None,
        monotonic_cst: np.ndarray = None,
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

    PREFIX = "epm_random_forest"

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for EPMRandomForest."""
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
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(EPMRandomForest, **config)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        sample_weight : np.ndarray, optional
            Sample weights. Default is None.

        Raises
        ------
        AssertionError
            If sample weights are provided, as they are not supported.
        """
        assert sample_weight is None, "Sample weights are not supported"
        super().fit(X=X, y=y, sample_weight=sample_weight)

        self.trainX = X
        self.trainY = y
        if self.log:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on of shape (n_samples, n_features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Predicted means of shape (n_samples, 1).
            - Predicted variances of shape (n_samples, 1).
        """
        preds = []
        for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
            preds.append(tree.predict(X))
        preds = np.array(preds).T

        means = preds.mean(axis=1)
        vars = preds.var(axis=1)

        if self.return_var:
            return means, vars
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
        import joblib

        joblib.dump(self, file_path)

    def load(self, file_path: str) -> "EPMRandomForest":
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
        import joblib

        return joblib.load(file_path)
