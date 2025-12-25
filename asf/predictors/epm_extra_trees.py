from sklearn.ensemble._forest import ExtraTreesRegressor
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


class EPMRandomForest(ExtraTreesRegressor, AbstractPredictor, ConfigurableMixin):
    """
    Implementation of random forest as done in the paper
    "Algorithm runtime prediction: Methods & evaluation" by Hutter, Xu, Hoos, and Leyton-Brown (2014).

    Attributes
    ----------
    log : bool
        Whether to apply logarithmic transformation to tree values during training.

    Methods
    -------
    fit(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None
        Fit the model to the data.
    predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]
        Predict using the model and return means and variances.
    save(file_path: str) -> None
        Save the model to a file.
    load(file_path: str) -> EPMRandomForest
        Load the model from a file.
    """

    def __init__(  # TODO check hparams
        self,
        *,
        log: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the EPMRandomForest model.

        Parameters
        ----------
        log : bool, optional
            Whether to apply logarithmic transformation to tree values during training, by default False.
        **kwargs : dict
            Additional keyword arguments passed to the ExtraTreesRegressor.
        """
        super().__init__(**kwargs)
        self.log = log

    PREFIX = "epm_extra_trees"

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for EPMRandomForest (ExtraTrees)."""
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
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        sample_weight : np.ndarray | None, optional
            Sample weights, by default None. Currently not supported.

        Raises
        ------
        AssertionError
            If sample weights are provided.
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
            - means : np.ndarray of shape (n_samples, 1)
                Predicted mean values.
            - vars : np.ndarray of shape (n_samples, 1)
                Predicted variances.
        """
        preds = []
        for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
            preds.append(tree.predict(X))
        preds = np.array(preds).T

        means = preds.mean(axis=1)
        vars = preds.var(axis=1)

        return means.reshape(-1, 1), vars.reshape(-1, 1)

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
