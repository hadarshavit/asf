from __future__ import annotations

import inspect
import re
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from asf.predictors.ridge import RidgeRegressorWrapper
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class CosineSelector(ConfigurableMixin, AbstractSelector):
    """
    Cosine similarity based selector using a shared latent space.

    Attributes
    ----------
    normalize_features : bool
        If True, standardize instance and algorithm features.
    shared_latent_dim : int
        Dimensionality of the shared latent space.
    _ridge_alpha : float
        Regularization strength for the default Ridge projection wrapper.
    _svd_random_state : int
        Random seed for TruncatedSVD.
    _projection_model : type or Callable or Any or None
        Optional projection model class or instantiated object.
    _projection_model_kwargs : dict
        Keyword arguments forwarded to projection_model.
    _svd : TruncatedSVD or None
        Truncated SVD model.
    _proj : Any or None
        Projection model instance.
    _alg_feats : pd.DataFrame or None
        Processed algorithm features.
    _alg_matrix : np.ndarray or None
        Algorithm embeddings in latent space.
    _scaler_inst : StandardScaler or None
        StandardScaler for instance features.
    _scaler_alg : StandardScaler or None
        StandardScaler for algorithm features.
    """

    PREFIX = "cosine"
    RETURN_TYPE = "single"

    def __init__(
        self,
        normalize_features: bool = True,
        shared_latent_dim: int = 4,
        ridge_alpha: float = 1.0,
        svd_random_state: int = 0,
        projection_model: type | Callable[..., Any] | Any | None = None,
        projection_model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CosineSelector.

        Parameters
        ----------
        normalize_features : bool, default=True
            If True, standardize instance and algorithm features.
        shared_latent_dim : int, default=4
            Dimensionality of the shared latent space.
        ridge_alpha : float, default=1.0
            Regularization strength for the default Ridge projection wrapper.
        svd_random_state : int, default=0
            Random seed for TruncatedSVD.
        projection_model : type or Callable or Any, default=None
            Optional projection model class or instantiated object.
        projection_model_kwargs : dict, default=None
            Keyword arguments forwarded to projection_model.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.normalize_features = bool(normalize_features)
        self.shared_latent_dim = int(shared_latent_dim)
        self._ridge_alpha = float(ridge_alpha)
        self._svd_random_state = int(svd_random_state)
        self._projection_model = projection_model
        self._projection_model_kwargs = projection_model_kwargs or {}

        self._svd: TruncatedSVD | None = None
        self._proj: Any | None = None
        self._alg_feats: pd.DataFrame | None = None
        self._alg_matrix: np.ndarray | None = None
        self._scaler_inst: StandardScaler | None = None
        self._scaler_alg: StandardScaler | None = None

    def _normalize_rows(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        L2-normalize rows of X with numerical stability.

        Parameters
        ----------
        X : np.ndarray
            2D array whose rows should be normalized.
        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        np.ndarray
            L2-normalized array.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms

    def _norm(self, s: str) -> str:
        """
        Minimal string normalization for matching algorithm identifiers.

        Parameters
        ----------
        s : str
            String to normalize.

        Returns
        -------
        str
            Normalized string.
        """
        s = str(s).lower().strip()
        return re.sub(r"[\W_]+", "", s)

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the cosine selector.

        Parameters
        ----------
        features : pd.DataFrame
            Instance feature matrix (rows = instances).
        performance : pd.DataFrame
            Performance matrix (rows = instances, columns = algorithms).
        **kwargs : Any
            Additional keyword arguments.
        """
        alg_df = getattr(self, "algorithm_features", None)
        if alg_df is None or not isinstance(alg_df, pd.DataFrame):
            raise ValueError(
                "Set selector.algorithm_features (pd.DataFrame indexed by algorithm names) before fit()"
            )

        self.algorithms = [str(a) for a in performance.columns]
        alg_df.index = alg_df.index.astype(str)
        norm_to_orig = {self._norm(n): n for n in alg_df.index}
        mapped = []
        missing = []
        for a in self.algorithms:
            na = self._norm(a)
            orig = norm_to_orig.get(na)
            if orig is None:
                missing.append(a)
            else:
                mapped.append(orig)
        if missing:
            avail = list(alg_df.index)[:10]
            raise ValueError(
                f"Algorithm feature rows do not match performance columns. Missing: {missing}. Available sample: {avail}"
            )

        alg_df = alg_df.loc[mapped].copy()
        alg_df.index = [str(a) for a in self.algorithms]
        alg_df = alg_df.select_dtypes(include=[np.number]).astype(float)
        self._alg_feats = alg_df

        X_inst = features.fillna(0.0).to_numpy(dtype=float)

        if self.normalize_features:
            self._scaler_inst = StandardScaler().fit(X_inst)
            X_inst = self._scaler_inst.transform(X_inst)

        Y = performance.loc[features.index, self.algorithms].to_numpy(dtype=float)
        col_mean = np.nanmean(Y, axis=0)
        inds = np.where(np.isnan(Y))
        if inds[0].size:
            Y[inds] = np.take(col_mean, inds[1])

        n_comp = min(self.shared_latent_dim, min(Y.shape[0] - 1, Y.shape[1]))
        n_comp = max(1, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=self._svd_random_state)
        inst_emb = svd.fit_transform(Y)
        alg_emb = svd.components_.T
        self._svd = svd

        if self._projection_model is None:
            proj = RidgeRegressorWrapper(init_params={"alpha": self._ridge_alpha})
        elif isinstance(self._projection_model, type):
            init_kwargs = dict(self._projection_model_kwargs or {})
            try:
                sig = inspect.signature(self._projection_model.__init__)
                if "init_params" in sig.parameters and "init_params" not in init_kwargs:
                    init_kwargs = {"init_params": init_kwargs}
            except Exception:
                pass
            proj = self._projection_model(**init_kwargs)
        elif isinstance(self._projection_model, partial):
            proj = self._projection_model()
        else:
            proj = self._projection_model

        proj.fit(X_inst, inst_emb)  # type: ignore[attr-defined]
        self._proj = proj

        self._alg_matrix = self._normalize_rows(alg_emb)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each query instance.
        """
        if self._alg_matrix is None:
            raise ValueError("fit() must be called before predict()")

        if features is None:
            raise ValueError("CosineSelector requires features for prediction.")
        if self._proj is None:
            raise RuntimeError(
                "internal projection model missing; fit() must produce a mapper"
            )

        Xq = features.fillna(0.0).to_numpy(dtype=float)
        if self.normalize_features and self._scaler_inst is not None:
            Xq = self._scaler_inst.transform(Xq)

        Xq_emb = self._proj.predict(Xq)
        Xq_n = self._normalize_rows(Xq_emb)

        sims = Xq_n.dot(self._alg_matrix.T)

        out: dict[str, list[tuple[str, float]]] = {}
        for i, inst in enumerate(features.index):
            row = sims[i]
            j = int(np.argmax(row))
            chosen = str(self.algorithms[j])
            out[str(inst)] = [(chosen, float(self.budget or 0))]
        return out

    @staticmethod
    def _define_hyperparameters(
        projection_model: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for CosineSelector.

        Parameters
        ----------
        projection_model : list[type] or None, default=None
            List of projection model classes.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if projection_model is None:
            projection_model = [RidgeRegressorWrapper]

        projection_model_param = ClassChoice(
            name="projection_model",
            choices=projection_model,
            default=projection_model[0],
        )

        normalize_features_param = Categorical(
            name="normalize_features",
            items=[True, False],
            default=True,
        )

        shared_latent_dim_param = Integer(
            name="shared_latent_dim",
            bounds=(1, 20),
            default=4,
        )

        ridge_alpha_param = Float(
            name="ridge_alpha",
            bounds=(1e-3, 100.0),
            log=True,
            default=1.0,
        )

        params = [
            projection_model_param,
            normalize_features_param,
            shared_latent_dim_param,
            ridge_alpha_param,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[CosineSelector]:
        """
                Create a partial function from a clean configuration.

                Parameters
                ----------
                clean_config : dict
                    The clean configuration.
                **kwargs : Any
                    Additional keyword arguments.

                Returns
        -------
                partial
                    Partial function for CosineSelector.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(CosineSelector, **config)
