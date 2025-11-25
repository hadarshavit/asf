from typing import Optional, Dict, List, Any, Union, Type

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors.ridge import RidgeRegressorWrapper
import inspect


class CosineSelector(AbstractSelector):
    """
    Cosine similarity based selector using a shared latent space learned from
    the performance (interaction) matrix Y.

    Parameters
    ----------
    normalize_features : bool
        If True, standardize instance and algorithm features (per-column StandardScaler).
    shared_latent_dim : int
        Dimensionality of the shared latent space (number of factors extracted from Y).
    ridge_alpha : float
        Regularization strength for the default Ridge projection wrapper.
    svd_random_state : int
        Random seed for TruncatedSVD.
    projection_model : Optional[Type | object]
        Optional projection model class or instantiated object. If None, the default
        RidgeRegressorWrapper is used. If a wrapper class (e.g. RandomForestRegressorWrapper)
        is provided, projection_model_kwargs are passed on construction.
    projection_model_kwargs : Optional[dict]
        Keyword arguments forwarded to projection_model when it is a class.
    """

    PREFIX = "cosine"
    RETURN_TYPE = "single"

    def __init__(
        self,
        normalize_features: bool = True,
        shared_latent_dim: int = 4,
        ridge_alpha: float = 1.0,
        svd_random_state: int = 0,
        projection_model: Optional[Union[Type, Any]] = None,
        projection_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize_features = bool(normalize_features)
        self.shared_latent_dim = int(shared_latent_dim)
        self._ridge_alpha = float(ridge_alpha)
        self._svd_random_state = int(svd_random_state)
        self._projection_model = projection_model
        self._projection_model_kwargs = projection_model_kwargs or {}

        self._svd: Optional[TruncatedSVD] = None
        self._proj: Optional[RidgeRegressorWrapper] = None
        self._alg_feats: Optional[pd.DataFrame] = None
        self._alg_matrix: Optional[np.ndarray] = None
        self._scaler_inst: Optional[StandardScaler] = None
        self._scaler_alg: Optional[StandardScaler] = None
        self.algorithms: List[str] = []

    def _normalize_rows(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        L2-normalize rows of X with numerical stability.

        Parameters
        ----------
        X : np.ndarray
            2D array whose rows should be normalized.
        eps : float
            Small value to avoid division by zero.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms

    def _norm(self, s: str) -> str:
        """
        Minimal string normalization used for matching algorithm identifiers.
        """
        s = str(s).lower().strip()
        return re.sub(r"[\W_]+", "", s)

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the cosine selector.

        Parameters
        ----------
        features : pd.DataFrame
            Instance feature matrix (rows = instances).
        performance : pd.DataFrame
            Performance matrix (rows = instances, columns = algorithms).
        """
        alg_df = getattr(self, "algorithm_features", None)
        if alg_df is None or not isinstance(alg_df, pd.DataFrame):
            raise ValueError(
                "Set selector.algorithm_features (pd.DataFrame indexed by algorithm names) before fit()"
            )

        self.algorithms = list(performance.columns)
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

        # Interaction-matrix SVD (learn shared latent space from performance Y)
        # Align performance rows with features and columns with algorithms
        Y = performance.loc[features.index, self.algorithms].to_numpy(dtype=float)
        # simple NaN handling: replace per-column NaN with column mean
        col_mean = np.nanmean(Y, axis=0)
        inds = np.where(np.isnan(Y))
        if inds[0].size:
            Y[inds] = np.take(col_mean, inds[1])

        n_comp = min(self.shared_latent_dim, min(Y.shape[0] - 1, Y.shape[1]))
        n_comp = max(1, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=self._svd_random_state)
        inst_emb = svd.fit_transform(Y)  # shape (n_instances, n_comp)
        alg_emb = svd.components_.T  # shape (n_algorithms, n_comp)
        self._svd = svd

        # fit projection model X_inst -> inst_emb (default: Ridge wrapper)
        if self._projection_model is None:
            proj = RidgeRegressorWrapper(init_params={"alpha": self._ridge_alpha})
        elif isinstance(self._projection_model, type):
            # Some wrapper classes (e.g. RandomForestRegressorWrapper) expect a single
            # 'init_params' dict argument. Detect that and adapt kwargs automatically.
            kwargs = dict(self._projection_model_kwargs or {})
            try:
                sig = inspect.signature(self._projection_model.__init__)
                if "init_params" in sig.parameters and "init_params" not in kwargs:
                    kwargs = {"init_params": kwargs}
            except Exception:
                # Safely ignore any exception when inspecting the __init__ signature,
                # as not all classes may have a standard signature or may not be inspectable.
                pass
            proj = self._projection_model(**kwargs)
        else:
            proj = self._projection_model

        proj.fit(X_inst, inst_emb)
        self._proj = proj

        self._alg_matrix = self._normalize_rows(alg_emb)

    def _predict(self, features: pd.DataFrame) -> Dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each query instance.

        Parameters
        ----------
        features : pd.DataFrame
            Query instance features (rows = instances).

        Returns
        -------
        Dict[str, list[tuple[str, float]]]
            Mapping from instance id to a single recommendation (algorithm name, score_or_budget).
        """
        if self._alg_matrix is None:
            raise ValueError("fit() must be called before predict()")

        Xq = features.fillna(0.0).to_numpy(dtype=float)
        if self.normalize_features and self._scaler_inst is not None:
            Xq = self._scaler_inst.transform(Xq)

        # project queries into the same instance-embedding space via learned mapper
        if self._proj is None:
            raise RuntimeError(
                "internal projection model missing; fit() must produce a mapper"
            )
        Xq_emb = self._proj.predict(Xq)
        Xq_n = self._normalize_rows(Xq_emb)

        sims = Xq_n.dot(self._alg_matrix.T)

        out: Dict[str, list[tuple[str, float]]] = {}
        budget = getattr(self, "budget", None)

        for i, inst in enumerate(features.index):
            row = sims[i]
            j = int(np.argmax(row))
            chosen = self.algorithms[j]
            score = float(budget) if budget is not None else float(row[j])
            out[inst] = [(chosen, score)]
        return out
