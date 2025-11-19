from typing import Optional, Dict, List

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from asf.selectors.abstract_selector import AbstractSelector


class CosineSelector(AbstractSelector):
    PREFIX = "cosine"
    RETURN_TYPE = "single"

    def __init__(self, normalize_features: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_features = bool(normalize_features)
        self._alg_feats: Optional[pd.DataFrame] = None
        self._alg_matrix: Optional[np.ndarray] = None
        self._scaler_inst: Optional[StandardScaler] = None
        self._scaler_alg: Optional[StandardScaler] = None
        self.algorithms: List[str] = []

    def _normalize_rows(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms

    def _norm(self, s: str) -> str:
        s = str(s).lower().strip()
        return re.sub(r"[\W_]+", "", s)

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        # algorithm features must be provided on selector instance
        alg_df = getattr(self, "algorithm_features", None)
        if alg_df is None or not isinstance(alg_df, pd.DataFrame):
            raise ValueError("Set selector.algorithm_features (pd.DataFrame indexed by algorithm names) before fit()")

        self.algorithms = list(performance.columns)

        # build mapping by normalized names and align order
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
            raise ValueError(f"Algorithm feature rows do not match performance columns. Missing: {missing}. Available sample: {avail}")

        alg_df = alg_df.loc[mapped].copy()
        alg_df.index = [str(a) for a in self.algorithms]
        alg_df = alg_df.select_dtypes(include=[np.number]).astype(float)
        self._alg_feats = alg_df

        X_inst = features.fillna(0.0).to_numpy(dtype=float)
        X_alg = alg_df.fillna(0.0).to_numpy(dtype=float)

        if self.normalize_features:
            self._scaler_inst = StandardScaler().fit(X_inst)
            X_inst = self._scaler_inst.transform(X_inst)
            self._scaler_alg = StandardScaler().fit(X_alg)
            X_alg = self._scaler_alg.transform(X_alg)

        self._inst_matrix = self._normalize_rows(X_inst)
        self._alg_matrix = self._normalize_rows(X_alg)

    def _predict(self, features: pd.DataFrame) -> Dict[str, list[tuple[str, float]]]:
        if self._alg_matrix is None:
            raise ValueError("fit() must be called before predict()")

        Xq = features.fillna(0.0).to_numpy(dtype=float)
        if self.normalize_features and self._scaler_inst is not None:
            Xq = self._scaler_inst.transform(Xq)
        Xq_n = self._normalize_rows(Xq)

        sims = Xq_n.dot(self._alg_matrix.T)

        out: Dict[str, list[tuple[str, float]]] = {}
        for i, inst in enumerate(features.index):
            j = int(np.argmax(sims[i]))
            chosen = self.algorithms[j]
            out[inst] = [(chosen, self.budget)]
        return out