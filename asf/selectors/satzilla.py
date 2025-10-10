from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import pandas as pd

from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors.ridge import RidgeRegressorWrapper


class SATzillaSelector(AbstractSelector):
    """
    SATzilla-like selector implementing the requested data preparation and model steps.

    Key behaviors implemented here (per spec):
      - Impute censored runs (timeouts) using cutoff * impute_factor before log10 transform.
      - Target transformation: base-10 log of runtimes.
      - Feature normalization (store training means/stds) and apply at predict time.
      - Feature expansion: include original normalized features and all pairwise
        products (j <= l).
      - Train one Ridge model per algorithm on expanded features. Default model
        is RidgeRegressorWrapper but a model_factory/class can be provided.
      - Prediction returns instance -> [(algorithm_name, predicted_runtime_float)].
    """

    def __init__(
        self,
        model_class: Optional[type] = None,
        model_factory: Optional[Callable[..., object]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        cutoff: Optional[float] = None,
        impute_factor: float = 1.1,
        use_log10: bool = True,
        presolvers: Optional[List[Tuple[str, float]]] = None,
        backup_solver: Optional[str] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize SATzillaSelector.

        Parameters:
          model_class (Optional[type]): Class to instantiate per-algorithm models.
              If None, defaults to RidgeRegressorWrapper.
          model_factory (Optional[Callable]): Callable returning a fresh model instance.
              If provided, it takes precedence over model_class.
          model_kwargs (Optional[dict]): Keyword arguments forwarded to model constructor/factory.
          cutoff (Optional[float]): Runtime cutoff; values >= cutoff are treated as timeouts and imputed.
          impute_factor (float): Factor multiplied with cutoff to impute censored runs (default 1.1).
          use_log10 (bool): If True, apply base-10 log to targets during training and invert at prediction.
          presolvers (Optional[List[Tuple[str, float]]]): Optional list of (algorithm_name, time) presolvers.
          backup_solver (Optional[str]): Fallback solver name used when no finite prediction is available.
          random_state (Optional[int]): Random seed passed to models/factories where applicable.
        """
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs or {}
        self.cutoff = cutoff
        self.impute_factor = float(impute_factor)
        self.use_log10 = use_log10
        self.presolvers = presolvers or []
        self.backup_solver = backup_solver
        self.random_state = random_state

        self.models: Dict[str, Optional[object]] = {}
        self.algorithms: List[str] = []

        # default model
        if self.model_factory is None and self.model_class is None:
            self.model_class = RidgeRegressorWrapper

        # training normalization parameters (populated in _fit)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def _make_model(self):
        if self.model_factory is not None:
            return self.model_factory(**self.model_kwargs)
        if self.model_class is not None:
            try:
                return self.model_class(init_params=self.model_kwargs)
            except TypeError:
                return self.model_class(**self.model_kwargs)
        raise RuntimeError(
            "No model_factory or model_class available for SATzillaSelector"
        )

    @staticmethod
    def _expand_features(X_norm: np.ndarray) -> np.ndarray:
        """
        Expand features to include original normalized features and pairwise products (j <= l).
        X_norm: (n_samples, k)
        Returns: (n_samples, k + k*(k+1)/2)
        """
        n, k = X_norm.shape
        # number of pairwise products with j<=l
        m = k + (k * (k + 1)) // 2
        X_exp = np.empty((n, m), dtype=float)
        # first k columns: original normalized features
        X_exp[:, :k] = X_norm
        col = k
        for j in range(k):
            for l in range(j, k):
                X_exp[:, col] = X_norm[:, j] * X_norm[:, l]
                col += 1
        return X_exp

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Impute censored timeouts and apply log10 transform (if enabled).
        y: raw runtimes vector
        """
        y = y.astype(float).copy()
        if self.cutoff is not None:
            # treat values >= cutoff as censored/timeouts
            censored_mask = y >= self.cutoff
            if censored_mask.any():
                y[censored_mask] = self.cutoff * self.impute_factor
        # ensure strictly positive before log
        y = np.maximum(y, 1e-8)
        if self.use_log10:
            return np.log10(y)
        else:
            return np.log(y)  # fallback to natural log if not using base-10

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Train one Ridge model per algorithm on expanded features.
        """
        self.algorithms = list(performance.columns)
        X = features.values.astype(float)
        # compute normalization stats
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0, ddof=0)
        # avoid division by zero
        self._feature_stds[self._feature_stds == 0.0] = 1.0
        X_norm = (X - self._feature_means) / self._feature_stds
        X_exp = self._expand_features(X_norm)

        self.models = {}
        for algo in self.algorithms:
            y_raw = performance[algo].values
            mask = ~np.isnan(y_raw)
            if mask.sum() == 0:
                self.models[algo] = None
                continue
            X_train = X_exp[mask]
            y_train = self._prepare_targets(y_raw[mask])
            model = self._make_model()
            model.fit(X_train, y_train)
            self.models[algo] = model

    def _predict(
        self, features: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[Tuple[Optional[str], float]]]:
        """
        Predict per-algorithm runtimes for each instance and pick the minimum.
        Returns mapping instance_name -> [(algorithm_name_or_None, predicted_runtime)].
        """
        if features is None:
            raise ValueError("Features must be provided for prediction.")
        if self._feature_means is None or self._feature_stds is None:
            raise RuntimeError("SATzillaSelector must be fitted before prediction.")

        X = features.values.astype(float)
        # normalize using training statistics
        X_norm = (X - self._feature_means) / self._feature_stds
        X_exp = self._expand_features(X_norm)
        n = X_exp.shape[0]
        algos = self.algorithms
        preds = np.full((n, len(algos)), np.inf, dtype=float)

        for j, algo in enumerate(algos):
            model = self.models.get(algo)
            if model is None:
                continue
            y_pred = np.asarray(model.predict(X_exp)).reshape(-1)
            # inverse of log10
            if self.use_log10:
                y_pred = np.power(10.0, y_pred)
            else:
                y_pred = np.exp(y_pred)
            if y_pred.shape[0] != n:
                raise RuntimeError(
                    f"Model for {algo} returned unexpected number of predictions"
                )
            preds[:, j] = y_pred

        best_idx = np.argmin(preds, axis=1)
        results: Dict[str, List[Tuple[Optional[str], float]]] = {}
        for i, inst in enumerate(features.index):
            j = int(best_idx[i])
            algo = algos[j] if np.isfinite(preds[i, j]) else None
            pred_time = float(preds[i, j]) if algo is not None else float("inf")
            if algo is None and self.backup_solver is not None:
                algo = self.backup_solver
                pred_time = float("inf")
            results[inst] = [(algo, pred_time)]
        return results
