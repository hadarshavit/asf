from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm

from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors.ridge import RidgeRegressorWrapper


class SATzilla(AbstractSelector):
    """
    SATzilla-like selector using Schmee & Hahn (1979) iterative imputation
    for censored runtimes (log-scale) and per-algorithm ridge models on
    expanded features (original + pairwise products).
    -> Feature Selection is recommended.
    """

    def __init__(
        self,
        model_factory: Optional[Callable[..., object]] = RidgeRegressorWrapper,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_log10: bool = True,
        random_state: Optional[int] = None,
        em_max_iter: int = 20,
        em_tol: float = 1e-3,
        em_min_sigma: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize the SATzillaSelector.

        Args:
            model_factory: Callable returning a fresh model instance; takes precedence.
            model_kwargs: Keyword args forwarded to model constructor/factory.
            use_log10: If True, use base-10 log transform for targets.
            random_state: Random seed for reproducibility.
            em_max_iter: Max iterations for Schmee & Hahn imputation.
            em_tol: Convergence tolerance for imputed target updates.
            em_min_sigma: Minimum residual std to avoid numerical issues.
            **kwargs: Additional args passed to parent.
        """
        super().__init__(**kwargs)

        self.model_factory = model_factory
        self.model_kwargs = model_kwargs or {}
        self.use_log10 = use_log10
        self.random_state = random_state

        self.em_max_iter = int(em_max_iter)
        self.em_tol = float(em_tol)
        self.em_min_sigma = float(em_min_sigma)

        self.models: Dict[str, Optional[object]] = {}
        self.algorithms: List[str] = []

        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _make_model(self):
        """Instantiate a new model using model_factory."""
        if self.model_factory is None:
            raise RuntimeError("No model_factory available for SATzillaSelector")
        return self.model_factory(**self.model_kwargs)

    def _log_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply log transform to runtimes for numerical stability.
        Ensures values are strictly positive before taking log.
        """
        y = np.maximum(y, 1e-8)
        return np.log10(y) if self.use_log10 else np.log(y)

    def _inv_log_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse of the log transform used during training."""
        return np.power(10.0, y) if self.use_log10 else np.exp(y)

    @staticmethod
    def _expand_features(X_norm: np.ndarray) -> np.ndarray:
        """
        Expand normalized features to include original features followed by
        all pairwise products (j <= l). Returns expanded matrix.
        """
        n, k = X_norm.shape
        m = k + (k * (k + 1)) // 2
        X_exp = np.empty((n, m), dtype=float)
        X_exp[:, :k] = X_norm
        col = k
        for j in range(k):
            for m in range(j, k):
                X_exp[:, col] = X_norm[:, j] * X_norm[:, m]
                col += 1
        return X_exp

    def _schmee_hahn_impute(
        self, y_raw: np.ndarray, X_exp: np.ndarray
    ) -> Tuple[np.ndarray, Optional[object]]:
        """
        Perform Schmee & Hahn iterative imputation on the log scale.

        Args:
            y_raw: raw runtimes array (NaN for missing entries).
            X_exp: expanded feature matrix used for fitting.

        Returns:
            (y_imputed, model): y_imputed is array of log-scale targets (NaN where input was NaN).
                model is the fitted model instance trained on the final imputed targets.
        """
        mask_not_nan = ~np.isnan(y_raw)
        if not mask_not_nan.any():
            return np.full_like(y_raw, np.nan, dtype=float), None

        cutoff_val = float(self.budget)
        a = np.log10(cutoff_val) if self.use_log10 else np.log(cutoff_val)

        mask_obs = mask_not_nan & (y_raw < cutoff_val)
        mask_cens = mask_not_nan & (y_raw >= cutoff_val)

        y_imputed = np.full_like(y_raw, np.nan, dtype=float)
        if mask_obs.any():
            y_imputed[mask_obs] = self._log_transform(y_raw[mask_obs].astype(float))
        if mask_cens.any():
            y_imputed[mask_cens] = a

        fit_mask = mask_not_nan
        prev_vals = y_imputed.copy()

        model = self._make_model()

        for _ in range(self.em_max_iter):
            model.fit(X_exp[fit_mask], y_imputed[fit_mask])

            mu_all = np.asarray(model.predict(X_exp)).reshape(-1)

            if mask_obs.any():
                resid = y_imputed[mask_obs] - mu_all[mask_obs]
                sigma = np.sqrt(np.mean(resid**2))
            else:
                sigma = 0.0
            sigma = max(sigma, self.em_min_sigma)

            if mask_cens.any():
                mu_c = mu_all[mask_cens]
                z = (a - mu_c) / sigma
                sf = 1.0 - norm.cdf(z)
                sf = np.maximum(sf, 1e-12)
                expected = mu_c + sigma * (norm.pdf(z) / sf)
                y_imputed[mask_cens] = expected

            if mask_cens.any():
                delta = np.max(np.abs(y_imputed[mask_cens] - prev_vals[mask_cens]))
            else:
                delta = 0.0

            prev_vals[mask_cens] = y_imputed[mask_cens]
            if delta < self.em_tol:
                break

        return y_imputed, model

    def _fit_target_model(
        self, y_raw: np.ndarray, X_exp: np.ndarray
    ) -> Optional[object]:
        """
        Impute censored targets (Schmee & Hahn) and return the fitted model.
        """
        y_imputed, model = self._schmee_hahn_impute(y_raw.astype(float), X_exp)
        fit_mask = ~np.isnan(y_imputed)
        if not fit_mask.any():
            return None
        return model

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Normalize features using training stats and return expanded matrix.
        """
        X = features.values.astype(float)
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0, ddof=0)
        self._feature_stds[self._feature_stds == 0.0] = 1.0
        X_norm = (X - self._feature_means) / self._feature_stds
        return self._expand_features(X_norm)

    def _normalize_sat_labels(
        self, sat_arr: Optional[List[Any]], n: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Convert various booleans/strings into 'SAT'/'UNSAT' array aligned with instances.
        If sat_arr is None, returns None. If n provided, validates length.
        """
        if sat_arr is None:
            return None
        sat_np = np.asarray(sat_arr)
        if n is not None and sat_np.shape[0] != n:
            raise ValueError("sat labels length must match number of instances")
        if sat_np.dtype == bool:
            mask = sat_np
        else:
            # Convert to lowercase strings and compare to "sat"
            sat_str = np.char.lower(sat_np.astype(str))
            mask = sat_str == "sat"
        return (np.where(mask, "SAT", "UNSAT"),)

    def _predict_for_entry(
        self,
        model_entry: Union[object, Dict[str, Optional[object]]],
        X_exp: np.ndarray,
        sat_norm: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Return per-instance predicted runtimes (not log) for a single algorithm entry.
        Handles single-model and dict {'ALL','SAT','UNSAT'} entries.
        """
        n = X_exp.shape[0]
        if model_entry is None:
            return np.full(n, np.inf, dtype=float)

        if not isinstance(model_entry, dict):
            y_log = np.asarray(model_entry.predict(X_exp)).reshape(-1)
            return self._inv_log_transform(y_log)

        # dict case
        if sat_norm is not None:
            pred_j = np.full(n, np.inf, dtype=float)
            for status in ("SAT", "UNSAT"):
                model = model_entry.get(status)
                if model is None:
                    continue
                mask = sat_norm == status
                if not mask.any():
                    continue
                y_log = np.asarray(model.predict(X_exp[mask])).reshape(-1)
                pred_j[mask] = self._inv_log_transform(y_log)
            # fallback to ALL
            all_model = model_entry.get("ALL")
            if all_model is not None:
                missing = ~np.isfinite(pred_j)
                if missing.any():
                    y_log_all = np.asarray(all_model.predict(X_exp[missing])).reshape(
                        -1
                    )
                    pred_j[missing] = self._inv_log_transform(y_log_all)
            return pred_j

        # no sat per-instance: prefer ALL, else min(SAT,UNSAT)
        if model_entry.get("ALL") is not None:
            y_log = np.asarray(model_entry["ALL"].predict(X_exp)).reshape(-1)
            return self._inv_log_transform(y_log)

        pred_j = np.full(n, np.inf, dtype=float)
        for status in ("SAT", "UNSAT"):
            model = model_entry.get(status)
            if model is None:
                continue
            y_log = np.asarray(model.predict(X_exp)).reshape(-1)
            pred_j = np.minimum(pred_j, self._inv_log_transform(y_log))
        return pred_j

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        sat_labels: Optional[List[Any]] = None,
    ) -> None:
        """
        Fit per-algorithm models.

        Args:
            features: DataFrame of instance features (n_instances x n_features).
            performance: DataFrame of runtimes (n_instances x n_algorithms).
            sat_labels: Optional array-like aligned with features.index containing
                        SAT/UNSAT labels; if provided, train per-status models.
        """
        self.algorithms = list(performance.columns)
        X_exp = self._prepare_features(features)
        n = X_exp.shape[0]

        sat_norm = self._normalize_sat_labels(sat_labels, n=n)

        self.models = {}
        for algo in self.algorithms:
            y_raw = performance[algo].values.astype(float)
            if sat_norm is None:
                self.models[algo] = self._fit_target_model(y_raw, X_exp)
            else:
                entry: Dict[str, Optional[object]] = {}
                entry["ALL"] = self._fit_target_model(y_raw, X_exp)
                for status in ("SAT", "UNSAT"):
                    mask_status = (~np.isnan(y_raw)) & (sat_norm == status)
                    if not mask_status.any():
                        entry[status] = None
                        continue
                    y_status = np.full_like(y_raw, np.nan, dtype=float)
                    y_status[mask_status] = y_raw[mask_status]
                    entry[status] = self._fit_target_model(y_status, X_exp)
                self.models[algo] = entry

    def _predict(
        self,
        features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[Tuple[Optional[str], float]]]:
        """
        Predict best algorithm per instance.

        Args:
            features: DataFrame of instance features to predict for.

        Returns:
            Mapping instance_name -> [(algorithm_name_or_None, predicted_runtime)].
        """
        if features is None:
            raise ValueError("Features must be provided for prediction.")
        if self._feature_means is None or self._feature_stds is None:
            raise RuntimeError("SATzillaSelector must be fitted before prediction.")

        X = features.values.astype(float)
        X_norm = (X - self._feature_means) / self._feature_stds
        X_exp = self._expand_features(X_norm)
        n = X_exp.shape[0]
        algos = self.algorithms
        preds = np.full((n, len(algos)), np.inf, dtype=float)

        for j, algo in enumerate(algos):
            model_entry = self.models.get(algo)
            preds[:, j] = self._predict_for_entry(model_entry, X_exp, None)

        best_idx = np.argmin(preds, axis=1)
        results: Dict[str, List[Tuple[Optional[str], float]]] = {}
        for i, inst in enumerate(features.index):
            j = int(best_idx[i])
            algo = algos[j] if np.isfinite(preds[i, j]) else None
            pred_time = (
                float(preds[i, j])
                if algo is not None and np.isfinite(preds[i, j])
                else float("inf")
            )
            results[inst] = [(algo, pred_time)]
        return results
