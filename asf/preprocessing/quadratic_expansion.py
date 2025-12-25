"""
Schmee & Hahn iterative imputation for censored data.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import norm


def schmee_hahn_impute(
    base_epm: Callable[[], Any],
    y_raw: np.ndarray,
    X_exp: np.ndarray,
    em_max_iter: int = 20,
    em_tol: float = 1e-3,
    em_min_sigma: float = 1e-6,
    budget: float = 300.0,
) -> tuple[np.ndarray, Any | None]:
    """
    Perform Schmee & Hahn iterative imputation on the log scale.

    Parameters
    ----------
    base_epm : Callable
        A callable that returns a fresh EPM model instance.
    y_raw : np.ndarray
        Raw runtimes array (NaN for missing entries).
    X_exp : np.ndarray
        Expanded feature matrix used for fitting.
    em_max_iter : int, default=20
        Maximum number of EM iterations.
    em_tol : float, default=1e-3
        Tolerance for convergence.
    em_min_sigma : float, default=1e-6
        Minimum standard deviation for the normal distribution.
    budget : float, default=300.0
        The algorithm cutoff time.

    Returns
    -------
    tuple
        (y_imputed, model) where y_imputed is the array of log-scale targets
        and model is the fitted model instance trained on the final imputed targets.
    """
    mask_not_nan = ~np.isnan(y_raw)
    if not mask_not_nan.any():
        return np.full_like(y_raw, np.nan, dtype=float), None

    mask_obs = mask_not_nan & (y_raw < budget)
    mask_cens = mask_not_nan & (y_raw >= budget)

    y_imputed = np.full_like(y_raw, np.nan, dtype=float)
    if mask_obs.any():
        y_imputed[mask_obs] = y_raw[mask_obs].astype(float)
    if mask_cens.any():
        y_imputed[mask_cens] = budget

    fit_mask = mask_not_nan
    prev_vals = y_imputed.copy()

    model = base_epm()

    for _ in range(em_max_iter):
        model.fit(X_exp[fit_mask], y_imputed[fit_mask])

        mu_all = np.asarray(model.predict(X_exp)).reshape(-1)

        if mask_obs.any():
            resid = y_imputed[mask_obs] - mu_all[mask_obs]
            sigma = np.sqrt(np.mean(resid**2))
        else:
            sigma = 0.0
        sigma = max(sigma, em_min_sigma)

        if mask_cens.any():
            mu_c = mu_all[mask_cens]
            z = (budget - mu_c) / sigma
            sf = 1.0 - norm.cdf(z)
            sf = np.maximum(sf, 1e-12)
            expected = mu_c + sigma * (norm.pdf(z) / sf)
            y_imputed[mask_cens] = expected

            delta = np.max(np.abs(y_imputed[mask_cens] - prev_vals[mask_cens]))
        else:
            delta = 0.0

        prev_vals[mask_cens] = y_imputed[mask_cens]
        if delta < em_tol:
            break

    return y_imputed, model
