import numpy as np
import pandas as pd

from typing import Dict

from asf.selectors.abstract_selector import AbstractSelector

try:
    from scipy.optimize import minimize
except Exception as e:
    raise ImportError("scipy is required for OSL optimizer") from e


from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (
        Categorical,
        Integer,
        Float,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Any


class OSLLinearSelector(ConfigurableMixin, AbstractSelector):
    """
    Selector using Optimistic Superset Loss (OSL) to predict runtimes.
    """

    PREFIX = "osl_linear"
    RETURN_TYPE = "single"

    def __init__(
        self,
        budget: float,
        reg: float = 0.0,
        optimizer_method: str = "L-BFGS-B",
        maxiter: int = 1000,
        tol: float | None = None,
        **kwargs,
    ):
        """
        Args:
            budget (float): global cutoff time. Observations with runtime >= budget
                or NaN are treated as right-censored at this cutoff.
            reg (float): L2 regularization strength applied to theta (0 disables).
            optimizer_method (str): optimization algorithm name for scipy.optimize.minimize
                (e.g. "L-BFGS-B", "CG").
            maxiter (int): maximum number of optimizer iterations.
            tol (float | None): tolerance for the optimizer (if supported by method).
            **kwargs: forwarded to AbstractSelector.
        """
        super().__init__(**kwargs)
        self.budget = float(budget)
        self.reg = float(reg)
        self.optimizer_method = optimizer_method
        self.maxiter = int(maxiter)
        self.tol = None if tol is None else float(tol)
        self.thetas: Dict[str, np.ndarray] = {}
        self.algorithms: list[str] = []

    # --- OSL objective + gradient for one algorithm ---
    def _osl_obj_grad(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        censored_mask: np.ndarray,
        C: float,
    ):
        """
        Compute loss and gradient for parameters theta.

        X: (n, d) design matrix (including bias column if present)
        y: (n,) observed runtimes (NaN allowed)
        censored_mask: (n,) bool array: True if observation is censored (right-censored at C)
        C: cutoff (float)
        """
        preds = X.dot(theta)
        precise_mask = ~censored_mask & ~np.isnan(y)
        cens_pred_mask = censored_mask & (preds < C)

        loss_precise = (
            ((y[precise_mask] - preds[precise_mask]) ** 2).sum()
            if precise_mask.any()
            else 0.0
        )
        loss_cens = (
            (((C - preds[cens_pred_mask]) ** 2).sum()) if cens_pred_mask.any() else 0.0
        )
        loss = loss_precise + loss_cens

        # L2 regularizer
        if self.reg:
            loss += 0.5 * self.reg * np.sum(theta**2)

        # gradient
        grad = np.zeros_like(theta)
        if precise_mask.any():
            # derivative: d/dtheta (y - p)^2 = -2 * X^T (y - p)
            resid = y[precise_mask] - preds[precise_mask]
            grad_prec = -2.0 * (X[precise_mask].T.dot(resid))
            grad += grad_prec
        if cens_pred_mask.any():
            # derivative: d/dtheta (C - p)^2 = -2 * X^T (C - p)
            diff = C - preds[cens_pred_mask]
            grad_cens = -2.0 * (X[cens_pred_mask].T.dot(diff))
            grad += grad_cens

        if self.reg:
            grad += self.reg * theta

        return float(loss), grad

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit linear models for each algorithm by minimizing OSL.

        features: DataFrame (n_instances x n_features)
        performance: DataFrame (n_instances x n_algorithms) observed runtimes (timeouts may be >= budget or NaN)
        """
        X_base = np.asarray(features, dtype=float)
        n, d = X_base.shape
        # augment bias term
        X = np.hstack([X_base, np.ones((n, 1), dtype=float)])

        self.algorithms = list(performance.columns)
        thetas: Dict[str, np.ndarray] = {}

        censored = (performance >= self.budget) | performance.isna()

        for algo in self.algorithms:
            y_col = performance[algo].to_numpy(dtype=float)
            cens_mask = censored[algo].to_numpy(dtype=bool)

            # init theta by ordinary least squares on uncensored/zeros
            try:
                unc_idx = (~cens_mask) & (~np.isnan(y_col))
                if unc_idx.sum() >= d + 1:
                    theta0, *_ = np.linalg.lstsq(X[unc_idx], y_col[unc_idx], rcond=None)
                else:
                    theta0 = np.zeros(d + 1, dtype=float)
            except Exception:
                theta0 = np.zeros(d + 1, dtype=float)

            def fun_and_grad(th):
                val, grad = self._osl_obj_grad(th, X, y_col, cens_mask, self.budget)
                return val, grad

            res = minimize(
                fun=lambda th: fun_and_grad(th)[0],
                x0=theta0,
                jac=lambda th: fun_and_grad(th)[1],
                method=self.optimizer_method,
                options={"maxiter": self.maxiter, "disp": False},
            )
            theta_opt = res.x if res.success else res.x
            thetas[algo] = theta_opt

        self.thetas = thetas

    def _predict(self, features: pd.DataFrame | None = None) -> dict:
        """
        Predict best algorithm per instance.

        Returns mapping instance_name -> [(algo, budget)]
        """
        X_base = np.asarray(features, dtype=float)
        n = X_base.shape[0]
        X = np.hstack([X_base, np.ones((n, 1), dtype=float)])

        preds_per_algo = {}
        for algo, theta in self.thetas.items():
            preds = X.dot(theta)
            preds_per_algo[algo] = preds

        out: dict[str, list[tuple[str, float]]] = {}
        algs = list(self.algorithms)
        for i, idx in enumerate(features.index):
            best_algo = None
            best_val = float("inf")
            for algo in algs:
                val = float(preds_per_algo[algo][i])
                if val < best_val:
                    best_val = val
                    best_algo = algo
            out[idx] = [(best_algo, self.budget)]
        return out

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for OSLLinearSelector."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        reg_param = Float(
            name="reg",
            bounds=(0.0, 10.0),  # Allow 0.0
            default=0.0,
        )

        optimizer_method_param = Categorical(
            name="optimizer_method",
            items=["L-BFGS-B", "CG", "BFGS", "TNC", "SLSQP"],
            default="L-BFGS-B",
        )

        maxiter_param = Integer(
            name="maxiter",
            bounds=(100, 5000),
            default=1000,
        )

        tol_param = Float(
            name="tol",
            bounds=(1e-6, 1e-2),
            log=True,
            default=1e-5,
        )

        params = [
            reg_param,
            optimizer_method_param,
            maxiter_param,
            tol_param,
        ]

        return params, [], []

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
        return partial(OSLLinearSelector, **config)
