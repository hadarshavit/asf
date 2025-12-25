from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

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


class OSLLinearSelector(ConfigurableMixin, AbstractSelector):
    """
    Selector using Optimistic Superset Loss (OSL) to predict runtimes.

    Attributes
    ----------
    reg : float
        L2 regularization strength.
    optimizer_method : str
        Method for scipy.optimize.minimize.
    maxiter : int
        Maximum number of optimizer iterations.
    tol : float or None
        Tolerance for the optimizer.
    thetas : dict[str, np.ndarray]
        Learned parameters for each algorithm.
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
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OSLLinearSelector.

        Parameters
        ----------
        budget : float
            Global cutoff time.
        reg : float, default=0.0
            L2 regularization strength.
        optimizer_method : str, default="L-BFGS-B"
            Optimization algorithm name for scipy.optimize.minimize.
        maxiter : int, default=1000
            Maximum number of optimizer iterations.
        tol : float or None, default=None
            Tolerance for the optimizer.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.budget = float(budget)
        self.reg = float(reg)
        self.optimizer_method = optimizer_method
        self.maxiter = int(maxiter)
        self.tol = None if tol is None else float(tol)
        self.thetas: dict[str, np.ndarray] = {}

    def _osl_obj_grad(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        censored_mask: np.ndarray,
        C: float,
    ) -> tuple[float, np.ndarray]:
        """
                Compute loss and gradient for parameters theta.

                Parameters
                ----------
                theta : np.ndarray
                    Parameter vector.
                X : np.ndarray
                    Design matrix.
                y : np.ndarray
                    Observed runtimes.
                censored_mask : np.ndarray
                    Boolean mask of censored observations.
                C : float
                    Cutoff time.

                Returns
        -------
                tuple
                    Tuple of (loss, gradient).
        """
        preds = X.dot(theta)
        precise_mask = ~censored_mask & ~np.isnan(y)
        cens_pred_mask = censored_mask & (preds < C)

        loss_precise = (
            float(((y[precise_mask] - preds[precise_mask]) ** 2).sum())
            if precise_mask.any()
            else 0.0
        )
        loss_cens = (
            float(((C - preds[cens_pred_mask]) ** 2).sum())
            if cens_pred_mask.any()
            else 0.0
        )
        loss = loss_precise + loss_cens

        if self.reg:
            loss += 0.5 * self.reg * float(np.sum(theta**2))

        grad = np.zeros_like(theta)
        if precise_mask.any():
            resid = y[precise_mask] - preds[precise_mask]
            grad_prec = -2.0 * (X[precise_mask].T.dot(resid))
            grad += grad_prec
        if cens_pred_mask.any():
            diff = C - preds[cens_pred_mask]
            grad_cens = -2.0 * (X[cens_pred_mask].T.dot(diff))
            grad += grad_cens

        if self.reg:
            grad += self.reg * theta

        return float(loss), grad

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Fit linear models for each algorithm by minimizing OSL.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        X_base = np.asarray(features, dtype=float)
        n, d = X_base.shape
        X = np.hstack([X_base, np.ones((n, 1), dtype=float)])

        self.algorithms = [str(a) for a in performance.columns]
        thetas: dict[str, np.ndarray] = {}

        censored = (performance >= self.budget) | performance.isna()

        for algo in self.algorithms:
            y_col = performance[algo].to_numpy(dtype=float)
            cens_mask = censored[algo].to_numpy(dtype=bool)

            try:
                unc_idx = (~cens_mask) & (~np.isnan(y_col))
                if unc_idx.sum() >= d + 1:
                    theta0, *_ = np.linalg.lstsq(X[unc_idx], y_col[unc_idx], rcond=None)
                    theta0 = np.asarray(theta0, dtype=float)
                else:
                    theta0 = np.zeros(d + 1, dtype=float)
            except Exception:
                theta0 = np.zeros(d + 1, dtype=float)

            def fun_and_grad(th: np.ndarray) -> tuple[float, np.ndarray]:
                val, grad = self._osl_obj_grad(
                    th, X, y_col, cens_mask, float(self.budget or 0)
                )
                return val, grad

            res = minimize(
                fun=lambda th: fun_and_grad(th)[0],
                x0=theta0,
                jac=lambda th: fun_and_grad(th)[1],
                method=self.optimizer_method,
                tol=self.tol,
                options={"maxiter": self.maxiter, "disp": False},
            )
            theta_opt = np.asarray(res.x, dtype=float)
            thetas[str(algo)] = theta_opt

        self.thetas = thetas

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict best algorithm per instance.

        Parameters
        ----------
        features : pd.DataFrame or None
            The input features.
        performance : pd.DataFrame or None, default=None
            Partial performance data.

        Returns
        -------
        dict
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("OSLLinearSelector requires features for prediction.")
        X_base = np.asarray(features, dtype=float)
        n = X_base.shape[0]
        X = np.hstack([X_base, np.ones((n, 1), dtype=float)])

        preds_per_algo = {}
        for algo, theta in self.thetas.items():
            preds = X.dot(theta)
            preds_per_algo[algo] = np.asarray(preds, dtype=float)

        out: dict[str, list[tuple[str, float]]] = {}
        algs = [str(a) for a in self.algorithms]
        for i, idx in enumerate(features.index):
            best_algo = None
            best_val = float("inf")
            for algo in algs:
                val = float(preds_per_algo[algo][i])
                if val < best_val:
                    best_val = val
                    best_algo = algo
            out[str(idx)] = [(str(best_algo), float(self.budget or 0))]
        return out

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for OSLLinearSelector.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        reg_param = Float(
            name="reg",
            bounds=(0.0, 10.0),
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
        **kwargs: Any,
    ) -> partial[OSLLinearSelector]:
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
            Partial function for OSLLinearSelector.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(OSLLinearSelector, **config)
