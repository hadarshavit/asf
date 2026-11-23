from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from asf.predictors.survival import SKSURV_AVAILABLE, RandomSurvivalForestWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

from asf.utils.configurable import ClassChoice, ConfigurableMixin

if SKSURV_AVAILABLE:
    from sksurv.functions import StepFunction
    from sksurv.util import Surv

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

    class SurvivalAnalysis(ConfigurableMixin, AbstractModelBasedSelector):
        """
        Selector using survival analysis for algorithm selection.

        References
        ----------
        Tornede, A., et al. (2020).
        "Algorithm Selection with Survival Forests."
        https://arxiv.org/abs/2007.02816
        """

        PREFIX = "survival"
        RETURN_TYPE = "single"

        def __init__(
            self,
            model_class: Any = RandomSurvivalForestWrapper,
            random_state: int | None = 42,
            **kwargs: Any,
        ) -> None:
            """Select the solver with the lowest expected runtime."""
            super().__init__(
                model_class=model_class, random_state=random_state, **kwargs
            )
            self.use_schedule = False
            if self.budget is None or not np.isfinite(self.budget) or self.budget <= 0:
                raise ValueError(
                    "budget must be a finite positive number for survival analysis."
                )

            self.model: RandomSurvivalForestWrapper | None = None
            self.models_: dict[str, Any] | None = None

        def _build_design_matrix(self, features: pd.DataFrame) -> np.ndarray:
            """Recreate the design matrix for a saved pooled forest."""
            n_instances = features.shape[0]
            n_algorithms = len(self.algorithms)
            instance_features = features.to_numpy(dtype=np.float32, copy=False)
            expanded_features = np.repeat(instance_features, n_algorithms, axis=0)
            algorithm_features = np.tile(
                np.eye(n_algorithms, dtype=np.float32), (n_instances, 1)
            )
            return np.concatenate((expanded_features, algorithm_features), axis=1)

        def _fit(
            self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
        ) -> None:
            budget = float(self.budget or 0)
            runtimes = performance.reindex(index=features.index).to_numpy(
                dtype=float, na_value=np.nan
            )
            if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms):
                raise ValueError("Survival analysis requires unique algorithm names.")

            fit_features = features.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(runtimes)
            runtime_horizon = float(np.max(runtimes, where=finite, initial=budget))
            models = {}
            for algo, times, observed in zip(self.algorithms, runtimes.T, finite.T):
                finished = observed & (times < budget)
                if not finished.any():
                    models[algo] = None
                    continue
                model = self._make_model()
                model.fit(
                    fit_features,
                    Surv.from_arrays(finished, np.where(observed, times, budget)),
                )
                models[algo] = model

            self._survival_columns = features.columns.copy()
            self._runtime_horizon = runtime_horizon
            self.__dict__.pop("_survival_preprocessor", None)
            self.models_ = models
            self.model = None

        def _budget_curve(self, curve: StepFunction) -> StepFunction:
            """Extend a predicted survival curve to cover the full budget."""
            budget = float(self.budget or 0)
            events = curve.x[(curve.x > 0) & (curve.x < budget)]
            times = np.r_[0.0, events, budget]
            values = np.asarray(curve(np.minimum(times, curve.x[-1]))).copy()
            values[times < curve.x[0]] = 1.0
            return StepFunction(times, values)

        def _expected_runtime(self, curve: StepFunction) -> float:
            budget = float(self.budget or 0)
            horizon = getattr(self, "_runtime_horizon", budget)
            # No completion events occur after the budget.
            return float(
                np.dot(curve.y[:-1], np.diff(curve.x))
                + (horizon - budget) * curve.y[-1]
            )

        def _predict(
            self,
            features: pd.DataFrame | None,
            performance: pd.DataFrame | None = None,
        ) -> dict[str, list[tuple[str, float]]]:
            if features is None:
                raise ValueError("SurvivalAnalysis requires features for prediction.")

            budget = float(self.budget or 0)
            models = getattr(self, "models_", None)
            if models is not None:
                if hasattr(self, "_survival_preprocessor"):
                    raise ValueError(
                        "Refit this survival selector to use untransformed features."
                    )
                pred_features = features.loc[:, self._survival_columns].to_numpy(
                    dtype=float, na_value=np.nan
                )
                no_event = StepFunction(np.array([0.0, budget]), np.ones(2))
                curves_by_algorithm = {
                    algo: (
                        [no_event] * len(features)
                        if model is None
                        else [
                            self._budget_curve(curve)
                            for curve in model.predict_survival_function(pred_features)
                        ]
                    )
                    for algo, model in models.items()
                }
            elif self.model is not None:
                # Saved selectors may still contain a single pooled forest.
                curves = self.model.predict_survival_function(
                    self._build_design_matrix(features)
                )
                curves_by_algorithm = {
                    algo: [
                        self._budget_curve(curve)
                        for curve in curves[i :: len(self.algorithms)]
                    ]
                    for i, algo in enumerate(self.algorithms)
                }
            else:
                raise ValueError("Model has not been fitted yet.")

            predictions = {}
            for row, instance in enumerate(features.index):
                surv_funcs = {
                    algo: curves[row] for algo, curves in curves_by_algorithm.items()
                }
                predictions[str(instance)] = self._find_optimal_schedule(surv_funcs)
            return predictions

        def _find_optimal_schedule(
            self, surv_funcs: dict[str, StepFunction]
        ) -> list[tuple[str, float]]:
            best_algo = min(
                surv_funcs, key=lambda algo: self._expected_runtime(surv_funcs[algo])
            )
            return [(best_algo, float(self.budget or 0))]

        @staticmethod
        def _define_hyperparameters(
            model_class: list[type | bool] | None = None,
            **kwargs: Any,
        ) -> tuple[list[Any], list[Any], list[Any]]:
            """
            Define hyperparameters for SurvivalAnalysis.

            Parameters
            ----------
            model_class : list[type] or None, default=None
                List of model classes to choose from.
            **kwargs : Any
                Additional keyword arguments.

            Returns
            -------
            tuple
                Tuple of (hyperparameters, conditions, forbiddens).
            """
            if not CONFIGSPACE_AVAILABLE:
                return [], [], []

            if model_class is None:
                choices: list[type | bool] = [RandomSurvivalForestWrapper]
            else:
                choices = model_class

            model_class_param = ClassChoice(
                name="model_class",
                choices=choices,
                default=choices[0],
            )

            return [model_class_param], [], []

    class SurvivalAnalysisScheduler(SurvivalAnalysis):
        """
        Scheduling variant of SurvivalAnalysis.

        This selector predicts schedules of multiple algorithms and exposes the
        differential-evolution schedule optimizer hyperparameters.
        """

        PREFIX = "survival_schedule"
        RETURN_TYPE = "schedule"

        def __init__(
            self,
            model_class: Any = RandomSurvivalForestWrapper,
            max_schedule_length: int | None = None,
            popsize: int = 20,
            maxiter: int = 150,
            tol: float = 0.01,
            dominance_resolution: int = 100,
            random_state: int | None = 42,
            **kwargs: Any,
        ) -> None:
            """Keep accepting dominance_resolution for older configurations."""
            super().__init__(
                model_class=model_class, random_state=random_state, **kwargs
            )
            self.use_schedule = True
            if max_schedule_length is not None and max_schedule_length < 1:
                raise ValueError("max_schedule_length must be at least 1.")
            self.max_schedule_length = max_schedule_length
            self.popsize = int(popsize)
            self.maxiter = int(maxiter)
            self.tol = float(tol)
            self.random_state = random_state

        def _decode_schedule(
            self,
            x: np.ndarray,
            algorithms: list[str],
            surv_funcs: dict[str, StepFunction],
        ) -> list[tuple[str, float]]:
            budget = float(self.budget or 0)
            included = [
                (algo, float(end) * budget)
                for algo, flag, end in zip(
                    algorithms, x[: len(algorithms)], x[len(algorithms) :]
                )
                if flag >= 0.5
            ]
            included.sort(key=lambda item: item[1])
            schedule = []
            previous_end = 0.0
            for i, (algo, end) in enumerate(included):
                if i == len(included) - 1:
                    end = budget
                duration = end - previous_end
                if i < len(included) - 1:
                    # Extra time within a step can go to the next solver.
                    times = surv_funcs[algo].x
                    duration = float(
                        times[np.searchsorted(times, duration, side="right") - 1]
                    )
                if duration > 0:
                    schedule.append((algo, duration))
                    previous_end += duration
            return schedule

        def _eval_schedule(
            self,
            x: np.ndarray,
            algorithms: list[str],
            surv_funcs: dict[str, StepFunction],
        ) -> float:
            schedule = self._decode_schedule(x, algorithms, surv_funcs)
            if not schedule or (
                self.max_schedule_length is not None
                and len(schedule) > self.max_schedule_length
            ):
                return 1.0
            return float(
                np.prod([surv_funcs[algo](time) for algo, time in schedule]) - 1
            )

        def _find_optimal_schedule(
            self, surv_funcs: dict[str, StepFunction]
        ) -> list[tuple[str, float]]:
            budget = float(self.budget or 0)
            failure = {algo: float(curve(budget)) for algo, curve in surv_funcs.items()}
            best_algo = min(failure, key=failure.__getitem__)
            baseline = [(best_algo, budget)]
            # A solver that never completes by budget cannot improve any slice.
            candidates = [algo for algo in failure if failure[algo] < 1.0]
            if len(candidates) <= 1 or self.max_schedule_length == 1:
                return baseline

            start = np.zeros(2 * len(candidates))
            start[candidates.index(best_algo)] = 1
            result = differential_evolution(
                func=self._eval_schedule,
                bounds=[(0, 1)] * len(start),
                args=(candidates, surv_funcs),
                popsize=self.popsize,
                maxiter=self.maxiter,
                tol=self.tol,
                seed=self.random_state,
                x0=start,
            )
            if result.fun < failure[best_algo] - 1:
                return self._decode_schedule(result.x, candidates, surv_funcs)
            return baseline

        @staticmethod
        def _define_hyperparameters(
            model_class: list[type | bool] | None = None,
            **kwargs: Any,
        ) -> tuple[list[Any], list[Any], list[Any]]:
            """
            Define hyperparameters for SurvivalAnalysisScheduler.
            """
            if not CONFIGSPACE_AVAILABLE:
                return [], [], []

            if model_class is None:
                choices: list[type | bool] = [RandomSurvivalForestWrapper]
            else:
                choices = model_class

            model_class_param = ClassChoice(
                name="model_class",
                choices=choices,
                default=choices[0],
            )

            max_schedule_length_param = Categorical(
                name="max_schedule_length",
                items=[None, *range(1, 11)],
                default=None,
            )

            popsize_param = Integer(
                name="popsize",
                bounds=(10, 100),
                default=20,
            )

            maxiter_param = Integer(
                name="maxiter",
                bounds=(50, 500),
                default=150,
            )

            tol_param = Float(
                name="tol",
                bounds=(1e-4, 1e-1),
                log=True,
                default=0.01,
            )

            params = [
                model_class_param,
                max_schedule_length_param,
                popsize_param,
                maxiter_param,
                tol_param,
            ]

            return params, [], []


else:
    from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

    class SurvivalAnalysis(ConfigurableMixin, AbstractModelBasedSelector):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("sksurv is not installed.")

        @staticmethod
        def _define_hyperparameters(
            **kwargs: Any,
        ) -> tuple[list[Any], list[Any], list[Any]]:
            return [], [], []

    class SurvivalAnalysisScheduler(SurvivalAnalysis):
        pass
