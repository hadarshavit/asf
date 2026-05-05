from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from asf.predictors.survival import SKSURV_AVAILABLE, RandomSurvivalForestWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

from asf.utils.configurable import ClassChoice, ConfigurableMixin

if SKSURV_AVAILABLE:
    from sksurv.util import Surv

    try:
        from ConfigSpace import (  # noqa: F401
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

        Attributes
        ----------
        use_schedule : bool
            Whether to build a schedule of multiple algorithms.
        max_schedule_length : int or None
            Maximum number of algorithms in a schedule.
        popsize : int
            Population size for differential evolution.
        maxiter : int
            Maximum iterations for differential evolution.
        tol : float
            Tolerance for differential evolution.
        dominance_resolution : int
            Resolution for dominance analysis.
        random_state : int or None
            Random seed for differential evolution.
        survival_features : list[str]
            Feature column names used by the survival model.
        model : RandomSurvivalForestWrapper or None
            Trained survival model.
        """

        PREFIX = "survival"
        RETURN_TYPE = "single"

        def __init__(
            self,
            model_class: Any = RandomSurvivalForestWrapper,
            **kwargs: Any,
        ) -> None:
            """
            Initialize the SurvivalAnalysis selector.

            Parameters
            ----------
            model_class : type[RandomSurvivalForestWrapper], default=RandomSurvivalForestWrapper
                Wrapper class for the survival model.
            **kwargs : Any
                Additional keyword arguments.
            """
            super().__init__(model_class=model_class, **kwargs)
            self.use_schedule = False
            self.max_schedule_length: int | None = None
            self.popsize = 20
            self.maxiter = 150
            self.tol = 0.01
            self.dominance_resolution = 100
            self.random_state: int | None = 42

            if not isinstance(self.budget, (int, float)) or self.budget <= 0:
                raise ValueError(
                    "budget must be a positive number for survival analysis selector."
                )

            self.survival_features: list[str] = []
            self.model: RandomSurvivalForestWrapper | None = None

        def _build_design_matrix(self, features: pd.DataFrame) -> np.ndarray:
            """Build the instance-algorithm design matrix without object-heavy pandas expansion."""
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
            """
            Fit the survival analysis model.

            Parameters
            ----------
            features : pd.DataFrame
                Training features.
            performance : pd.DataFrame
                Training performance data.
            """
            self.survival_features = [str(col) for col in features.columns] + [
                f"algo_{algo}" for algo in self.algorithms
            ]

            fit_features = self._build_design_matrix(features)

            budget = float(self.budget or 0)
            runtimes = performance.reindex(
                index=features.index, columns=self.algorithms
            ).to_numpy(dtype=float, copy=False)
            finished = np.isfinite(runtimes) & (runtimes < budget)
            runtime_values = np.where(finished, runtimes, budget)

            y_structured = Surv.from_arrays(
                event=finished.ravel(),
                time=runtime_values.ravel(),
            )

            self.model = self.model_class()
            self.model.fit(fit_features, y_structured)

        def _predict(
            self,
            features: pd.DataFrame | None,
            performance: pd.DataFrame | None = None,
        ) -> dict[str, list[tuple[str, float]]]:
            """
            Predict algorithm schedules for each instance.

            Parameters
            ----------
            features : pd.DataFrame
                The query instance features.

            Returns
            -------
            dict
                Mapping from instance name to algorithm schedules.
            """
            if features is None:
                raise ValueError("SurvivalAnalysis require features for prediction.")
            if self.model is None:
                raise ValueError("Model has not been fitted yet.")

            pred_features = self._build_design_matrix(features)
            predicted_survival_functions = self.model.predict_survival_function(
                pred_features
            )

            survival_by_instance: dict[Any, dict[str, Any]] = {}
            n_algorithms = len(self.algorithms)
            for row_idx, surv_func in enumerate(predicted_survival_functions):
                instance = features.index[row_idx // n_algorithms]
                algo = self.algorithms[row_idx % n_algorithms]
                survival_by_instance.setdefault(instance, {})[str(algo)] = surv_func

            predictions: dict[str, list[tuple[str, float]]] = {}
            for instance in features.index:
                surv_funcs = survival_by_instance[instance]

                if not self.use_schedule:
                    best_algo = None
                    best_prob = -1.0
                    for algo, surv_func in surv_funcs.items():
                        completion_prob = 1.0 - float(surv_func(self.budget))
                        if completion_prob > best_prob:
                            best_prob = completion_prob
                            best_algo = algo
                    predictions[str(instance)] = [
                        (str(best_algo), float(self.budget or 0))
                    ]
                else:
                    schedule = self._find_optimal_schedule(surv_funcs)
                    predictions[str(instance)] = schedule if schedule else []

            return predictions

        def _eval_schedule(
            self, x: np.ndarray, all_algos: list[str], surv_funcs: dict[str, Any]
        ) -> float:
            """
            Evaluate the fitness of a schedule for differential evolution.

            Parameters
            ----------
            x : np.ndarray
                The vector from the optimizer.
            all_algos : list[str]
                List of available algorithms.
            surv_funcs : dict
                Mapping from algorithm names to survival functions.

            Returns
            -------
            float
                Negative success probability.
            """
            n_algorithms = len(all_algos)

            inclusion_flags = x[:n_algorithms]
            normalized_end_times = x[n_algorithms:]

            n_included = np.sum(inclusion_flags >= 0.5)
            if n_included == 0:
                return 0.0
            if (
                self.max_schedule_length is not None
                and n_included > self.max_schedule_length
            ):
                return 1.0

            included_schedule_info = []
            for i, algo in enumerate(all_algos):
                if inclusion_flags[i] >= 0.5:
                    included_schedule_info.append((algo, normalized_end_times[i]))

            included_schedule_info.sort(key=lambda item: item[1])

            total_success_prob = 0.0
            prob_of_reaching_step = 1.0
            last_actual_end_time = 0.0

            for i, (algo, norm_end_time) in enumerate(included_schedule_info):
                if i == len(included_schedule_info) - 1:
                    time_slice = float(self.budget or 0) - last_actual_end_time
                else:
                    actual_end_time = float(norm_end_time) * float(self.budget or 0)
                    time_slice = actual_end_time - last_actual_end_time

                if time_slice <= 1e-6:
                    continue

                sur_func = surv_funcs[algo]
                prob_solve_at_this_step = 1.0 - float(sur_func(time_slice))

                total_success_prob += prob_of_reaching_step * prob_solve_at_this_step
                prob_of_reaching_step *= float(sur_func(time_slice))

                last_actual_end_time += time_slice

            return -total_success_prob

        def _find_optimal_schedule(
            self, surv_funcs: dict[str, Any]
        ) -> list[tuple[str, float]]:
            """
            Find the optimal schedule using dominance analysis and evolution.

            Parameters
            ----------
            surv_funcs : dict
                Mapping from algorithm names to survival functions.

            Returns
            -------
            list[tuple[str, float]]
                The optimized algorithm schedule.
            """
            time_grid = np.linspace(
                0, float(self.budget or 0), self.dominance_resolution
            )
            prob_matrix = np.array(
                [surv_funcs[algo](time_grid) for algo in self.algorithms]
            )

            for i, algo in enumerate(self.algorithms):
                is_dominant = np.all(prob_matrix[i, :] <= prob_matrix)
                if is_dominant:
                    return [(str(algo), float(self.budget or 0))]

            lower_envelope = np.min(prob_matrix, axis=0)
            non_dominated_algos = []
            for i, algo in enumerate(self.algorithms):
                if np.any(np.isclose(prob_matrix[i, :], lower_envelope)):
                    non_dominated_algos.append(str(algo))

            if len(non_dominated_algos) <= 1:
                if non_dominated_algos:
                    return [(str(non_dominated_algos[0]), float(self.budget or 0))]
                else:
                    return []

            n_to_optimize = len(non_dominated_algos)
            bounds = [(0, 1)] * (2 * n_to_optimize)

            result = differential_evolution(
                func=self._eval_schedule,
                bounds=bounds,
                args=(non_dominated_algos, surv_funcs),
                popsize=self.popsize,
                maxiter=self.maxiter,
                tol=self.tol,
                seed=self.random_state,
            )

            best_x = result.x
            inclusion_flags = best_x[:n_to_optimize]
            normalized_end_times = best_x[n_to_optimize:]

            included_info = []
            for i, algo in enumerate(non_dominated_algos):
                if inclusion_flags[i] >= 0.5:
                    included_info.append((algo, normalized_end_times[i]))

            if not included_info:
                return []

            included_info.sort(key=lambda item: item[1])

            schedule: list[tuple[str, float]] = []
            last_actual_end_time = 0.0
            for i, (algo, norm_end_time) in enumerate(included_info):
                if i == len(included_info) - 1:
                    time_slice = float(self.budget or 0) - last_actual_end_time
                else:
                    actual_end_time = float(norm_end_time) * float(self.budget or 0)
                    time_slice = actual_end_time - last_actual_end_time

                if time_slice > 1e-6:
                    schedule.append((str(algo), float(time_slice)))
                last_actual_end_time += time_slice

            return schedule

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
            super().__init__(model_class=model_class, **kwargs)
            self.use_schedule = True
            self.max_schedule_length = max_schedule_length
            self.popsize = int(popsize)
            self.maxiter = int(maxiter)
            self.tol = float(tol)
            self.dominance_resolution = int(dominance_resolution)
            self.random_state = random_state

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

            dominance_resolution_param = Integer(
                name="dominance_resolution",
                bounds=(50, 500),
                default=100,
            )

            params = [
                model_class_param,
                popsize_param,
                maxiter_param,
                tol_param,
                dominance_resolution_param,
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
