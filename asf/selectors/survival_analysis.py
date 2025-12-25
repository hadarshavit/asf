from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from asf.predictors.survival import SKSURV_AVAILABLE, RandomSurvivalForestWrapper
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

if SKSURV_AVAILABLE:
    from sksurv.util import Surv

    from asf.utils.configurable import ClassChoice, ConfigurableMixin

    try:
        from ConfigSpace import (  # noqa: F401
            Categorical,
            ConfigurationSpace,
            EqualsCondition,
            Float,
            Integer,
        )

        CONFIGSPACE_AVAILABLE = True
    except ImportError:
        CONFIGSPACE_AVAILABLE = False

    class SurvivalAnalysis(ConfigurableMixin, AbstractModelBasedSelector):
        """
        Selector using survival analysis for algorithm selection.

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
            use_schedule: bool = False,
            max_schedule_length: int | None = None,
            popsize: int = 20,
            maxiter: int = 150,
            tol: float = 0.01,
            dominance_resolution: int = 100,
            **kwargs: Any,
        ) -> None:
            """
            Initialize the SurvivalAnalysis selector.

            Parameters
            ----------
            model_class : type[RandomSurvivalForestWrapper], default=RandomSurvivalForestWrapper
                Wrapper class for the survival model.
            use_schedule : bool, default=False
                Whether to build a schedule.
            max_schedule_length : int or None, default=None
                Maximum number of algorithms in a schedule.
            popsize : int, default=20
                Population size for differential_evolution.
            maxiter : int, default=150
                Max iterations for differential_evolution.
            tol : float, default=0.01
                Tolerance for convergence.
            dominance_resolution : int, default=100
                Resolution for dominance analysis grid.
            **kwargs : Any
                Additional keyword arguments.
            """
            super().__init__(model_class=model_class, **kwargs)
            self.use_schedule = bool(use_schedule)
            self.max_schedule_length = max_schedule_length
            self.popsize = int(popsize)
            self.maxiter = int(maxiter)
            self.tol = float(tol)
            self.dominance_resolution = int(dominance_resolution)

            if use_schedule:
                self.RETURN_TYPE = "schedule"

            if not isinstance(self.budget, (int, float)) or self.budget <= 0:
                raise ValueError(
                    "budget must be a positive number for survival analysis selector."
                )

            self.survival_features: list[str] = []
            self.model: RandomSurvivalForestWrapper | None = None

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
            fit_data = []
            for instance in features.index:
                instance_features = features.loc[instance]
                for algo in self.algorithms:
                    runtime = performance.loc[instance, algo]
                    finished = not pd.isna(runtime) and runtime < float(
                        self.budget or 0
                    )
                    status = int(finished)
                    runtime_val = (
                        float(runtime) if finished else float(self.budget or 0)
                    )
                    row = {
                        **instance_features.to_dict(),
                        "algorithm": algo,
                        "runtime": runtime_val,
                        "status": status,
                    }
                    fit_data.append(row)
            fit_df = pd.DataFrame(fit_data)

            fit_features = pd.get_dummies(
                fit_df.drop(columns=["runtime", "status"]),
                columns=["algorithm"],
                prefix="algo",
            )

            self.survival_features = fit_features.columns.tolist()

            y_structured = Surv.from_arrays(
                event=fit_df["status"].astype(bool).values,
                time=fit_df["runtime"].values,
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

            predictions: dict[str, list[tuple[str, float]]] = {}
            for instance, instance_features in features.iterrows():
                surv_funcs = {}
                for algo in self.algorithms:
                    pred_row = pd.DataFrame(
                        [{**instance_features.to_dict(), "algorithm": algo}]
                    )
                    pred_row = pd.get_dummies(
                        pred_row, columns=["algorithm"], prefix="algo"
                    )
                    pred_row = pred_row.reindex(
                        columns=self.survival_features, fill_value=0
                    )
                    surv_funcs[algo] = self.model.predict_survival_function(pred_row)[0]

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
                seed=42,
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
            model_class: list[type[RandomSurvivalForestWrapper]] | None = None,
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
                choices: list[Any] = [RandomSurvivalForestWrapper]
            else:
                choices = model_class

            model_class_param = ClassChoice(
                name="model_class",
                choices=choices,  # type: ignore[arg-type]
                default=choices[0],
            )

            use_schedule_param = Categorical(
                name="use_schedule",
                items=[True, False],
                default=False,
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
                use_schedule_param,
                popsize_param,
                maxiter_param,
                tol_param,
                dominance_resolution_param,
            ]

            conditions = [
                EqualsCondition(popsize_param, use_schedule_param, True),
                EqualsCondition(maxiter_param, use_schedule_param, True),
                EqualsCondition(tol_param, use_schedule_param, True),
                EqualsCondition(dominance_resolution_param, use_schedule_param, True),
            ]

            return params, conditions, []

        @classmethod
        def _get_from_clean_configuration(
            cls,
            clean_config: dict[str, Any],
            **kwargs: Any,
        ) -> partial[SurvivalAnalysis]:
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
                Partial function for SurvivalAnalysis.
            """
            config = clean_config.copy()
            config.update(kwargs)
            return partial(SurvivalAnalysis, **config)

else:

    class SurvivalAnalysis(AbstractSelector):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("sksurv is not installed.")
