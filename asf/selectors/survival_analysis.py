import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.predictors.survival import RandomSurvivalForestWrapper, SKSURV_AVAILABLE
from functools import partial
from typing import Any


if SKSURV_AVAILABLE:
    from sksurv.util import Surv

    try:
        from ConfigSpace import (
            Categorical,
            EqualsCondition,
            Integer,
            Float,
        )

        CONFIGSPACE_AVAILABLE = True
    except ImportError:
        CONFIGSPACE_AVAILABLE = False

    from asf.utils.configurable import ConfigurableMixin, ClassChoice

    class SurvivalAnalysis(ConfigurableMixin, AbstractModelBasedSelector):
        """
        Selects the best algorithm for a given problem instance using survival analysis.
        Tries to maximize the probability of finishing within a given time budget.
        Can optionally build a schedule of multiple algorithms using RunAndSchedule2Survive logic.
        """

        PREFIX = "survival"
        RETURN_TYPE = "single"

        def __init__(
            self,
            model_class: type[
                RandomSurvivalForestWrapper
            ] = RandomSurvivalForestWrapper,
            use_schedule: bool = False,
            max_schedule_length: int | None = None,
            popsize: int = 20,
            maxiter: int = 150,
            tol: float = 0.01,
            dominance_resolution: int = 100,
            **kwargs,
        ):
            """
            Initializes the SurvivalAnalysis.

            Args:
                model_class: Wrapper class for survival model (default: RandomSurvivalForestWrapper).
                use_schedule (bool): If True, build a schedule using an optimization-based approach.
                                   If False, select the single best algorithm.
                max_schedule_length (Optional[int]): The maximum number of algorithms in a schedule.
                popsize (int): Population size for differential_evolution.
                maxiter (int): Max iterations for differential_evolution.
                tol (float): Tolerance for convergence for differential_evolution.
                dominance_resolution (int): Number of points for the time grid in dominance analysis.
                **kwargs: Additional arguments for the parent classes.

            Raises:
                ValueError: If budget is not a positive number.
            """
            super().__init__(model_class=model_class, **kwargs)
            self.use_schedule = use_schedule
            self.max_schedule_length = max_schedule_length
            self.popsize = popsize
            self.maxiter = maxiter
            self.tol = tol
            self.dominance_resolution = dominance_resolution

            if use_schedule:
                self.RETURN_TYPE = "schedule"

            if not isinstance(self.budget, (int, float)) or self.budget <= 0:
                raise ValueError(
                    "budget must be a positive number for survival analysis selector."
                )

        def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
            """
            Fits the Random Survival Forest model to the given data.

            Args:
                features (pd.DataFrame): DataFrame containing problem instance features.
                performance (pd.DataFrame): DataFrame where columns are algorithms and rows are instances.
                                            Values are runtimes, with NaN indicating a timeout.
            """

            # 1. Reshape and preprocess the data
            fit_data = []
            for instance in features.index:
                instance_features = features.loc[instance]
                for algo in self.algorithms:
                    runtime = performance.loc[instance, algo]
                    # Treat as timeout if runtime is missing or exceeds budget
                    finished = not pd.isna(runtime) and runtime < self.budget
                    status = int(finished)
                    runtime = runtime if finished else self.budget
                    row = {
                        **instance_features.to_dict(),
                        "algorithm": algo,
                        "runtime": runtime,
                        "status": status,
                    }
                    fit_data.append(row)
            fit_df = pd.DataFrame(fit_data)

            fit_features = pd.get_dummies(
                fit_df.drop(columns=["runtime", "status"]),
                columns=["algorithm"],
                prefix="algo",
            )

            # Store the feature column names for prediction
            self.survival_features = fit_features.columns.tolist()

            y_structured = Surv.from_arrays(
                event=fit_df["status"].astype(bool).values,
                time=fit_df["runtime"].values,
            )

            self.model = self.model_class()
            self.model.fit(fit_features, y_structured)

        def _predict(
            self, features: pd.DataFrame
        ) -> dict[str, list[tuple[str, float]]]:
            """
            Predicts the best algorithm for a new problem instance.

            Args:
                features (pd.DataFrame): DataFrame containing the feature data.

            Returns:
                Dict[str, List[Tuple[str, float]]]: A dictionary mapping instance names to the predicted
                best algorithm and the associated budget.

            Raises:
                ValueError: If the model has not been fitted yet.
            """
            if self.model is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")

            predictions = {}
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
                    # Original logic: find the single best algorithm
                    best_algo = None
                    best_prob = -1.0
                    for algo, surv_func in surv_funcs.items():
                        completion_prob = 1.0 - surv_func(self.budget)
                        if completion_prob > best_prob:
                            best_prob = completion_prob
                            best_algo = algo
                    predictions[instance] = [(best_algo, self.budget)]
                else:
                    # Build schedule using differential evolution
                    schedule = self._find_optimal_schedule(surv_funcs)
                    predictions[instance] = (
                        schedule if schedule else [(None, self.budget)]
                    )

            return predictions

        def _eval_schedule(
            self, x: np.ndarray, all_algos: list, surv_funcs: dict
        ) -> float:
            """
            Evaluates the sequential success probability of a schedule encoded by the optimizer's vector `x`.
            This is the fitness function for the optimizer.

            The vector `x` encodes inclusion flags and normalized end times for all algorithms.

            Args:
                x: The vector from the optimizer.
                all_algos: The complete list of algorithm names.
                surv_funcs: A dictionary mapping algorithm names to their survival functions.

            Returns:
                The negative success probability (since optimizers minimize).
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
                    time_slice = self.budget - last_actual_end_time
                else:
                    actual_end_time = norm_end_time * self.budget
                    time_slice = actual_end_time - last_actual_end_time

                if time_slice <= 1e-6:
                    continue

                surv_func = surv_funcs[algo]
                prob_solve_at_this_step = 1.0 - surv_func(time_slice)

                total_success_prob += prob_of_reaching_step * prob_solve_at_this_step
                prob_of_reaching_step *= surv_func(time_slice)

                last_actual_end_time += time_slice

            return -total_success_prob

        def _find_optimal_schedule(self, surv_funcs: dict) -> list:
            """
            Performs dominance analysis and then uses differential evolution
            to find the optimal schedule on the non-dominated set of algorithms.
            """
            # Dominance Analysis
            time_grid = np.linspace(0, self.budget, self.dominance_resolution)
            prob_matrix = np.array(
                [surv_funcs[algo](time_grid) for algo in self.algorithms]
            )

            # Case 1: Check for a single, globally dominant algorithm
            for i, algo in enumerate(self.algorithms):
                is_dominant = np.all(prob_matrix[i, :] <= prob_matrix)
                if is_dominant:
                    return [(algo, self.budget)]

            # Case 2: Filter out algorithms that are dominated by others
            lower_envelope = np.min(prob_matrix, axis=0)
            non_dominated_algos = []
            for i, algo in enumerate(self.algorithms):
                # An algorithm is non-dominated if its curve touches the lower envelope at any point
                if np.any(np.isclose(prob_matrix[i, :], lower_envelope)):
                    non_dominated_algos.append(algo)

            if len(non_dominated_algos) <= 1:
                if non_dominated_algos:
                    return [(non_dominated_algos[0], self.budget)]
                else:
                    return []

            # Optimization
            n_algorithms_to_optimize = len(non_dominated_algos)
            bounds = [(0, 1)] * (2 * n_algorithms_to_optimize)

            result = differential_evolution(
                func=self._eval_schedule,
                bounds=bounds,
                args=(non_dominated_algos, surv_funcs),
                popsize=self.popsize,
                maxiter=self.maxiter,
                tol=self.tol,
                seed=42,
            )

            # Post-process the best vector found by the optimizer
            best_x = result.x
            inclusion_flags = best_x[:n_algorithms_to_optimize]
            normalized_end_times = best_x[n_algorithms_to_optimize:]

            included_schedule_info = []
            for i, algo in enumerate(non_dominated_algos):
                if inclusion_flags[i] >= 0.5:
                    included_schedule_info.append((algo, normalized_end_times[i]))

            if not included_schedule_info:
                return []

            included_schedule_info.sort(key=lambda item: item[1])

            # Convert to the final schedule format with actual time slices
            schedule = []
            last_actual_end_time = 0.0
            for i, (algo, norm_end_time) in enumerate(included_schedule_info):
                # The last algorithm in the schedule runs for the remaining budget
                if i == len(included_schedule_info) - 1:
                    time_slice = self.budget - last_actual_end_time
                else:
                    actual_end_time = norm_end_time * self.budget
                    time_slice = actual_end_time - last_actual_end_time

                if time_slice > 1e-6:
                    schedule.append((algo, time_slice))
                last_actual_end_time += time_slice

            return schedule

        @staticmethod
        def _define_hyperparameters(model_class=None, **kwargs):
            """Define hyperparameters for SurvivalAnalysis."""
            if not CONFIGSPACE_AVAILABLE:
                return [], [], []

            if model_class is None:
                model_class = [RandomSurvivalForestWrapper]

            model_class_param = ClassChoice(
                name="model_class",
                choices=model_class,
                default=model_class[0],
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
            **kwargs,
        ) -> partial:
            """
            Create a partial function from a clean (unprefixed) configuration.
            """
            config = clean_config.copy()
            # If use_schedule is False, ConfigSpace might not return dependent params if they are inactive?
            # But ConfigurableMixin/ConfigSpace usually handles this.
            # We just pass the config.

            config.update(kwargs)
            return partial(SurvivalAnalysis, **config)

else:

    class SurvivalAnalysis:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sksurv is not installed. Please install sksurv to use SurvivalAnalysis."
            )
