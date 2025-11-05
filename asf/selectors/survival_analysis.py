import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.predictors.survival import RandomSurvivalForestWrapper, SKSURV_AVAILABLE

if SKSURV_AVAILABLE:
    from sksurv.util import Surv

    try:
        from ConfigSpace import (
            ConfigurationSpace,
            Categorical,
            Configuration,
            EqualsCondition,
        )
        from ConfigSpace.hyperparameters import Hyperparameter

        CONFIGSPACE_AVAILABLE = True
    except ImportError:
        CONFIGSPACE_AVAILABLE = False

    class SurvivalAnalysisSelector(AbstractModelBasedSelector):
        """
        Selects the best algorithm for a given problem instance using survival analysis.
        Tries to maximize the probability of finishing within a given time budget.
        Can optionally build a schedule of multiple algorithms using RunAndSchedule2Survive logic.
        """

        PREFIX = "survival"
        RETURN_TYPE = "mixed"

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
            Initializes the SurvivalAnalysisSelector.

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

        if CONFIGSPACE_AVAILABLE:

            @staticmethod
            def get_configuration_space(
                cs: ConfigurationSpace | None = None,
                cs_transform: dict[str, dict] | None = None,
                model_class: list[type] | None = None,
                pre_prefix: str = "",
                parent_param: Hyperparameter | None = None,
                parent_value: str | None = None,
                **kwargs,
            ) -> tuple[ConfigurationSpace, dict[str, dict]]:
                """
                Get the configuration space for SurvivalAnalysisSelector.

                Args:
                    cs: The configuration space to use. If None, a new one will be created.
                    cs_transform: A dictionary for transforming configuration space parameters.
                    model_class: List of survival model wrapper classes to choose from.
                    pre_prefix: Prefix for parameter names.
                    parent_param: Parent parameter for conditional configuration.
                    parent_value: Value of parent parameter that activates these hyperparameters.
                    **kwargs: Additional keyword arguments.

                Returns:
                    Tuple[ConfigurationSpace, Dict[str, dict]]: The configuration space and its transformation dictionary.
                """
                if cs is None:
                    cs = ConfigurationSpace()

                if cs_transform is None:
                    cs_transform = dict()

                if model_class is None:
                    model_class = [RandomSurvivalForestWrapper]

                if pre_prefix != "":
                    prefix = f"{pre_prefix}:{SurvivalAnalysisSelector.PREFIX}"
                else:
                    prefix = SurvivalAnalysisSelector.PREFIX

                model_class_param = Categorical(
                    name=f"{prefix}:model_class",
                    items=[str(c.__name__) for c in model_class],
                )

                cs_transform[f"{prefix}:model_class"] = {
                    str(c.__name__): c for c in model_class
                }

                params = [model_class_param]

                if parent_param is not None:
                    conditions = [
                        EqualsCondition(
                            child=param,
                            parent=parent_param,
                            value=parent_value,
                        )
                        for param in params
                    ]
                else:
                    conditions = []

                cs.add(params + conditions)

                for mc in model_class:
                    mc.get_configuration_space(
                        cs=cs,
                        pre_prefix=f"{prefix}:model_class",
                        parent_param=model_class_param,
                        parent_value=str(mc.__name__),
                        **kwargs,
                    )

                return cs, cs_transform

            @staticmethod
            def get_from_configuration(
                configuration: Configuration,
                cs_transform: dict[str, dict],
                pre_prefix: str = "",
                **kwargs,
            ) -> "SurvivalAnalysisSelector":
                """
                Get the SurvivalAnalysisSelector from a given configuration.

                Args:
                    configuration: The configuration object.
                    cs_transform: The transformation dictionary for the configuration space.
                    pre_prefix: Prefix for parameter names.
                    **kwargs: Additional keyword arguments for SurvivalAnalysisSelector initialization.

                Returns:
                    SurvivalAnalysisSelector: An instance configured according to the given configuration.
                """
                if pre_prefix != "":
                    prefix = f"{pre_prefix}:{SurvivalAnalysisSelector.PREFIX}"
                else:
                    prefix = SurvivalAnalysisSelector.PREFIX

                model_cls = cs_transform[f"{prefix}:model_class"][
                    configuration[f"{prefix}:model_class"]
                ]
                model_ctor = model_cls.get_from_configuration(
                    configuration, pre_prefix=f"{prefix}:model_class"
                )

                return SurvivalAnalysisSelector(
                    model_class=model_ctor,
                    **kwargs,
                )

else:

    class SurvivalAnalysisSelector:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sksurv is not installed. Please install sksurv to use SurvivalAnalysisSelector."
            )
