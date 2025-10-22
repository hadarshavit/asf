import pandas as pd

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
        """

        PREFIX = "survival"

        def __init__(
            self,
            model_class: type[
                RandomSurvivalForestWrapper
            ] = RandomSurvivalForestWrapper,
            **kwargs,
        ):
            """
            Initializes the SurvivalAnalysisSelector.

            Args:
                model_class: Wrapper class for survival model (default: RandomSurvivalForestWrapper).
                **kwargs: Additional arguments for the parent classes.

            Raises:
                ValueError: If budget is not a positive number.
            """
            super().__init__(model_class=model_class, **kwargs)

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
                best_algo = None
                best_prob = -1.0

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

                    surv_func = self.model.predict_survival_function(pred_row)[0]
                    completion_prob = 1.0 - surv_func(self.budget)

                    if completion_prob > best_prob:
                        best_prob = completion_prob
                        best_algo = algo

                predictions[instance] = [(best_algo, self.budget)]

            return predictions

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
