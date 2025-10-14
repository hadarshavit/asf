import pandas as pd
from typing import Dict, List, Tuple

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.predictors.survival import RandomSurvivalForestWrapper, SKSURV_AVAILABLE

if SKSURV_AVAILABLE:
    from sksurv.util import Surv

    class SurvivalAnalysisSelector(AbstractModelBasedSelector):
        """
        Selects the best algorithm for a given problem instance using survival analysis.
        Tries to maximize the probability of finishing within a given time budget.
        """

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
                cutoff_time (float): The time budget for the decision-making policy.
                model_params (Optional[Dict]): Parameters for the Random Survival Forest model.
                **kwargs: Additional arguments for the parent classes.

            Raises:
                ValueError: If cutoff_time is not a positive number.
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
        ) -> Dict[str, List[Tuple[str, float]]]:
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
else:

    class SurvivalAnalysisSelector:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sksurv is not installed. Please install sksurv to use SurvivalAnalysisSelector."
            )
