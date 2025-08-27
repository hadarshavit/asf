import pandas as pd
from typing import Dict, List, Tuple, Optional
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class _DummyModel:
    """Dummy model class to satisfy AbstractModelBasedSelector requirements."""
    pass


class SurvivalAnalysisSelector(AbstractModelBasedSelector):
    """
    Selects the best algorithm for a given problem instance using survival analysis.
    Tries to maximize the probability of finishing within a given time budget.
    """

    def __init__(
        self,
        cutoff_time: float,
        model_params: Optional[Dict] = None,
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
        # Pass a dummy model class since we'll manage the survival model ourselves
        super().__init__(model_class=_DummyModel, **kwargs)

        if not isinstance(cutoff_time, (int, float)) or cutoff_time <= 0:
            raise ValueError("cutoff_time must be a positive number.")
            
        self.cutoff_time = cutoff_time
        self.model_params = model_params if model_params is not None else {}
        self.survival_model: RandomSurvivalForest = None
        self.algorithms: List[str] = []
        self.feature_columns: List[str] = []


    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the Random Survival Forest model to the given data.

        Args:
            features (pd.DataFrame): DataFrame containing problem instance features.
            performance (pd.DataFrame): DataFrame where columns are algorithms and rows are instances.
                                        Values are runtimes, with NaN indicating a timeout.
        """
        self.algorithms = list(performance.columns)

        # 1. Reshape and preprocess the data
        fit_data = []
        for instance in features.index:
            instance_features = features.loc[instance]
            for algo in self.algorithms:
                runtime = performance.loc[instance, algo]
                # Treat as timeout if runtime is missing or exceeds cutoff
                finished = not pd.isna(runtime) and runtime < self.cutoff_time
                status = int(finished)
                runtime = runtime if finished else self.cutoff_time
                row = {**instance_features.to_dict(), 'algorithm': algo, 'runtime': runtime, 'status': status}
                fit_data.append(row)
        fit_df = pd.DataFrame(fit_data)
        
        # 2. One-hot encode the 'algorithm' feature
        fit_features = pd.get_dummies(fit_df.drop(columns=['runtime', 'status']), columns=['algorithm'], prefix='algo')
        
        # Store the feature columns for prediction alignment
        self.feature_columns = list(fit_features.columns)
        
        # 3. Create the structured array for the survival target
        y_structured = Surv.from_arrays(
            event=fit_df['status'].astype(bool).values,
            time=fit_df['runtime'].values
        )
        
        # 4. Instantiate and fit the model
        self.survival_model = RandomSurvivalForest(
            n_estimators=100,
            random_state=42,
            **self.model_params
        )
        self.survival_model.fit(fit_features, y_structured)


    def _predict(self, features: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
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
        if self.survival_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        predictions = {}
        # Ensure all algorithms are in the prediction loop
        for instance, instance_features in features.iterrows():
            best_algo = None
            best_prob = -1.0
            
            # Get the survival curve for each algorithm
            for algo in self.algorithms:
                # 1. Prepare the input for prediction
                pred_row = pd.DataFrame([{**instance_features.to_dict(), 'algorithm': algo}])
                pred_row = pd.get_dummies(pred_row, columns=['algorithm'], prefix='algo')
                pred_row = pred_row.reindex(columns=self.feature_columns, fill_value=0)

                # 2. Predict the survival probability at the cutoff time
                surv_func = self.survival_model.predict_survival_function(pred_row)[0]
                prob = 1.0 - surv_func(self.cutoff_time)

                # 3. Apply the decision policy
                if prob > best_prob:
                    best_prob = prob
                    best_algo = algo
            
            predictions[instance] = [(best_algo, self.cutoff_time)]

        return predictions