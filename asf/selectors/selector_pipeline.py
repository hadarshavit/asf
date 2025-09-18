from typing import Optional, Callable, Any
from asf.selectors.abstract_selector import AbstractSelector
from asf.presolving.presolver import AbstractPresolver
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd


class SelectorPipeline:
    """
    A pipeline for applying a sequence of preprocessing, feature selection, and algorithm selection
    steps before fitting a final selector model.

    Attributes:
        selector (AbstractSelector): The main selector model to be used.
        preprocessor (Optional[Callable]): A callable for preprocessing the input data.
        pre_solving (Optional[Callable]): A callable for pre-solving steps.
        feature_selector (Optional[Callable]): A callable for feature selection.
        algorithm_pre_selector (Optional[Callable]): A callable for algorithm pre-selection.
        budget (Optional[Any]): The budget constraint for the selector.
        maximize (bool): Whether to maximize the objective function.
        feature_groups (Optional[Any]): Feature groups to be used by the selector.
    """

    def __init__(
        self,
        selector: AbstractSelector,
        preprocessor: Optional[Any] = None,
        pre_solving: AbstractPresolver = None,
        feature_selector: Optional[Callable] = None,
        algorithm_pre_selector: Optional[Callable] = None,
        budget: Optional[Any] = None,
        maximize: bool = False,
        feature_groups: Optional[Any] = None,
    ) -> None:
        """
        Initializes the SelectorPipeline.

        Args:
            selector (AbstractSelector): The main selector model to be used.
            preprocessor (Optional[Callable], optional): A callable for preprocessing the input data. Defaults to None.
            pre_solving (Optional[Callable], optional): A callable for pre-solving steps. Defaults to None.
            feature_selector (Optional[Callable], optional): A callable for feature selection. Defaults to None.
            algorithm_pre_selector (Optional[Callable], optional): A callable for algorithm pre-selection. Defaults to None.
            budget (Optional[Any], optional): The budget constraint for the selector. Defaults to None.
            maximize (bool, optional): Whether to maximize the objective function. Defaults to False.
            feature_groups (Optional[Any], optional): Feature groups to be used by the selector. Defaults to None.
        """
        self.selector = selector
        self.pre_solving = pre_solving
        self.feature_selector = feature_selector
        self.algorithm_pre_selector = algorithm_pre_selector
        self.budget = budget
        self.maximize = maximize

        # Always include SimpleImputer as the first step in the preprocessing pipeline
        if preprocessor is None:
            preprocessor = []
        elif not isinstance(preprocessor, list):
            preprocessor = [preprocessor]
        preprocessor = [SimpleImputer(strategy="mean")] + preprocessor
        steps = [(type(p).__name__, p) for p in preprocessor]
        self.preprocessor = Pipeline(steps)
        self.preprocessor.set_output(transform="pandas")

        self._orig_columns = None
        self._orig_index = None

    def fit(self, X: Any, y: Any) -> None:
        """
        Fits the pipeline to the input data.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        if isinstance(X, pd.DataFrame):
            self._orig_columns = X.columns
            self._orig_index = X.index

        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)

        if self.pre_solving:
            self.pre_solving.fit(X, y)

        if self.algorithm_pre_selector:
            y = self.algorithm_pre_selector.fit_transform(y)

        if self.feature_selector:
            X, y = self.feature_selector.fit_transform(X, y)

        self.selector.fit(X, y)

    def predict(self, X: Any) -> dict:
        """
        Makes predictions using the fitted pipeline.

        Args:
            X (Any): The input features.

        Returns:
            Any: The predictions made by the selector.
        """
        if self.preprocessor:
            X = self.preprocessor.transform(X)

        scheds = None
        if self.pre_solving:
            scheds = self.pre_solving.predict(X)

        if self.feature_selector:
            X = self.feature_selector.transform(X)

        predictions = self.selector.predict(X)

        # Ensure predictions use the same index as X
        predictions = pd.Series(predictions, index=X.index)
        if scheds is not None:
            for instance_id, pre_schedule in scheds.items():
                if instance_id in predictions:
                    predictions[instance_id] = pre_schedule + predictions[instance_id]
        return predictions.to_dict()

    def save(self, path: str) -> None:
        """
        Saves the pipeline to a file.

        Args:
            path (str): The file path where the pipeline will be saved.
        """
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SelectorPipeline":
        """
        Loads a pipeline from a file.

        Args:
            path (str): The file path from which the pipeline will be loaded.

        Returns:
            SelectorPipeline: The loaded pipeline.
        """
        import joblib

        return joblib.load(path)

    def get_config(self) -> dict:
        """
        Returns a dictionary with the configuration of the pipeline.

        Returns:
            dict: Configuration details of the pipeline.
        """

        def get_model_class_name(selector):
            if hasattr(selector, "model_class"):
                mc = selector.model_class
                # Handle functools.partial
                if hasattr(mc, "func"):
                    return mc.func.__name__
                elif hasattr(mc, "__name__"):
                    return mc.__name__
                else:
                    return str(type(mc))
            return None

        config = {
            "budget": self.budget,
            "selector": type(self.selector).__name__,
            "selector_model": get_model_class_name(self.selector),
            "pre_solving": type(self.pre_solving).__name__
            if self.pre_solving
            else None,
            "presolving_budget": getattr(self.pre_solving, "budget", None)
            if self.pre_solving
            else None,
            "preprocessor": type(self.preprocessor).__name__
            if self.preprocessor
            else None,
            "preprocessor_steps": [
                type(step[1]).__name__ for step in self.preprocessor.steps
            ]
            if hasattr(self.preprocessor, "steps")
            else None,
            "feature_selector": type(self.feature_selector).__name__
            if self.feature_selector
            else None,
            "algorithm_pre_selector": type(self.algorithm_pre_selector).__name__
            if self.algorithm_pre_selector
            else None,
        }
        return config
