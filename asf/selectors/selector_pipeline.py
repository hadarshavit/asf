from typing import Optional, Callable, Any
from asf.selectors.abstract_selector import AbstractSelector
from asf.presolving.presolver import AbstractPresolver


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
        preprocessor: Optional[Callable] = None,
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
        self.preprocessor = preprocessor
        self.pre_solving = pre_solving
        self.feature_selector = feature_selector
        self.algorithm_pre_selector = algorithm_pre_selector
        self.budget = budget
        self.maximize = maximize

    def fit(self, X: Any, y: Any) -> None:
        """
        Fits the pipeline to the input data.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)

        if self.pre_solving:
            self.pre_solving.fit(X, y)

        if self.algorithm_pre_selector:
            y = self.algorithm_pre_selector.fit_transform(y)

        if self.feature_selector:
            X, y = self.feature_selector.fit_transform(X, y)

        self.selector.fit(X, y)

    def predict(self, X: Any) -> Any:
        """
        Makes predictions using the fitted pipeline.

        Args:
            X (Any): The input features.

        Returns:
            Any: The predictions made by the selector.
        """
        if self.preprocessor:
            X = self.preprocessor.transform(X)

        if self.pre_solving:
            scheds = self.pre_solving.predict()

        if self.feature_selector:
            X = self.feature_selector.transform(X)

        predictions = self.selector.predict(X)
        pre_solver_schedule = scheds["default"] if self.pre_solving else None

        if isinstance(predictions, dict):
            predictions["pre_solver_schedule"] = pre_solver_schedule
            return predictions
        else:
            return (predictions, pre_solver_schedule)

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
