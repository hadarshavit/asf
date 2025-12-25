from functools import partial
from typing import Any, Callable
import time
import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from asf.presolving.presolver import AbstractPresolver
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    from ConfigSpace import (
        Categorical,
        Configuration,
        EqualsCondition,
        ForbiddenAndConjunction,
        ForbiddenEqualsClause,
        UniformFloatHyperparameter,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SelectorPipeline(ConfigurableMixin):
    """
    A pipeline for applying a sequence of preprocessing, feature selection, and algorithm selection
    steps before fitting a final selector model.

    Attributes:
        selector (AbstractSelector): The main selector model to be used.
    preprocessor (Callable | None): A callable for preprocessing the input data.
    pre_solving (Callable | None): A callable for pre-solving steps.
    feature_selector (Callable | None): A callable for feature selection.
    algorithm_pre_selector (Callable | None): A callable for algorithm pre-selection.
    feature_groups (Any | None): Feature groups to be used by the selector.
    """

    PREFIX = "pipeline"

    def __init__(
        self,
        selector: AbstractSelector,
        preprocessor: Any | None = None,
        pre_solving: AbstractPresolver | None = None,
        feature_selector: Callable | None = None,
        algorithm_pre_selector: Callable | None = None,
        feature_groups: Any | None = None,
        max_feature_time: float | None = None,
    ) -> None:
        """
        Initializes the SelectorPipeline.

        Args:
            selector (AbstractSelector): The main selector model to be used.
            preprocessor (Callable | None, optional): A callable for preprocessing the input data. Defaults to None.
            pre_solving (Callable | None, optional): A callable for pre-solving steps. Defaults to None.
            feature_selector (Callable | None, optional): A callable for feature selection. Defaults to None.
            algorithm_pre_selector (Callable | None, optional): A callable for algorithm pre-selection. Defaults to None.
            feature_groups (Any | None, optional): Feature groups to be used by the selector. Defaults to None.
            max_feature_time (float | None, optional): Budget (seconds) to allocate per feature group in predictions. Defaults to None.
        """
        self.selector = selector
        self.pre_solving = pre_solving
        self.feature_selector = feature_selector
        self.algorithm_pre_selector = algorithm_pre_selector
        self.feature_groups = feature_groups
        # Optional budget (seconds) to allocate per feature group
        self.max_feature_time = max_feature_time

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

        self._logger = logging.getLogger(__name__)

    def _filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filters features based on selected feature groups.
        """
        if self.feature_groups and isinstance(self.feature_groups, dict):
            selected_features = []
            for fg_info in self.feature_groups.values():
                if "provides" in fg_info:
                    selected_features.extend(fg_info["provides"])

            # Only keep features that are in X
            available_features = [f for f in selected_features if f in X.columns]
            if available_features:
                return X[available_features]
        return X

    def fit(self, X: Any, y: Any, algorithm_features: Any = None) -> None:
        """
        Fits the pipeline to the input data.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        if isinstance(X, pd.DataFrame):
            self._orig_columns = X.columns
            self._orig_index = X.index

        start = time.time()
        self._logger.debug("Starting fit process")
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        self._logger.debug(
            f"Preprocessing completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        # Algorithm pre-selection should happen BEFORE pre-solving
        # so that the presolver only considers the pre-selected algorithms
        if self.algorithm_pre_selector:
            y = self.algorithm_pre_selector.fit_transform(y)

        self._logger.debug(
            f"Algorithm pre-selection completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        if self.pre_solving:
            self.pre_solving.fit(X, y)

        self._logger.debug(
            f"Pre-solving completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        if self.feature_selector:
            X, y = self.feature_selector.fit_transform(X, y)

        self._logger.debug(
            f"Feature selection completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        X = self._filter_features(X)

        self.selector.fit(X, y, algorithm_features=algorithm_features)

        self._logger.debug(
            f"Selector fitting completed in {time.time() - start:.2f} seconds"
        )

    def predict(self, X: Any, performance: Any = None) -> dict:
        """
        Makes predictions using the fitted pipeline.

        Args:
            X (Any): The input features.
            performance (Any, optional): The performance data for oracle selectors. Defaults to None.

        Returns:
            Any: The predictions made by the selector.
        """
        if self.preprocessor:
            X = self.preprocessor.transform(X)

        scheds = None
        if self.pre_solving:
            scheds = self.pre_solving.predict()

        if self.feature_selector:
            X = self.feature_selector.transform(X)

        X = self._filter_features(X)

        # Pass performance through to selector (needed for oracle selectors like VBS)
        predictions = self.selector.predict(X, performance=performance)

        # Ensure predictions use the same index as X
        predictions = pd.Series(predictions, index=X.index)

        feature_steps = []
        if self.feature_groups is not None:
            if isinstance(self.feature_groups, dict):
                feature_steps = list(self.feature_groups.keys())
            elif isinstance(self.feature_groups, list):
                feature_steps = self.feature_groups

        # Convert feature_steps to budgeted tuples if max_feature_time is set
        if self.max_feature_time is not None and feature_steps:
            feature_steps_with_budget = [
                (fg, self.max_feature_time) for fg in feature_steps
            ]
        else:
            feature_steps_with_budget = feature_steps

        final_preds = {}
        for instance_id, prediction in predictions.items():
            # If pre-solving schedules are provided, prepend/concatenate them
            # to the algorithm predictions for each instance. When no pre-solving
            # schedule is used `scheds` will be None and we should still return
            # the selector's predictions.
            if scheds is not None:
                final_preds[instance_id] = (
                    scheds + feature_steps_with_budget + prediction
                )
            else:
                final_preds[instance_id] = feature_steps_with_budget + prediction

        return final_preds

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
            "selector": type(self.selector).__name__,
            "selector_model": get_model_class_name(self.selector),
            "pre_solving": type(self.pre_solving).__name__
            if self.pre_solving
            else None,
            "selector_budget": self.selector.budget if self.selector else None,
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

    @staticmethod
    def _define_hyperparameters(
        selector_class: list[type] = None,
        preprocessing_class: list[type] | None = None,
        pre_solving_class: list[type] | None = None,
        feature_groups: dict | None = None,
        algorithm_pre_selector: type | tuple[type, dict] | None = None,
        max_feature_time: float | None | bool = False,
        budget: float | None = None,
        **kwargs,
    ):
        """
        Define hyperparameters for the SelectorPipeline.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = []
        conditions = []
        forbiddens = []

        # 1. Selector Selection
        if selector_class:
            # Handle list of tuples (class, kwargs)
            if (
                isinstance(selector_class, list)
                and len(selector_class) > 0
                and isinstance(selector_class[0], tuple)
            ):
                selector_choices = [c[0] for c in selector_class]
            elif not isinstance(selector_class, list):
                selector_choices = [selector_class]
            else:
                selector_choices = selector_class

            # Use ClassChoice for automatic recursion
            hyperparameters.append(ClassChoice("selector", choices=selector_choices))

        # 2. Presolver Selection
        if pre_solving_class:
            if not isinstance(pre_solving_class, list):
                pre_solving_class = [pre_solving_class]

            # Allow "None" as a choice by using a separate boolean switch
            use_presolver = Categorical(
                "use_presolver", items=[True, False], default=False
            )
            hyperparameters.append(use_presolver)

            presolver_choice = ClassChoice("presolver", choices=pre_solving_class)
            hyperparameters.append(presolver_choice)

            # Condition: presolver active only if use_presolver is True
            conditions.append(EqualsCondition(presolver_choice, use_presolver, True))

        # 3. Preprocessors
        if preprocessing_class:
            for preproc_cls in preprocessing_class:
                hp = ClassChoice(
                    f"preprocessor:{preproc_cls.__name__}",
                    choices=[preproc_cls, False],
                    default=False,
                )
                hyperparameters.append(hp)

        # 4. Feature Groups
        fg_params = {}
        if feature_groups and len(feature_groups) > 1:
            for fg_name in feature_groups.keys():
                fg_param = Categorical(
                    f"feature_group:{fg_name}", [True, False], default=True
                )
                hyperparameters.append(fg_param)
                fg_params[fg_name] = fg_param

            # Forbiddens for prerequisites
            for fg_name, fg_info in feature_groups.items():
                required_groups = fg_info.get("requires", [])
                for required_group in required_groups:
                    if required_group in fg_params:
                        # Forbid: current=True AND required=False
                        forbidden = ForbiddenAndConjunction(
                            ForbiddenEqualsClause(fg_params[fg_name], True),
                            ForbiddenEqualsClause(fg_params[required_group], False),
                        )
                        forbiddens.append(forbidden)

            # Forbid all False
            all_false = ForbiddenAndConjunction(
                *[ForbiddenEqualsClause(param, False) for param in fg_params.values()]
            )
            forbiddens.append(all_false)

        # 5. Algorithm Pre-selector
        if algorithm_pre_selector:
            if isinstance(algorithm_pre_selector, tuple):
                pre_sel_cls = algorithm_pre_selector[0]
            else:
                pre_sel_cls = algorithm_pre_selector

            # Just one option usually, but wrapped in ClassChoice allows recursion
            hyperparameters.append(
                ClassChoice("algorithm_pre_selector", choices=[pre_sel_cls])
            )

        if max_feature_time is None:
            assert budget is not None
            upper = float(budget)
            mf_param = UniformFloatHyperparameter(
                "max_feature_time",
                lower=0.0,
                upper=upper,
                default_value=min(60.0, upper),
                log=False,
            )
            hyperparameters.append(mf_param)

        return hyperparameters, conditions, forbiddens

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict,
        pre_prefix: str = "",
        selector_class: list[type] = None,
        preprocessing_class: list[type] | None = None,
        pre_solving_class: list[type] | None = None,
        feature_groups: dict | None = None,
        algorithm_pre_selector: type | tuple[type, dict] | None = None,
        max_feature_time: float | None = None,
        budget: float | None = None,
        **kwargs,
    ) -> partial:
        """
        Create a SelectorPipeline from a configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError("ConfigSpace is not installed.")

        # Compute prefix
        if pre_prefix:
            prefix = f"{pre_prefix}:{cls.PREFIX}:"
        else:
            prefix = f"{cls.PREFIX}:"

        init_kwargs = {}

        # 1. Selector
        # Try to resolve selector class from config space if selector_class is not provided
        chosen_cls = None
        selector_name = configuration.get(f"{prefix}selector")

        if selector_class:
            # Re-normalize selector_class to list of classes
            if (
                isinstance(selector_class, list)
                and len(selector_class) > 0
                and isinstance(selector_class[0], tuple)
            ):
                selector_choices = [c[0] for c in selector_class]
            elif not isinstance(selector_class, list):
                selector_choices = [selector_class]
            else:
                selector_choices = selector_class

            selector_map = {c.__name__: c for c in selector_choices}
            if selector_name and selector_name in selector_map:
                chosen_cls = selector_map[selector_name]
        elif selector_name and hasattr(configuration, "config_space"):
            hp = configuration.config_space.get(f"{prefix}selector")
            if hp:
                chosen_cls = cls._resolve_class_from_hp(hp, selector_name)

        if chosen_cls:
            child_pre_prefix = f"{prefix}selector"
            if hasattr(chosen_cls, "get_from_configuration"):
                child_kwargs = kwargs.copy()
                if budget is not None:
                    child_kwargs["budget"] = budget

                val_partial = chosen_cls.get_from_configuration(
                    configuration=configuration,
                    pre_prefix=child_pre_prefix,
                    **child_kwargs,
                )
                init_kwargs["selector"] = val_partial()
            else:
                selector_init_kwargs = {}
                if budget is not None:
                    selector_init_kwargs["budget"] = budget
                init_kwargs["selector"] = chosen_cls(**selector_init_kwargs)

        # 2. Presolver
        use_presolver = configuration.get(f"{prefix}use_presolver")
        presolver_name = configuration.get(f"{prefix}presolver")
        chosen_cls = None

        if use_presolver and presolver_name:
            if pre_solving_class:
                if not isinstance(pre_solving_class, list):
                    pre_solving_class = [pre_solving_class]
                presolver_map = {c.__name__: c for c in pre_solving_class}
                if presolver_name in presolver_map:
                    chosen_cls = presolver_map[presolver_name]
            elif hasattr(configuration, "config_space"):
                hp = configuration.config_space.get(f"{prefix}presolver")
                if hp:
                    chosen_cls = cls._resolve_class_from_hp(hp, presolver_name)

        if chosen_cls:
            child_pre_prefix = f"{prefix}presolver"
            if hasattr(chosen_cls, "get_from_configuration"):
                val_partial = chosen_cls.get_from_configuration(
                    configuration=configuration,
                    pre_prefix=child_pre_prefix,
                    **kwargs,
                )
                init_kwargs["pre_solving"] = val_partial()
            else:
                init_kwargs["pre_solving"] = chosen_cls()

        # 3. Preprocessors
        selected_preprocessors = []
        if preprocessing_class:
            for preproc_cls in preprocessing_class:
                key = f"{prefix}preprocessor:{preproc_cls.__name__}"
                val = configuration.get(key)
                if val == preproc_cls.__name__:
                    selected_preprocessors.append(preproc_cls())
        elif hasattr(configuration, "config_space"):
            # Discover preprocessors from space
            for hp in configuration.config_space.get_hyperparameters():
                if hp.name.startswith(f"{prefix}preprocessor:"):
                    val = configuration.get(hp.name)
                    resolved = cls._resolve_class_from_hp(hp, val)
                    if resolved and resolved is not False:
                        if hasattr(resolved, "get_from_configuration"):
                            selected_preprocessors.append(
                                resolved.get_from_configuration(
                                    configuration, pre_prefix=hp.name, **kwargs
                                )()
                            )
                        else:
                            selected_preprocessors.append(resolved())

        if selected_preprocessors:
            init_kwargs["preprocessor"] = selected_preprocessors

        # 4. Feature Groups
        if feature_groups:
            # Reconstruct selected groups dict
            from asf.preprocessing.feature_group_selector import FeatureGroupSelector

            selected_fg = FeatureGroupSelector.get_selected_groups_from_config(
                feature_groups, configuration, prefix=f"{prefix}feature_group:"
            )
            init_kwargs["feature_groups"] = selected_fg
        elif hasattr(configuration, "config_space"):
            # Discover feature groups from space if not provided explicitly?
            # FeatureGroupSelector.get_selected_groups_from_config needs the full feature_groups dict
            # because it contains the 'provides' and 'requires' info.
            # If it's missing, we only have names from the config.
            # For now, we keep this as is, as feature_groups is usually passed.
            pass

        # 5. Algorithm Pre-selector
        chosen_cls = None
        if algorithm_pre_selector:
            if isinstance(algorithm_pre_selector, tuple):
                chosen_cls = algorithm_pre_selector[0]
            else:
                chosen_cls = algorithm_pre_selector
        elif hasattr(configuration, "config_space"):
            hp = configuration.config_space.get(f"{prefix}algorithm_pre_selector")
            val = configuration.get(f"{prefix}algorithm_pre_selector")
            if hp and val:
                chosen_cls = cls._resolve_class_from_hp(hp, val)

        if chosen_cls:
            child_pre_prefix = f"{prefix}algorithm_pre_selector"
            if hasattr(chosen_cls, "get_from_configuration"):
                val_partial = chosen_cls.get_from_configuration(
                    configuration=configuration, pre_prefix=child_pre_prefix, **kwargs
                )
                init_kwargs["algorithm_pre_selector"] = val_partial()
            else:
                init_kwargs["algorithm_pre_selector"] = chosen_cls()

        if max_feature_time is None:
            mft_val = configuration.get(f"{prefix}max_feature_time")
            if mft_val is not None:
                init_kwargs["max_feature_time"] = mft_val
        elif max_feature_time is not False and isinstance(
            max_feature_time, (int, float)
        ):
            init_kwargs["max_feature_time"] = max_feature_time

        return partial(cls, **init_kwargs)
