from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from asf.presolving.presolver import AbstractPresolver
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
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
    Sequence of preprocessing, feature selection, and algorithm selection steps.

    Attributes
    ----------
    selector : AbstractSelector
        The main selector model to be used.
    pre_solving : AbstractPresolver or None
        A presolver for selecting initial algorithms.
    feature_selector : Any or None
        A component for feature selection.
    algorithm_pre_selector : Any or None
        A component for algorithm pre-selection.
    feature_groups : Any or None
        Feature groups to be used by the selector.
    max_feature_time : float or None
        Budget (seconds) to allocate per feature group in predictions.
    preprocessor : Pipeline
        The preprocessing pipeline (including SimpleImputer).
    """

    PREFIX = "pipeline"

    def __init__(
        self,
        selector: AbstractSelector,
        preprocessor: Any | list[Any] | None = None,
        pre_solving: AbstractPresolver | None = None,
        feature_selector: Any | None = None,
        algorithm_pre_selector: Any | None = None,
        feature_groups: dict[str, Any] | list[str] | None = None,
        max_feature_time: float | None = None,
    ) -> None:
        """
        Initialize the SelectorPipeline.

        Parameters
        ----------
        selector : AbstractSelector
            The main selector model to be used.
        preprocessor : Any or list or None, default=None
            Preprocessing steps. SimpleImputer(strategy="mean") is always added first.
        pre_solving : AbstractPresolver or None, default=None
            Presolver for initial algorithm selection.
        feature_selector : Any or None, default=None
            Component for feature selection.
        algorithm_pre_selector : Any or None, default=None
            Component for algorithm pre-selection.
        feature_groups : dict or list or None, default=None
            Feature groups configuration.
        max_feature_time : float or None, default=None
            Budget (seconds) per feature group.
        """
        self.selector = selector
        self.pre_solving = pre_solving
        self.feature_selector = feature_selector
        self.algorithm_pre_selector = algorithm_pre_selector
        self.feature_groups = feature_groups
        self.max_feature_time = (
            float(max_feature_time) if max_feature_time is not None else None
        )

        if preprocessor is None:
            preproc_list = []
        elif not isinstance(preprocessor, list):
            preproc_list = [preprocessor]
        else:
            preproc_list = preprocessor

        # Always include SimpleImputer as the first step
        steps = [("SimpleImputer", SimpleImputer(strategy="mean"))]
        for p in preproc_list:
            steps.append((type(p).__name__, p))

        self.preprocessor = Pipeline(steps)
        self.preprocessor.set_output(transform="pandas")

        self._logger = logging.getLogger(__name__)

    def _filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features based on selected feature groups.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.

        Returns
        -------
        pd.DataFrame
            Filtered features.
        """
        if self.feature_groups and isinstance(self.feature_groups, dict):
            selected_features = []
            for fg_info in self.feature_groups.values():
                if isinstance(fg_info, dict) and "provides" in fg_info:
                    selected_features.extend(fg_info["provides"])

            available_features = [f for f in selected_features if f in X.columns]
            if available_features:
                return X[available_features]
        return X

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the pipeline.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        algorithm_features : pd.DataFrame or None, default=None
            Optional algorithm features.
        **kwargs : Any
            Additional keyword arguments.
        """
        start = time.time()
        self._logger.debug("Starting fit process")

        X = self.preprocessor.fit_transform(features, performance)
        self._logger.debug(
            f"Preprocessing completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        # Update y (performance) alias for local usage
        y = performance

        if self.algorithm_pre_selector:
            if hasattr(self.algorithm_pre_selector, "fit_transform"):
                y = self.algorithm_pre_selector.fit_transform(X, y)  # type: ignore
            else:
                self.algorithm_pre_selector.fit(X, y)  # type: ignore
                # Some pre-selectors might not have transform for y, checking usage
                if hasattr(self.algorithm_pre_selector, "transform"):
                    y = self.algorithm_pre_selector.transform(y)  # type: ignore

        self._logger.debug(
            f"Algorithm pre-selection completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        if self.pre_solving:
            self.pre_solving.fit(features, performance)

        self._logger.debug(
            f"Pre-solving completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        if self.feature_selector:
            if hasattr(self.feature_selector, "fit_transform"):
                X, y = self.feature_selector.fit_transform(X, y)  # type: ignore
            else:
                self.feature_selector.fit(X, y)  # type: ignore
                X = self.feature_selector.transform(X)  # type: ignore

        self._logger.debug(
            f"Feature selection completed in {time.time() - start:.2f} seconds"
        )
        start = time.time()

        X = self._filter_features(X)
        self.selector.fit(X, y, algorithm_features=algorithm_features, **kwargs)

        self._logger.debug(
            f"Selector fitting completed in {time.time() - start:.2f} seconds"
        )

    def predict(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, list[tuple[str, float] | tuple[str, float, float]]]:
        """
        Make predictions.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame or None, default=None
            Performance data for oracle selectors.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        dict
            Predictions mapping instance IDs to schedules.
        """
        X = self.preprocessor.transform(features)

        scheds: list[Any] = []
        if self.pre_solving:
            # Presolver prediction returns a single schedule applied to all test instances
            scheds = list(self.pre_solving.predict())

        if self.feature_selector:
            X = self.feature_selector.transform(X)  # type: ignore

        X = self._filter_features(X)

        # Pass performance to selector (needed for oracle selectors like VBS)
        predictions = self.selector.predict(X, performance=performance)

        feature_steps: list[Any] = []
        if self.feature_groups is not None:
            if isinstance(self.feature_groups, dict):
                feature_steps = list(self.feature_groups.keys())
            elif isinstance(self.feature_groups, list):
                feature_steps = self.feature_groups

        if self.max_feature_time is not None and feature_steps:
            feature_steps = [
                (str(fg), float(self.max_feature_time)) for fg in feature_steps
            ]

        final_preds: dict[str, list[tuple[str, float] | tuple[str, float, float]]] = {}
        for instance_id in X.index:
            prediction = predictions.get(str(instance_id), [])  # type: ignore
            final_preds[str(instance_id)] = scheds + feature_steps + list(prediction)

        return final_preds

    def save(self, path: str | Path) -> None:
        """
        Save the pipeline to a file.

        Parameters
        ----------
        path : str or Path
            File path to save the pipeline.
        """
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> SelectorPipeline:
        """
        Load a pipeline from a file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        SelectorPipeline
            The loaded pipeline.
        """
        import joblib

        return joblib.load(path)

    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details.

        Returns
        -------
        dict
            Configuration metadata.
        """

        def get_model_name(obj: Any) -> str | None:
            if obj is None:
                return None
            if hasattr(obj, "model_class"):
                mc = obj.model_class
                if hasattr(mc, "func"):
                    return str(mc.func.__name__)
                return str(getattr(mc, "__name__", type(mc).__name__))
            return type(obj).__name__

        return {
            "selector": type(self.selector).__name__,
            "selector_model": get_model_name(self.selector),
            "pre_solving": type(self.pre_solving).__name__
            if self.pre_solving
            else None,
            "selector_budget": getattr(self.selector, "budget", None),
            "presolving_budget": getattr(self.pre_solving, "budget", None),
            "preprocessor_steps": [
                type(step[1]).__name__ for step in self.preprocessor.steps
            ],
            "feature_selector": type(self.feature_selector).__name__
            if self.feature_selector
            else None,
            "algorithm_pre_selector": type(self.algorithm_pre_selector).__name__
            if self.algorithm_pre_selector
            else None,
        }

    @staticmethod
    def _define_hyperparameters(
        selector_class: list[type] | None = None,
        preprocessing_class: list[type] | None = None,
        pre_solving_class: list[type] | None = None,
        feature_groups: dict[str, Any] | None = None,
        algorithm_pre_selector: type | tuple[type, dict[str, Any]] | None = None,
        max_feature_time: float | None | bool = False,
        budget: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for SelectorPipeline.

        Parameters
        ----------
        selector_class : list[type] or None, default=None
            List of selector classes.
        preprocessing_class : list[type] or None, default=None
            List of preprocessor classes.
        pre_solving_class : list[type] or None, default=None
            List of presolver classes.
        feature_groups : dict or None, default=None
            Feature groups definition.
        algorithm_pre_selector : type or tuple or None, default=None
            Algorithm pre-selector class.
        max_feature_time : float or None or bool, default=False
            Maximum feature computation time.
        budget : float or None, default=None
            Total budget.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = []
        conditions = []
        forbiddens = []

        if selector_class:
            if (
                isinstance(selector_class, list)
                and selector_class
                and isinstance(selector_class[0], tuple)
            ):
                selector_choices = [
                    c[0]  # type: ignore[index]
                    for c in (
                        selector_class if isinstance(selector_class, list) else []
                    )
                ]  # type: ignore
            else:
                selector_choices = (
                    selector_class
                    if isinstance(selector_class, list)
                    else [selector_class]
                )
            hyperparameters.append(ClassChoice("selector", choices=selector_choices))

        if pre_solving_class:
            ps_choices = (
                pre_solving_class
                if isinstance(pre_solving_class, list)
                else [pre_solving_class]
            )
            use_presolver = Categorical(
                "use_presolver", items=[True, False], default=False
            )
            hyperparameters.append(use_presolver)
            presolver_choice = ClassChoice("presolver", choices=ps_choices)
            hyperparameters.append(presolver_choice)
            conditions.append(EqualsCondition(presolver_choice, use_presolver, True))  # type: ignore

        if preprocessing_class:
            for preproc_cls in preprocessing_class:
                hyperparameters.append(
                    ClassChoice(
                        f"preprocessor:{preproc_cls.__name__}",
                        choices=[preproc_cls, False],
                        default=False,
                    )
                )

        fg_params = {}
        if feature_groups and len(feature_groups) > 1:
            for fg_name in feature_groups:
                fg_param = Categorical(
                    f"feature_group:{fg_name}", [True, False], default=True
                )
                hyperparameters.append(fg_param)
                fg_params[fg_name] = fg_param

            for fg_name, fg_info in feature_groups.items():
                for req in fg_info.get("requires", []):
                    if req in fg_params:
                        forbiddens.append(
                            ForbiddenAndConjunction(
                                ForbiddenEqualsClause(fg_params[fg_name], True),
                                ForbiddenEqualsClause(fg_params[req], False),
                            )
                        )
            forbiddens.append(
                ForbiddenAndConjunction(
                    *[ForbiddenEqualsClause(p, False) for p in fg_params.values()]
                )
            )

        if algorithm_pre_selector:
            aps_cls = (
                algorithm_pre_selector[0]
                if isinstance(algorithm_pre_selector, tuple)
                else algorithm_pre_selector
            )
            hyperparameters.append(
                ClassChoice("algorithm_pre_selector", choices=[aps_cls])
            )

        if max_feature_time is None:
            upper = float(budget or 3600.0)
            hyperparameters.append(
                UniformFloatHyperparameter(
                    "max_feature_time",
                    lower=0.0,
                    upper=upper,
                    default_value=min(60.0, upper),
                )
            )

        return hyperparameters, conditions, forbiddens

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        configuration: Configuration | dict[str, Any] | None = None,
        pre_prefix: str = "",
        feature_groups: dict[str, Any] | None = None,
        max_feature_time: float | None = None,
        **kwargs: Any,
    ) -> partial[SelectorPipeline]:
        """
        Create a SelectorPipeline from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            Clean configuration mapping.
        configuration : Configuration or dict or None, default=None
            Original configuration.
        pre_prefix : str, default=""
            Prefix for nested lookups.
        feature_groups : dict or None, default=None
            Feature groups definition.
        max_feature_time : float or None, default=None
            Maximum feature computation time.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for SelectorPipeline.
        """
        init_kwargs: dict[str, Any] = {}
        prefix = f"{pre_prefix}:{cls.PREFIX}:" if pre_prefix else f"{cls.PREFIX}:"

        if "selector" in clean_config:
            val = clean_config["selector"]
            init_kwargs["selector"] = val() if callable(val) else val

        if clean_config.get("use_presolver") and "presolver" in clean_config:
            val = clean_config["presolver"]
            init_kwargs["pre_solving"] = val() if callable(val) else val

        preprocs = []
        for k, v in clean_config.items():
            if k.startswith("preprocessor:") and v and v != "False":
                preprocs.append(v() if callable(v) else v)
        if preprocs:
            init_kwargs["preprocessor"] = preprocs

        if feature_groups and configuration is not None:
            from asf.preprocessing.feature_group_selector import FeatureGroupSelector

            init_kwargs["feature_groups"] = (
                FeatureGroupSelector.get_selected_groups_from_config(
                    feature_groups, configuration, prefix=f"{prefix}feature_group:"
                )
            )

        if "algorithm_pre_selector" in clean_config:
            val = clean_config["algorithm_pre_selector"]
            init_kwargs["algorithm_pre_selector"] = val() if callable(val) else val

        init_kwargs["max_feature_time"] = clean_config.get(
            "max_feature_time", max_feature_time
        )

        return partial(cls, **init_kwargs)

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict[str, Any],
        pre_prefix: str = "",
        feature_groups: dict[str, Any] | None = None,
        budget: float | None = None,
        max_feature_time: float | None = None,
        **kwargs: Any,
    ) -> partial[SelectorPipeline]:
        """
        Create a SelectorPipeline from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            Configuration object.
        pre_prefix : str, default=""
            Prefix for nested lookups.
        feature_groups : dict or None, default=None
            Feature groups definition.
        budget : float or None, default=None
            Total budget.
        max_feature_time : float or None, default=None
            Maximum feature computation time.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for SelectorPipeline.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError("ConfigSpace is not available.")

        prefix = f"{pre_prefix}:{cls.PREFIX}:" if pre_prefix else f"{cls.PREFIX}:"
        clean_config: dict[str, Any] = {}
        cs = getattr(configuration, "config_space", None)

        def resolve(hp_name: str, val: Any) -> Any:
            if cs and val:
                hp = cs.get(hp_name)
                if hp:
                    return cls._resolve_class_from_hp(hp, str(val))
            return None

        # 1. Selector
        s_val = configuration.get(f"{prefix}selector")
        s_cls = resolve(f"{prefix}selector", s_val)
        if s_cls is None and callable(s_val):
            s_cls = s_val

        if s_cls:
            if hasattr(s_cls, "get_from_configuration"):
                clean_config["selector"] = s_cls.get_from_configuration(
                    configuration,
                    pre_prefix=f"{prefix}selector",
                    budget=budget,
                    **kwargs,
                )
            else:
                clean_config["selector"] = s_cls(budget=budget) if budget else s_cls()

        # 2. Presolver
        use_ps = configuration.get(f"{prefix}use_presolver")
        clean_config["use_presolver"] = use_ps
        ps_val = configuration.get(f"{prefix}presolver")
        ps_cls = resolve(f"{prefix}presolver", ps_val)
        if use_ps and ps_cls:
            if hasattr(ps_cls, "get_from_configuration"):
                clean_config["presolver"] = ps_cls.get_from_configuration(
                    configuration, pre_prefix=f"{prefix}presolver", **kwargs
                )
            else:
                clean_config["presolver"] = ps_cls()

        # 3. Preprocessors
        if cs:
            for hp in list(cs.values()):
                if hp.name.startswith(f"{prefix}preprocessor:"):
                    val = configuration.get(hp.name)
                    if val and val != "False":
                        res = cls._resolve_class_from_hp(hp, str(val))
                        if res:
                            key = hp.name[len(prefix) :]
                            clean_config[key] = (
                                res.get_from_configuration(  # type: ignore
                                    configuration, pre_prefix=hp.name, **kwargs
                                )
                                if hasattr(res, "get_from_configuration")
                                else (res() if callable(res) else res)
                            )
                    else:
                        clean_config[hp.name[len(prefix) :]] = False

        # 4. Algorithm Pre-selector
        aps_val = configuration.get(f"{prefix}algorithm_pre_selector")
        aps_cls = resolve(f"{prefix}algorithm_pre_selector", aps_val)
        if aps_cls:
            if hasattr(aps_cls, "get_from_configuration"):
                clean_config["algorithm_pre_selector"] = aps_cls.get_from_configuration(
                    configuration,
                    pre_prefix=f"{prefix}algorithm_pre_selector",
                    **kwargs,
                )
            else:
                clean_config["algorithm_pre_selector"] = aps_cls()

        # 5. Max feature time
        mft = configuration.get(f"{prefix}max_feature_time")
        if mft is not None:
            clean_config["max_feature_time"] = mft

        return cls._get_from_clean_configuration(
            clean_config=clean_config,
            configuration=configuration,
            pre_prefix=pre_prefix,
            feature_groups=feature_groups,
            max_feature_time=max_feature_time,
            **kwargs,
        )
