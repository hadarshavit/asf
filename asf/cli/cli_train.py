#!/usr/bin/env python3
"""CLI entry point for training selectors."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn import preprocessing

from asf import presolving, selectors
from asf.predictors.linear_model import LinearClassifierWrapper, LinearRegressorWrapper
from asf.predictors.mlp import MLPClassifierWrapper, MLPRegressorWrapper
from asf.predictors.random_forest import (
    RandomForestClassifierWrapper,
    RandomForestRegressorWrapper,
)
from asf.predictors.ridge import RidgeRegressorWrapper
from asf.predictors.survival import RandomSurvivalForestWrapper
from asf.predictors.svm import SVMClassifierWrapper, SVMRegressorWrapper
from asf.predictors.xgboost import XGBoostClassifierWrapper, XGBoostRegressorWrapper
from asf.selectors import (
    AbstractModelBasedSelector,
    AbstractSelector,
    tune_selector,
)
from asf.selectors.selector_pipeline import SelectorPipeline


# Mapping of file extensions to pandas read functions
pandas_read_map: dict[str, Callable] = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".feather": pd.read_feather,
    ".hdf": pd.read_hdf,
    ".html": pd.read_html,
    ".xml": pd.read_xml,
}

model_list: dict[str, Any] = {
    "RandomForestClassifier": RandomForestClassifierWrapper,
    "RandomForestRegressor": RandomForestRegressorWrapper,
    "Ridge": RidgeRegressorWrapper,
    "RandomSurvivalForest": RandomSurvivalForestWrapper,
    "SVMClassifier": SVMClassifierWrapper,
    "SVMRegressor": SVMRegressorWrapper,
    "XGBoostClassifier": XGBoostClassifierWrapper,
    "XGBoostRegressor": XGBoostRegressorWrapper,
    "LinearClassifier": LinearClassifierWrapper,
    "LinearRegressor": LinearRegressorWrapper,
    "MLPClassifier": MLPClassifierWrapper,
    "MLPRegressor": MLPRegressorWrapper,
}


def _fraction_type(val: str) -> float:
    """
    Validate and convert string to float fraction.

    Parameters
    ----------
    val : str
        String to convert.

    Returns
    -------
    float
        Float between 0 and 1.

    Raises
    ------
    argparse.ArgumentTypeError
        If value is not a float between 0 and 1.
    """
    try:
        f = float(val)
    except Exception:
        raise argparse.ArgumentTypeError("must be a float between 0 and 1")
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return f


def parser_function() -> argparse.ArgumentParser:
    """
    Define command line arguments for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selectors",
        choices=selectors.__all__,
        required=True,
        nargs="+",
        help="Selector(s) to train. Chooses from multiple when used with --tuning",
    )
    parser.add_argument(
        "--tuning",
        action="store_true",
        default=False,
        help="Enable tuning mode: allow multiple selectors to be passed via --selector",
    )
    parser.add_argument(
        "--model",
        default="RandomForestClassifier",
        help="Model to use for the selector. "
        "Make sure to specify as an attribute of sklearn.ensemble.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        required=False,
        help="Budget for the solvers",
    )
    parser.add_argument(
        "--maximize",
        type=bool,
        default=False,
        required=False,
        help="Maximize the objective",
    )
    parser.add_argument(
        "--feature-data",
        type=Path,
        required=True,
        help="Path to feature data",
    )
    parser.add_argument(
        "--performance-data",
        type=Path,
        required=True,
        help="Path to performance data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to save model",
    )
    parser.add_argument(
        "--preprocessors",
        nargs="*",
        default=None,
        help="Preprocessors to apply, choose from Sklearn preprocessors",
    )
    parser.add_argument(
        "--presolvers",
        nargs="*",
        default=None,
        help="Presolvers to apply",
    )
    parser.add_argument(
        "--presolver-budget",
        type=_fraction_type,
        default=0.0,
        help="Fraction of total budget to allocate to presolving (float between 0 and 1).",
    )
    parser.add_argument(
        "--runcount-limit",
        type=int,
        default=100,
        help="Maximum number of SMAC evaluations when tuning is enabled (default: 100).",
    )
    return parser


def build_cli_command(
    selector: AbstractSelector
    | AbstractModelBasedSelector
    | list[AbstractSelector | AbstractModelBasedSelector],
    feature_data: Path,
    performance_data: Path,
    destination: Path,
    model: type | None = None,
    tuning: bool = False,
    budget: int | None = None,
    maximize: bool | None = None,
    preprocessors: list[type | Any] | None = None,
    presolvers: list[type | Any] | None = None,
    presolver_budget: float | None = None,
    runcount_limit: int | None = None,
) -> list[str]:
    """
    Build CLI command from selector objects.

    Parameters
    ----------
    selector : AbstractSelector or list
        Selector object(s) to use.
    feature_data : Path
        Path to feature data.
    performance_data : Path
        Path to performance data.
    destination : Path
        Path to save the model.
    model : type or None, default=None
        Model class to use.
    tuning : bool, default=False
        Whether tuning is enabled.
    budget : int or None, default=None
        Budget for solvers.
    maximize : bool or None, default=None
        Whether to maximize the objective.
    preprocessors : list or None, default=None
        List of preprocessor classes or objects.
    presolvers : list or None, default=None
        List of presolver classes or objects.
    presolver_budget : float or None, default=None
        Budget fraction for presolving.
    runcount_limit : int or None, default=None
        Limit on SMAC runs.

    Returns
    -------
    list[str]
        List of command line arguments.
    """
    if isinstance(selector, (list, tuple)):
        sel_list = list(selector)
    else:
        sel_list = [selector]

    for s in sel_list:
        if isinstance(s, type):
            ok = issubclass(s, (selectors.AbstractSelector, AbstractModelBasedSelector))
        else:
            ok = isinstance(s, (selectors.AbstractSelector, AbstractModelBasedSelector))
        if not ok:
            raise TypeError(
                "selector must be an AbstractSelector or AbstractModelBasedSelector "
                "class/instance, or a list/tuple of such objects"
            )

    sel_names = []
    for s in sel_list:
        sel_names.append(s.__name__ if isinstance(s, type) else type(s).__name__)

    cmd: list[str] = [
        "python",
        str(Path(__file__).absolute()),
        "--selectors",
    ] + sel_names
    if tuning:
        cmd.append("--tuning")

    model_name: str | None = None
    if model is not None:
        model_name = str(model.__name__) if hasattr(model, "__name__") else str(model)
    else:
        first = sel_list[0] if len(sel_list) > 0 else None
        if first is not None:
            try:
                model_attr = getattr(first, "model_class", None)
                mc = (
                    model_attr.args[0]
                    if isinstance(model_attr, partial)
                    else model_attr
                )
                if mc is not None and hasattr(mc, "__name__"):
                    model_name = str(mc.__name__)
            except Exception:
                pass
    if model_name is not None:
        cmd += ["--model", model_name]

    if budget is not None:
        cmd += ["--budget", str(budget)]
    else:
        try:
            first = sel_list[0] if len(sel_list) > 0 else None
            if first is not None:
                b = getattr(first, "budget", None)
                if b is not None:
                    cmd += ["--budget", str(b)]
        except Exception:
            pass

    if maximize is not None:
        cmd += ["--maximize", str(bool(maximize))]
    else:
        try:
            first = sel_list[0] if len(sel_list) > 0 else None
            if first is not None:
                cmd += ["--maximize", str(bool(getattr(first, "maximize", False)))]
        except Exception:
            pass

    cmd += [
        "--feature-data",
        str(feature_data),
        "--performance-data",
        str(performance_data),
        "--model-path",
        str(destination),
    ]

    if preprocessors and len(preprocessors) > 0:
        proc_names = [
            str(p.__name__ if isinstance(p, type) else type(p).__name__)
            for p in preprocessors
        ]
        cmd += ["--preprocessors"] + proc_names
    if presolvers and len(presolvers) > 0:
        pres_names = [
            str(p.__name__ if isinstance(p, type) else type(p).__name__)
            for p in presolvers
        ]
        cmd += ["--presolvers"] + pres_names
    if presolver_budget is not None:
        cmd += ["--presolver-budget", str(presolver_budget)]
    if runcount_limit is not None and tuning:
        cmd += ["--runcount-limit", str(runcount_limit)]

    return cmd


if __name__ == "__main__":
    parser = parser_function()
    args = parser.parse_args()

    if not args.tuning and len(args.selectors) != 1:
        parser.error(
            "When --tuning is not set, --selectors must contain exactly one selector"
        )

    if not args.tuning and args.presolvers and len(args.presolvers) > 1:
        parser.error(
            "When --tuning is not set, --presolvers can only contain maximum one presolver"
        )

    selector_names = args.selectors
    presolver_names = args.presolvers if args.presolvers is not None else []
    preprocessor_names = args.preprocessors if args.preprocessors is not None else []

    budget = args.budget
    presolver_ratio = args.presolver_budget
    selector_budget = budget
    presolver_budget = 0

    if presolver_ratio > 0.0:
        selector_budget = int(budget * (1.0 - presolver_ratio))
        presolver_budget = budget - selector_budget

    selector_classes = [getattr(selectors, name) for name in selector_names]
    print("Selector classes:", selector_classes)

    model_class = model_list[args.model]
    print("Model class:", model_class)

    if args.tuning:
        presolver_classes = [getattr(presolving, name) for name in presolver_names]
        preprocessing_steps = [
            getattr(preprocessing, name) for name in preprocessor_names
        ]
    else:
        presolver_classes = [
            getattr(presolving, name)(budget=presolver_budget / len(presolver_names))
            for name in presolver_names
        ]
        preprocessing_steps = [
            getattr(preprocessing, name)() for name in preprocessor_names
        ]

    print("Presolver classes:", presolver_classes)
    print("Presolver budget fraction:", presolver_ratio)
    print("Preprocessing classes:", preprocessing_steps)

    # Parse training data into variables
    features: pd.DataFrame = pandas_read_map[args.feature_data.suffix](
        args.feature_data, index_col=0
    )
    performance: pd.DataFrame = pandas_read_map[args.performance_data.suffix](
        args.performance_data, index_col=0
    )

    if not args.tuning:
        selector_class = selector_classes[0]
        if issubclass(selector_class, AbstractModelBasedSelector):
            selector = selector_class(
                model_class,
                maximize=args.maximize,
                budget=selector_budget,
            )
        elif issubclass(selector_class, AbstractSelector):
            selector = selector_class(
                maximize=args.maximize,
                budget=selector_budget,
            )
        else:
            raise TypeError(
                "Selector must be a subclass of AbstractSelector or AbstractModelBasedSelector"
            )

        presolver = presolver_classes[0] if len(presolver_classes) > 0 else None

        pipeline = SelectorPipeline(
            selector,
            pre_solving=presolver,
            preprocessor=preprocessing_steps if len(preprocessing_steps) > 0 else None,
        )

        pipeline.fit(features, performance)
        pipeline.save(args.model_path)

    else:
        # Create dummy features_running_time if not available
        dummy_rt = pd.DataFrame(
            0.0,
            index=features.index,
            columns=["total_feature_time"],  # type: ignore[arg-type]
        )

        selector = tune_selector(
            features,
            performance,
            selector_class=selector_classes,
            features_running_time=dummy_rt,
            budget=budget,
            runcount_limit=args.runcount_limit,
            preprocessing_class=preprocessing_steps
            if len(preprocessing_steps) > 0
            else None,
            pre_solving_class=presolver_classes if len(presolver_classes) > 0 else None,
        )

        selector.fit(features, performance)
        selector.save(args.model_path)
