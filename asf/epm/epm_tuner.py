"""
EPM tuning logic using SMAC.
"""

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

try:
    from smac import HyperparameterOptimizationFacade, Scenario

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False

from asf.epm.epm import EPM
from asf.predictors.abstract_predictor import AbstractPredictor
from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle


def tune_epm(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model_class: type[AbstractPredictor],
    normalization_class: type[AbstractNormalization] = LogNormalization,
    features_preprocessing: str | TransformerMixin | None = "default",
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    groups: np.ndarray | None = None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed: int = 0,
    smac_metric: Callable[[np.ndarray, np.ndarray], float] = mean_squared_error,
    smac_scenario_kwargs: dict[str, Any] | None = None,
    smac_kwargs: dict[str, Any] | None = None,
    predictor_kwargs: dict[str, Any] | None = None,
) -> EPM:
    """
    Tune the Empirical Performance Model (EPM) using SMAC.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix for training and validation.
    y : np.ndarray or pd.Series
        Target values corresponding to the feature matrix.
    model_class : type[AbstractPredictor]
        The predictor class to be tuned.
    normalization_class : type[AbstractNormalization], default=LogNormalization
        The normalization class to be applied to the data.
    features_preprocessing : str or TransformerMixin, default="default"
        Preprocessing method for features.
    categorical_features : list or None, default=None
        List of categorical feature names.
    numerical_features : list or None, default=None
        List of numerical feature names.
    groups : np.ndarray or None, default=None
        Group labels for cross-validation.
    cv : int, default=5
        Number of cross-validation folds.
    timeout : int, default=3600
        Time limit for the tuning process in seconds.
    runcount_limit : int, default=100
        Maximum number of configurations to evaluate.
    output_dir : str, default="./smac_output"
        Directory to store SMAC output.
    seed : int, default=0
        Random seed for reproducibility.
    smac_metric : Callable, default=mean_squared_error
        Metric function to evaluate model performance.
    smac_scenario_kwargs : dict or None, default=None
        Additional keyword arguments for the SMAC scenario.
    smac_kwargs : dict or None, default=None
        Additional keyword arguments for SMAC optimization.
    predictor_kwargs : dict or None, default=None
        Additional keyword arguments for the predictor.

    Returns
    -------
    EPM
        The tuned Empirical Performance Model instance.

    Raises
    ------
    RuntimeError
        If SMAC is not installed.
    """
    if not SMAC_AVAILABLE:
        raise RuntimeError("SMAC is not installed. Install it with: pip install smac")

    smac_scenario_kwargs = smac_scenario_kwargs or {}
    smac_kwargs = smac_kwargs or {}
    predictor_kwargs = predictor_kwargs or {}

    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X_df = pd.DataFrame(
            X,
            index=range(len(X)),
            columns=[f"f_{i}" for i in range(X.shape[1])],  # type: ignore[arg-type]
        )
        y_ser = pd.Series(
            y,
            index=range(len(y)),
        )
    else:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y

    scenario = Scenario(
        configspace=model_class.get_configuration_space(),
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=Path(output_dir),
        seed=seed,
        **smac_scenario_kwargs,
    )

    def target_function(config: Any, seed: int) -> float:
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X_df, y_ser, groups):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_ser.iloc[train_idx], y_ser.iloc[test_idx]

            epm = EPM(
                predictor_class=model_class,
                normalization_class=normalization_class,
                transform_back=True,
                predictor_config=config,
                predictor_kwargs=predictor_kwargs,
                features_preprocessing=features_preprocessing,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
            epm.fit(X_train, y_train)

            y_pred = epm.predict(X_test)
            score = smac_metric(y_test.values, y_pred)
            scores.append(score)

        return float(np.mean(scores))

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    # handle Union[Configuration, list]
    if isinstance(best_config, list):
        best_config = best_config[0]

    return EPM(
        predictor_class=model_class,
        normalization_class=normalization_class,
        transform_back=True,
        predictor_config=dict(best_config),
        features_preprocessing=features_preprocessing,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
