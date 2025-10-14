from typing import Type, Union, Optional

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error  # Fixed incorrect import
from sklearn.model_selection import KFold
from smac import HyperparameterOptimizationFacade, Scenario
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle

from asf.epm.epm import EPM
from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.predictors.abstract_predictor import AbstractPredictor


def tune_epm(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Type[AbstractPredictor],
    normalization_class: Type[AbstractNormalization] = LogNormalization,
    features_preprocessing: Union[str, TransformerMixin] = "default",
    categorical_features: Optional[list] = None,
    numerical_features: Optional[list] = None,
    groups: Optional[np.ndarray] = None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed: int = 0,
    smac_metric: callable = mean_squared_error,  # Fixed incorrect import
    smac_scenario_kwargs: Optional[dict] = {},
    smac_kwargs: Optional[dict] = {},
    predictor_kwargs: Optional[dict] = {},
) -> EPM:
    """
    Tune the Empirical Performance Model (EPM) using SMAC (Sequential Model-based Algorithm Configuration).

    Parameters:
    ----------
    X : np.ndarray
        Feature matrix for training and validation.
    y : np.ndarray
        Target values corresponding to the feature matrix.
    model_class : Type[AbstractPredictor]
        The predictor class to be tuned.
    normalization_class : Type[AbstractNormalization], optional
        The normalization class to be applied to the data. Defaults to LogNormalization.
    features_preprocessing : Union[str, TransformerMixin], optional
        Preprocessing method for features. Defaults to "default".
    categorical_features : Optional[list], optional
        List of categorical feature names. Defaults to None.
    numerical_features : Optional[list], optional
        List of numerical feature names. Defaults to None.
    groups : Optional[np.ndarray], optional
        Group labels for cross-validation. Defaults to None.
    cv : int, optional
        Number of cross-validation folds. Defaults to 5.
    timeout : int, optional
        Time limit for the tuning process in seconds. Defaults to 3600.
    runcount_limit : int, optional
        Maximum number of configurations to evaluate. Defaults to 100.
    output_dir : str, optional
        Directory to store SMAC output. Defaults to "./smac_output".
    seed : int, optional
        Random seed for reproducibility. Defaults to 0.
    smac_metric : callable, optional
        Metric function to evaluate model performance. Defaults to mean_squared_error.
    smac_scenario_kwargs : Optional[dict], optional
        Additional keyword arguments for the SMAC scenario. Defaults to None.
    smac_kwargs : Optional[dict], optional
        Additional keyword arguments for SMAC optimization. Defaults to None.
    predictor_kwargs : Optional[dict], optional
        Additional keyword arguments for the predictor. Defaults to None.

    Returns:
    -------
    EPM
        The tuned Empirical Performance Model instance.
    """
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X = pd.DataFrame(
            X,
            index=range(len(X)),
            columns=[f"f_{i}" for i in range(X.shape[1])],
        )
        y = pd.Series(
            y,
            index=range(len(y)),
        )

    scenario = Scenario(
        configspace=model_class.get_configuration_space(),
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    def target_function(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
            score = smac_metric(y_test, y_pred)
            scores.append(score)

        return np.mean(scores)

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    return EPM(
        predictor_class=model_class,
        normalization_class=normalization_class,
        transform_back=True,
        predictor_config=best_config,
        features_preprocessing=features_preprocessing,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
