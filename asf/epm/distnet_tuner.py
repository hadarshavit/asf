import numpy as np
import pandas as pd
from typing import Callable

from sklearn.model_selection import KFold

try:
    from smac import HyperparameterOptimizationFacade, Scenario

    SMAC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SMAC_AVAILABLE = False

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Float,
        Integer,
        OrdinalHyperparameter,
    )
except Exception:  # pragma: no cover - optional dependency
    ConfigurationSpace = None
    Categorical = None
    Float = None
    Integer = None
    OrdinalHyperparameter = None

import torch

from asf.epm.distnet import DistNet

from asf.predictors.utils.losses import lognorm_loss
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle


def tune_distnet(
    model: type[DistNet],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    features_preprocessing: str | object = "default",
    categorical_features: list | None = None,
    numerical_features: list | None = None,
    groups: np.ndarray | None = None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed: int = 0,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    smac_scenario_kwargs: dict | None = None,
    smac_kwargs: dict | None = None,
    distnet_kwargs: dict | None = None,
) -> DistNet:
    """Tune DistNet with SMAC and return a configured (unfitted) instance.

    Parameters
    ----------
    X, y : array-like
        Training features and targets. If array-like, they will be converted to DataFrame/Series for CV splitting.
    features_preprocessing : str | TransformerMixin
        Passed through to DistNet/AbstractEPM.
    categorical_features, numerical_features : list | None
        Passed through to DistNet/AbstractEPM.
    groups : array-like | None
        Optional group labels for GroupKFold-style CV.
    cv : int
        Number of folds.
    timeout : int
        Walltime budget in seconds for SMAC.
    runcount_limit : int
        Max number of trials.
    output_dir : str
        SMAC output directory.
    seed : int
        Random seed for SMAC and CV split.
    metric : callable or None
        If provided, a function f(y_true_tensor, pred_params_tensor) -> loss_tensor.
        Defaults to lognorm_loss if None.
    smac_scenario_kwargs, smac_kwargs : dict | None
        Extra kwargs forwarded to SMAC Scenario/facade.
    distnet_kwargs : dict | None
        Extra kwargs forwarded to DistNet constructor (e.g., device).

    Returns
    -------
    DistNet
        A DistNet instance constructed with the best found configuration (not fitted).
    """
    assert SMAC_AVAILABLE, (
        "SMAC is not installed. Please install it to use this function."
    )

    smac_scenario_kwargs = smac_scenario_kwargs or {}
    smac_kwargs = smac_kwargs or {}
    distnet_kwargs = distnet_kwargs or {}

    # Ensure pandas containers for robust indexing in CV
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(
            X, index=range(len(X)), columns=[f"f_{i}" for i in range(X.shape[1])]
        )
    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=range(len(y)))

    cs = model.get_configuration_space()

    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    loss_fn = metric or lognorm_loss

    def target_function(config, seed):
        # Cross-validation setup
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        fold_losses: list[float] = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            dn = model.get_from_configuration(
                input_size=X.shape[1],
                config=config,
                extra_kwargs={
                    "features_preprocessing": features_preprocessing,
                    "categorical_features": categorical_features,
                    "numerical_features": numerical_features,
                    **distnet_kwargs,
                },
            )

            # Fit
            dn.fit(X_train, y_train)

            # Predict params and compute loss on raw targets
            with torch.no_grad():
                preds = dn.predict(X_test)  # Tensor [N, 2]
                y_true = torch.tensor(
                    y_test.to_numpy()
                    if isinstance(y_test, pd.Series)
                    else np.asarray(y_test),
                    dtype=torch.float32,
                )
                loss_val = loss_fn(y_true, preds).item()
            fold_losses.append(loss_val)

        return float(np.mean(fold_losses))

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    # Build best DistNet (unfitted), mirroring epm_tuner behavior
    best_dn = model.get_from_configuration(
        input_size=X.shape[1],
        config=best_config,
        extra_kwargs={
            "features_preprocessing": features_preprocessing,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            **distnet_kwargs,
        },
    )

    return best_dn
