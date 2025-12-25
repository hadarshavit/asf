import os
import numpy as np
import pandas as pd
from typing import Any, cast
from asf.predictors.random_forest import RandomForestRegressorWrapper

from asf.selectors.cosine_selector import CosineSelector
from asf.scenario.aslib_reader import read_aslib_scenario
from asf.utils.aslib_algorithm_features import get_algorithm_features_from_aslib


def evaluate_solve_rate(preds: Any, perf: pd.DataFrame, budget: float) -> float:
    solved = 0
    total = 0
    for inst, rec in preds.items():
        algo, _ = rec[0]
        if algo is None:
            continue
        total += 1
        rt = perf.at[inst, algo]
        if not np.isnan(rt) and float(rt) <= budget:
            solved += 1
    return solved / total if total > 0 else 0.0


def main(aslib_scenario_dir: str = "aslib_data/SAT11-INDU-ALGO"):
    if not os.path.isdir(aslib_scenario_dir):
        raise FileNotFoundError(
            f"ASLib scenario directory not found: {aslib_scenario_dir}"
        )

    (
        features,
        performance,
        _,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    ) = read_aslib_scenario(aslib_scenario_dir)
    budget = budget / 2

    alg_df = get_algorithm_features_from_aslib(aslib_scenario_dir)

    common_idx = features.index.intersection(performance.index)
    features = features.loc[common_idx].sort_index()
    performance = performance.loc[common_idx].sort_index()

    n_train = int(0.7 * len(features))
    X_train, X_test = features.iloc[:n_train], features.iloc[n_train:]
    Y_train, Y_test = performance.iloc[:n_train], performance.iloc[n_train:]

    sel = CosineSelector(
        normalize_features=True,
        shared_latent_dim=4,
        projection_model=RandomForestRegressorWrapper,
        projection_model_kwargs={"n_estimators": 100, "random_state": 42},
    )
    sel.fit(X_train, Y_train, algorithm_features=alg_df)

    preds = sel.predict(X_test)
    sr = evaluate_solve_rate(preds, Y_test, budget)

    best = ((Y_train <= budget).mean(axis=0)).idxmax()
    baseline_preds = {idx: [(best, budget)] for idx in X_test.index}
    base_sr = evaluate_solve_rate(baseline_preds, Y_test, budget)

    oracle_hits = Y_test.min(axis=1) <= budget
    oracle_sr = float(oracle_hits.mean())

    print("=" * 60)
    print("CosineSelector (real ASLib data)")
    print("=" * 60)
    print(f"Scenario: {aslib_scenario_dir}")
    print(
        f"Algorithms (sample): {list(alg_df.index)[:8]}{'...' if len(alg_df.index) > 8 else ''}"
    )
    print(f"Train / Test: {len(X_train)} / {len(X_test)}  Budget: {budget}")
    print()
    print(f"Cosine selector solve-rate: {sr:.2%}")
    print(f"Best-single ({best}) solve-rate: {base_sr:.2%}")
    print(f"Oracle solve-rate: {oracle_sr:.2%}")
    print()
    print("Sample decisions (first 12):")
    for inst in list(X_test.index)[:12]:
        algo, score = cast(dict, preds).get(inst, [(None, None)])[0]
        rt = Y_test.at[inst, algo] if algo is not None else float("nan")
        print(f"{inst}: chosen={algo} predicted_score={score:.4f} true_rt={rt:.2f}")
    print("=" * 60)

    return sel, X_test, Y_test, preds


if __name__ == "__main__":
    main()
