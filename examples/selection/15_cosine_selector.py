import os
from typing import cast

from asf.selectors.cosine_selector import CosineSelector
from asf.scenario.aslib_reader import read_aslib_scenario
from asf.utils.aslib_algorithm_features import get_algorithm_features_from_aslib
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


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

    # AS-LLM based CosineSelector
    sel = CosineSelector(
        normalize_features=True,
        embed_size=50,
        num_hiddens=50,
        num_layers=2,
        alpha=0.9,
        beta=0.1,
        num_epochs=50,  # Reduced for demo
        batch_size=128,
        lr=0.001,
    )
    sel.fit(X_train, Y_train, algorithm_features=alg_df)

    preds = sel.predict(X_test)
    preds = cast(dict[str, list[tuple[str, float] | str]], preds)
    sr = compute_solve_rate(preds, Y_test, budget)
    par10 = running_time_selector_performance(
        preds, Y_test, budget=budget, par=10.0, return_per_instance=False
    )

    best = ((Y_train <= budget).mean(axis=0)).idxmax()
    baseline_preds = {idx: [(best, budget)] for idx in X_test.index}
    base_sr = compute_solve_rate(baseline_preds, Y_test, budget)

    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    oracle_sr = float((Y_test.min(axis=1) <= budget).mean())

    print("=" * 60)
    print("CosineSelector - AS-LLM Architecture (real ASLib data)")
    print("=" * 60)
    print(f"Scenario: {aslib_scenario_dir}")
    print(
        f"Algorithms (sample): {list(alg_df.index)[:8]}{'...' if len(alg_df.index) > 8 else ''}"
    )
    print(f"Train / Test: {len(X_train)} / {len(X_test)}  Budget: {budget}")
    print()
    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) solve-rate: {oracle_sr:.2%}")
    print()
    print(f"Cosine selector solve-rate: {sr:.2%}")
    print(f"Cosine selector PAR10 Score: {par10:.2f}")
    print(f"Best-single ({best}) solve-rate: {base_sr:.2%}")
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
