import numpy as np
import pandas as pd
from typing import cast

from asf.presolving.static_3s import Static3S
from asf.selectors.sunny import SUNNY
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_simple_data(n_instances=100, n_algorithms=6, seed=1, budget=500.0):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.uniform(0, 10, size=(n_instances, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(10, 200) * (1 + 0.5 * (a % 2))
        noise = rng.normal(0, 30, size=n_instances)
        runtimes = np.maximum(5.0, bias + features["f0"] * rng.uniform(-5, 5) + noise)
        timeout_mask = rng.rand(n_instances) < (0.10 + 0.05 * (a % 3))
        runtimes[timeout_mask] = budget * 2
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def main():
    budget = 400.0
    presolve_budget = 150.0
    features, performance = generate_simple_data(
        n_instances=100, n_algorithms=6, seed=2, budget=budget
    )

    # split train / test
    n_train = int(0.7 * len(features))
    train_X = features.iloc[:n_train]
    train_Y = performance.iloc[:n_train]
    test_X = features.iloc[n_train:]
    test_Y = performance.iloc[n_train:]

    # Use ASF metrics for baselines
    sbs_score = single_best_solver(test_Y, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(test_Y, maximize=False, budget=budget, par=10.0)

    # 1) compute static preschedule
    presolver = Static3S(budget=presolve_budget, max_candidates_per_solver=10)
    presolver.fit(train_X, train_Y)

    preschedule_map = presolver.predict(test_X)

    presolver_sr = compute_solve_rate(
        cast(dict, preschedule_map), test_Y, presolve_budget
    )
    presolver_par10 = running_time_selector_performance(
        cast(dict, preschedule_map),
        test_Y,
        budget=presolve_budget,
        par=10.0,
        return_per_instance=False,
    )

    presolver_solved = []
    presolver_unsolved = []

    for inst in test_Y.index:
        schedule_for_inst = cast(dict, preschedule_map).get(inst)
        if schedule_for_inst and inst in test_Y.index:
            # Check if any algorithm in schedule solves within its allocated time
            algo_solved = False
            for algo, t in schedule_for_inst:
                rt = test_Y.loc[inst, algo]
                if not pd.isna(rt) and float(rt) <= float(t):
                    algo_solved = True
                    break
            if algo_solved:
                presolver_solved.append(inst)
            else:
                presolver_unsolved.append(inst)
        else:
            presolver_unsolved.append(inst)

    # 2) train SUNNY on train set and apply to instances not solved by presolver
    sunny = SUNNY(k=5, use_v2=True, budget=budget, k_candidates=[3, 5, 7])
    sunny.fit(train_X, train_Y)
    if presolver_unsolved:
        preds = sunny.predict(test_X.loc[presolver_unsolved])
    else:
        preds = {}

    sunny_sr = (
        compute_solve_rate(cast(dict, preds), test_Y.loc[presolver_unsolved], budget)
        if presolver_unsolved
        else 0.0
    )
    sunny_par10 = (
        running_time_selector_performance(
            cast(dict, preds),
            test_Y.loc[presolver_unsolved],
            budget=budget,
            par=10.0,
            return_per_instance=False,
        )
        if presolver_unsolved
        else 0.0
    )

    sunny_solved = []
    sunny_unsolved = []
    for inst in presolver_unsolved:
        sched = cast(dict, preds).get(inst, [])
        if sched and inst in test_Y.index:
            algo_solved = False
            for algo, t in sched:
                rt = test_Y.loc[inst, algo]
                if not pd.isna(rt) and float(rt) <= float(t):
                    algo_solved = True
                    break
            if algo_solved:
                sunny_solved.append(inst)
            else:
                sunny_unsolved.append(inst)
        else:
            sunny_unsolved.append(inst)

    # Output summary
    print("=" * 60)
    print("Static3S presolver + SUNNYv2 example")
    print("=" * 60)
    print(f"Presolve budget: {presolve_budget}s")
    print(f"Budget: {budget}s")
    print(f"Train instances: {len(train_X)}  Test instances: {len(test_X)}")
    print()
    print(f"Single Best Solver (SBS) PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10 Score: {vbs_score:.2f}")
    print()
    print("Presolver schedule (solver, time):")
    print(presolver.schedule)
    print(f"Presolver solve rate: {presolver_sr:.2%}")
    print(f"Presolver PAR10 score: {presolver_par10:.2f}")
    print()
    print(f"SUNNY solve rate (on presolver-unsolved): {sunny_sr:.2%}")
    print(f"SUNNY PAR10 score (on presolver-unsolved): {sunny_par10:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
