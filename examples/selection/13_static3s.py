import numpy as np
import pandas as pd

from asf.presolving.static_3s import Static3S
from asf.selectors.sunny import SUNNY


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


def run_schedule(schedule, perf_row, budget):
    """
    schedule: list[tuple[algo, time]] or empty list
    perf_row: Series of runtimes for that instance
    """
    for algo, t in schedule:
        rt = perf_row.get(algo)
        if pd.isna(rt):
            continue
        if float(rt) <= float(t):
            return True
    return False


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

    # 1) compute static preschedule
    presolver = Static3S(budget=presolve_budget, max_candidates_per_solver=10)
    presolver.fit(train_X, train_Y)

    preschedule_map = presolver.predict(test_X)

    presolver_solved = []
    presolver_unsolved = []

    for inst in test_Y.index:
        schedule_for_inst = preschedule_map.get(inst)
        solved = run_schedule(schedule_for_inst, test_Y.loc[inst], presolve_budget)
        if solved:
            presolver_solved.append(inst)
        else:
            presolver_unsolved.append(inst)

    # 2) train SUNNY on train set and apply to instances not solved by presolver
    sunny = SUNNY(k=5, use_v2=True, budget=budget, k_candidates=[3, 5, 7])
    sunny.fit(train_X, train_Y)
    if presolver_unsolved:
        preds = sunny.predict(test_X.loc[presolver_unsolved])
    else:
        preds = {}

    sunny_solved = []
    sunny_unsolved = []
    for inst in presolver_unsolved:
        sched = preds.get(inst, [])
        solved = run_schedule(sched, test_Y.loc[inst], budget)
        if solved:
            sunny_solved.append(inst)
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
    print("Presolver schedule (solver, time):")
    print(presolver.schedule)
    print()
    print(f"Instances solved by presolver: {len(presolver_solved)}")
    print(
        ", ".join(presolver_solved[:20])
        + ("" if len(presolver_solved) <= 20 else ", ...")
    )
    print()
    print(
        f"Instances handled by SUNNY (unsolved by presolver): {len(presolver_unsolved)}"
    )
    print(f" - solved by SUNNY: {len(sunny_solved)}")
    print(
        "   "
        + ", ".join(sunny_solved[:20])
        + ("" if len(sunny_solved) <= 20 else ", ...")
    )
    print(f" - still unsolved: {len(sunny_unsolved)}")
    print(
        "   "
        + ", ".join(sunny_unsolved[:20])
        + ("" if len(sunny_unsolved) <= 20 else ", ...")
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
