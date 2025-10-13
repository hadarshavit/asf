import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from asf.selectors.satzilla import SATzilla


def generate_data(n_instances=200, n_algorithms=6, seed=0, budget=60.0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, size=(n_instances, 3))
    features = pd.DataFrame(
        X, columns=["f1", "f2", "f3"], index=[f"inst_{i}" for i in range(n_instances)]
    )

    # per-algorithm base runtime and sensitivity to features
    base = rng.uniform(20, 80, size=(n_algorithms,))
    coeffs = rng.normal(scale=2.5, size=(n_algorithms, 3))

    # per-algorithm difficulty offset to induce timeouts for some instances
    difficulty = rng.uniform(-8.0, 8.0, size=(n_algorithms,))

    # collect raw predictions for post-processing so we can enforce at least one solver per instance
    raw_matrix = np.empty((n_instances, n_algorithms), dtype=float)
    solved_matrix = np.zeros((n_instances, n_algorithms), dtype=bool)

    for j in range(n_algorithms):
        # linear + noise raw runtime
        raw = (
            features.values @ coeffs[j] + base[j] + rng.normal(0, 5.0, size=n_instances)
        )
        raw = np.maximum(raw, 1.0)
        raw_matrix[:, j] = raw

        # compute a solve probability that depends on features and algorithm difficulty
        # make the probability sensitive to both features and raw predicted difficulty
        score = (
            -0.15 * features["f1"].values
            + 0.10 * features["f2"].values
            - 0.12 * features["f3"].values
        )
        score += -0.4 * difficulty[j]
        # shift by raw to reduce solve chance for large raw
        prob_solve = 1.0 / (1.0 + np.exp(-(score - (raw - 35.0) / 18.0)))
        solved = rng.rand(n_instances) < prob_solve
        solved_matrix[:, j] = solved

    # build values: timeouts are encoded as budget (censoring)
    vals_matrix = np.where(
        solved_matrix & (raw_matrix < budget * 2), raw_matrix, budget * 2
    )

    # introduce a few fast solves and extra variance
    fast_mask = rng.rand(n_instances, n_algorithms) < 0.02
    vals_matrix[fast_mask] = rng.uniform(0.1, 3.0, size=fast_mask.sum())

    # ensure each instance has at least one solver (avoid rows with all timeouts)
    for i in range(n_instances):
        row = vals_matrix[i]
        if np.all(row >= budget):
            # pick algorithm with smallest raw (most promising) and mark as solved just under budget
            best_j = int(np.argmin(raw_matrix[i]))
            candidate = raw_matrix[i, best_j]
            vals_matrix[i, best_j] = candidate if candidate < budget else (budget * 0.9)

    perf = pd.DataFrame(
        vals_matrix,
        index=features.index,
        columns=[f"algo{j + 1}" for j in range(n_algorithms)],
    )
    return features, perf


def evaluate(preds, true_perf, budget=None):
    total = 0
    correct = 0
    for inst, rec in preds.items():
        algo = rec[0][0]
        if algo is None or inst not in true_perf.index:
            continue
        total += 1
        runtime = true_perf.loc[inst, algo]
        if budget is not None:
            if runtime <= budget:
                correct += 1
        else:
            if runtime <= true_perf.loc[inst].min():
                correct += 1
    return (correct / total) if total > 0 else 0.0


def print_sample(preds, true_perf, n=10):
    print("\nSample predictions:")
    for inst in list(true_perf.index)[:n]:
        rec = preds.get(inst, [(None, float("inf"))])
        algo, _ = rec[0]
        rt = true_perf.loc[inst, algo] if (algo in true_perf.columns) else None
        print(f"{inst}: recommended={algo} | runtime={rt}")


if __name__ == "__main__":
    budget = 60.0
    features, perf = generate_data(
        n_instances=200, n_algorithms=6, seed=2, budget=budget
    )

    train_X, test_X, train_perf, test_perf = train_test_split(
        features, perf, test_size=0.3, random_state=2
    )

    best_train = train_perf.min(axis=1)
    sat_labels = (best_train < (budget * 0.7)).tolist()

    selector = SATzilla()
    selector.budget = budget
    selector.fit(train_X, train_perf, sat_labels=sat_labels)

    print(test_perf.head(10))

    preds = selector.predict(test_X)

    acc = evaluate(preds, test_perf, budget=budget)
    print(f"\nSATzilla example accuracy (<= {budget}s): {acc:.2%}")

    print_sample(preds, test_perf, n=12)
