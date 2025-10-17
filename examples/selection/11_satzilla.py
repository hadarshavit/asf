import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from asf.selectors.satzilla import SATzilla


def generate_data(n_instances=80, seed=0):
    np.random.seed(seed)
    features = pd.DataFrame(
        np.random.uniform(0, 10, size=(n_instances, 2)),
        columns=["size", "density"],
        index=[f"inst_{i}" for i in range(n_instances)],
    )
    # Each algorithm's performance is a different function of the features
    performance = pd.DataFrame(
        {
            "algo1": 400
            + 20 * features["size"]
            - 100 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo2": 20
            + 35 * features["size"]
            + 6 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo3": 90
            - 20 * features["size"]
            + 80 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo4": 100
            - 120 * features["size"]
            + 250 * features["density"]
            + np.random.normal(0, 5, n_instances),
        },
        index=features.index,
    )
    performance[performance < 5] = 5
    return features, performance


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
    budget = 200
    features, perf = generate_data(n_instances=200, seed=2)

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
