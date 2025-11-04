import pandas as pd
import numpy as np
from asf.selectors.sunny import SUNNY


def generate_simple_data(n_instances=80, seed=0):
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


def print_sunny_schedules(predictions, true_performance, budget, n=10):
    print("\nSample SUNNY schedules:")
    for instance in list(true_performance.index)[:n]:
        schedule = predictions[instance]
        row = true_performance.loc[instance]
        solved_algo = None
        time_used = 0
        solved = False
        for algo, t in schedule:
            runtime = row[algo]
            time_used += t
            if runtime <= budget and runtime <= time_used:
                solved_algo = algo
                solved = True
                break
        if solved:
            emoji = "✅"
        else:
            emoji = "❌"
        print(
            f"{instance}: schedule = {[(a, int(t)) for a, t in schedule]} | "
            f"Solved by: {solved_algo} {emoji}"
        )


if __name__ == "__main__":
    features, performance = generate_simple_data()
    budget = 200

    # Split into train/test
    n_train = int(0.7 * len(features))
    train_features = features.iloc[:n_train]
    train_performance = performance.iloc[:n_train]
    test_features = features.iloc[n_train:]
    test_performance = performance.iloc[n_train:]

    print("Training performance matrix (no NaNs):")
    print(train_performance.head())

    print("\nTest performance matrix (no NaNs):")
    print(test_performance.head())

    selector = SUNNY(k=5, use_v2=False, budget=budget)
    selector.fit(train_features, train_performance)

    predictions = selector.predict(test_features)
    print("\nK = ", selector.k)
    print_sunny_schedules(predictions, test_performance, budget, n=10)

    print("\nUsing v2")
    selector = SUNNY(k=5, use_v2=True, budget=budget)
    selector.fit(train_features, train_performance)
    predictions = selector.predict(test_features)
    print("K = ", selector.k)
    print_sunny_schedules(predictions, test_performance, budget, n=10)
