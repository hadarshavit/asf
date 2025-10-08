import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from asf.selectors.isac_selector import ISACSelector

def generate_isac_data(n_instances=120, n_algorithms=5, seed=0):
    """
    Synthetic dataset for ISAC example.
    """
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.uniform(0, 10, size=(n_instances, 2)),
        columns=["f1", "f2"],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    perf = pd.DataFrame(index=features.index)
    perf["algo1"] = 50 + 2.5 * features["f1"] + rng.normal(0, 3, n_instances)
    perf["algo2"] = 40 + 3 * features["f2"] + rng.normal(0, 3, n_instances)
    perf["algo3"] = 60 - 2 * features["f1"] + 1.5 * features["f2"] + rng.normal(0, 3, n_instances)
    perf["algo4"] = 30 + 1 * features["f1"] + 4 * features["f2"] + rng.normal(0, 3, n_instances)
    perf["algo5"] = 60 + 2 * features["f1"] - 1 * features["f2"] + rng.normal(0, 3, n_instances)

    # enforce reasonable minimum runtime
    perf[perf < 5] = 5
    return features, perf

def evaluate_isac(predictions, true_perf, budget=None):
    """
    Evaluate ISAC predictions.

    Returns:
        achieved_acc (float): Fraction of instances where the recommended algorithm
            achieves runtime <= budget (if budget provided) or is the fastest (otherwise).
        max_acc (float): Maximum achievable accuracy under the same evaluation criterion
            (i.e., fraction of instances for which some algorithm achieves runtime <= budget).
    """
    correct = 0
    total = 0
    for inst, rec in predictions.items():
        if inst not in true_perf.index:
            continue
        algo = rec[0][0]
        if algo is None:
            continue
        total += 1
        runtime = true_perf.loc[inst, algo]
        if budget is not None:
            if runtime <= budget:
                correct += 1
        else:
            # compare to true best
            if runtime <= true_perf.loc[inst].min():
                correct += 1
    achieved_acc = correct / total if total > 0 else 0.0

    # Compute maximum achievable accuracy under the same budget criterion
    if budget is not None:
        best_runtimes = true_perf.min(axis=1)
        max_acc = float((best_runtimes <= budget).mean())
    else:
        # If no budget provided, the maximum achievable accuracy (choosing best algorithm per instance) is 1.0
        max_acc = 1.0

    return achieved_acc, max_acc

def print_sample(predictions, true_perf, n=8):
    print("\nSample ISAC predictions:")
    for inst in list(true_perf.index)[:n]:
        rec = predictions.get(inst, [(None, 0.0)])
        algo = rec[0][0]
        runtime = true_perf.loc[inst, algo] if algo in true_perf.columns else None
        print(f"{inst}: recommended = {algo} | runtime = {runtime}")

if __name__ == "__main__":
    # generate data
    features, performance = generate_isac_data(n_instances=120, n_algorithms=5, seed=1)

    # split train/test
    n_train = int(0.7 * len(features))
    train_features = features.iloc[:n_train]
    train_perf = performance.iloc[:n_train]
    test_features = features.iloc[n_train:]
    test_perf = performance.iloc[n_train:]

    print("\nTest performance (head):")
    print(test_perf.head(10))

    # Default ISAC (GMeans)
    selector = ISACSelector()
    selector.fit(train_features, train_perf)
    preds = selector.predict(test_features)
    acc, max_acc = evaluate_isac(preds, test_perf, budget=60)
    print(f"\nISAC (GMeans) accuracy (<=60s): {acc:.2%} (max achievable: {max_acc:.2%})")
    print_sample(preds, test_perf, n=10)

    # ISAC with KMeans (example: 6 clusters)
    selector_km = ISACSelector(clusterer=KMeans, clusterer_kwargs={"n_clusters": 6})
    selector_km.fit(train_features, train_perf)
    preds_km = selector_km.predict(test_features)
    acc_km, max_acc_km = evaluate_isac(preds_km, test_perf, budget=60)
    print(f"\nISAC (KMeans, n_clusters=6) accuracy (<=60s): {acc_km:.2%} (max achievable: {max_acc_km:.2%})")
    print_sample(preds_km, test_perf, n=10)