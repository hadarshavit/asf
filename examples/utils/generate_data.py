# Python
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("asf/examples/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_and_save(n_instances=200, n_algorithms=6, seed=42, budget=60.0):
    rng = np.random.RandomState(seed)
    # features correlated structure
    X = rng.normal(loc=[5.0, 3.0, 7.0], scale=[2.5, 1.5, 3.0], size=(n_instances, 3))
    features = pd.DataFrame(
        X, columns=["f1", "f2", "f3"], index=[f"inst_{i}" for i in range(n_instances)]
    )

    # per-algo base + feature sensitivity
    base = rng.uniform(20, 90, size=n_algorithms)
    coeffs = rng.normal(scale=2.2, size=(n_algorithms, 3))
    difficulty = rng.uniform(-6.0, 6.0, size=n_algorithms)

    raw = np.empty((n_instances, n_algorithms))
    solved = np.zeros_like(raw, dtype=bool)

    for j in range(n_algorithms):
        raw_j = (
            features.values @ coeffs[j] + base[j] + rng.normal(0, 6.0, size=n_instances)
        )
        raw_j = np.clip(raw_j, 0.5, None)
        raw[:, j] = raw_j
        # solve probability depends on features and raw difficulty
        score = (
            -0.12 * features["f1"].values
            + 0.08 * features["f2"].values
            - 0.10 * features["f3"].values
        )
        score += -0.35 * difficulty[j] - (raw_j - 40.0) / 18.0
        p = 1.0 / (1.0 + np.exp(-score))
        solved[:, j] = rng.rand(n_instances) < p

    # encode timeouts as budget; keep solved & raw < budget as runtime
    vals = np.where(solved & (raw < budget), raw, budget)

    # inject some fast solves and variability
    fast = rng.rand(*vals.shape) < 0.02
    vals[fast] = rng.uniform(0.05, 3.0, size=fast.sum())

    # ensure at least one solver per instance
    for i in range(n_instances):
        if np.all(vals[i] >= budget):
            best = int(np.argmin(raw[i]))
            vals[i, best] = min(raw[i, best], budget * 0.9)

    perf = pd.DataFrame(
        vals,
        index=features.index,
        columns=[f"algo{j + 1}" for j in range(n_algorithms)],
    )

    features.to_csv(OUT_DIR / "features.csv")
    perf.to_csv(OUT_DIR / "performance.csv")
    print(f"Wrote {OUT_DIR / 'features.csv'} and {OUT_DIR / 'performance.csv'}")


if __name__ == "__main__":
    generate_and_save()
