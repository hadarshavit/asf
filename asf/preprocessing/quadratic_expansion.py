import numpy as np


def expand_features(X_norm: np.ndarray) -> np.ndarray:
    """
    Quadratic feature expansion as done in SATzilla 2007.
    Expand normalized features to include original features followed by
    all pairwise products (j <= l). Returns expanded matrix.
    """
    n, k = X_norm.shape
    m = k + (k * (k + 1)) // 2
    X_exp = np.empty((n, m), dtype=float)
    X_exp[:, :k] = X_norm
    col = k
    for j in range(k):
        for m in range(j, k):
            X_exp[:, col] = X_norm[:, j] * X_norm[:, m]
            col += 1
    return X_exp
