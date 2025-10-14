import numpy as np
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle


def test_groupkfoldshuffle_no_shuffle():
    # 10 samples, 5 groups (each group appears twice)
    groups = np.repeat(np.arange(5), 2)
    X = np.arange(len(groups))

    gkf = GroupKFoldShuffle(n_splits=5, shuffle=False)

    splits = list(gkf.split(X, groups=groups))

    # Expect 5 splits, each test contains exactly 2 indices (one group)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(test_idx) == 2
        # no overlap
        assert set(train_idx).isdisjoint(set(test_idx))


def test_groupkfoldshuffle_shuffle_deterministic():
    groups = np.repeat(np.arange(6), 3)  # 18 samples, 6 groups
    X = np.arange(len(groups))

    g1 = GroupKFoldShuffle(n_splits=3, shuffle=True, random_state=42)
    g2 = GroupKFoldShuffle(n_splits=3, shuffle=True, random_state=42)

    splits1 = list(g1.split(X, groups=groups))
    splits2 = list(g2.split(X, groups=groups))

    # with same random_state and shuffle True, splits should be identical
    for (t1, s1), (t2, s2) in zip(splits1, splits2):
        assert np.array_equal(t1, t2)
        assert np.array_equal(s1, s2)
