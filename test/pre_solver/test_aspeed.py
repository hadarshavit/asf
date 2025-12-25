import pandas as pd
from asf.presolving.aspeed import Aspeed


def get_small_data():
    # create a tiny deterministic performance table with 5 instances and 3 algorithms
    perf = pd.DataFrame(
        [
            [10, 20, 30],
            [5, 25, 40],
            [60, 7, 50],
            [12, 12, 12],
            [100, 200, 1],
        ],
        columns=pd.Index(["algoA", "algoB", "algoC"]),
    )
    features = pd.DataFrame([[0], [1], [2], [3], [4]])
    return features, perf


def test_aspeed_returns_schedule_non_empty():
    features, performance = get_small_data()
    presolver = Aspeed(cores=1, aspeed_cutoff=10, budget=30)
    # run fit and assert schedule is non-empty list
    presolver.fit(features, performance)
    assert isinstance(presolver.schedule, list)
    assert len(presolver.schedule) > 0


def test_aspeed_schedule_entries_well_formed():
    _, performance = get_small_data()
    presolver = Aspeed(cores=1, aspeed_cutoff=10, budget=30)
    presolver.fit(None, performance)
    # each entry must be tuple (algorithm_name, allocated_time)
    for entry in presolver.schedule:
        assert isinstance(entry, tuple)
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], (int, float))
