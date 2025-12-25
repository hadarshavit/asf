from asf.selectors.joint_ranking import JointRanking
import pandas as pd


# from asf import ScenarioMetadata
class ScenarioMetadata:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        import numpy as np

        return np.zeros(len(X))


r = JointRanking(
    ScenarioMetadata(  # type: ignore
        ["algo1", "algo2", "algo3"],
        features=["feature1", "feature2", "feature3"],
        performance_metric="runtime",
        maximize=False,
        budget=1000,
        feature_groups={"group1": ["feature1", "feature2"], "group2": ["feature3"]},
        algorith_features=["af1", "af2", "af3"],
    ),
)
r.fit(
    features=pd.DataFrame(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        index=pd.Index(["i1", "i2", "i3"]),
        columns=pd.Index(["feature1", "feature2", "feature3"]),
    ),
    performance=pd.DataFrame(
        [[1, 2, 3], [22, 3, 44], [33, 44, 5]],
        index=pd.Index(["i1", "i2", "i3"]),
        columns=pd.Index(["algo1", "algo2", "algo3"]),
    ),
    algorithm_features=pd.DataFrame(
        [[111, 111, 111], [222, 222, 222], [333, 333, 333]],
        index=pd.Index(["algo1", "algo2", "algo3"]),
        columns=pd.Index(["af1", "af2", "af3"]),
    ),
)

print(
    r.predict(
        pd.DataFrame(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            index=pd.Index(["i1", "i2", "i3"]),
            columns=pd.Index(["feature1", "feature2", "feature3"]),
        )
    )
)
