import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

from asf.selectors.performance_model import PerformanceModel
from asf.selectors.selector_pipeline import SelectorPipeline


def test_feature_group_filter_precedes_preprocessing():
    features = pd.DataFrame(
        {"cheap": [0.0, 1.0, 2.0, 3.0], "expensive": [100.0, 300.0, 200.0, 500.0]}
    )
    performance = pd.DataFrame({"algo": [1.0, 2.0, 3.0, 4.0]})
    pipeline = SelectorPipeline(
        PerformanceModel(estimator=DecisionTreeRegressor()),
        preprocessor=PCA(n_components=1),
        feature_groups={"cheap": {"provides": ["cheap"]}},
    )

    pipeline.fit(features, performance)

    assert pipeline.preprocessor["PCA"].n_features_in_ == 1
    assert all(
        schedule[0] == "cheap" for schedule in pipeline.predict(features).values()
    )
