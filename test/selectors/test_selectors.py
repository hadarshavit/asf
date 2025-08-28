import numpy as np
import pytest
import pandas as pd
from asf.selectors import (
    PairwiseClassifier,
    MultiClassClassifier,
    PairwiseRegressor,
    PerformanceModel,
    SimpleRanking,
    JointRanking,
    SurvivalAnalysisSelector,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRanker
from asf.selectors.selector_tuner import tune_selector
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.predictors import (
    RandomForestClassifierWrapper,
    RandomForestRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    SVMClassifierWrapper,
    SVMRegressorWrapper,
    LinearClassifierWrapper,
    LinearRegressorWrapper,
    EPMRandomForest,
    MLPClassifierWrapper,
    MLPRegressorWrapper,
    RegressionMLP,
)
import shutil


@pytest.fixture
def dummy_performance():
    data = np.array(
        [
            [120, 100, 110],
            [140, 150, 130],
            [180, 170, 190],
            [160, 150, 140],
            [250, 240, 260],
            [230, 220, 210],
            [300, 310, 320],
            [280, 290, 270],
            [350, 340, 360],
            [330, 320, 310],
            [400, 390, 410],
            [380, 370, 360],
            [450, 440, 460],
            [430, 420, 410],
            [500, 490, 510],
            [480, 470, 460],
            [550, 540, 560],
            [530, 520, 510],
            [600, 590, 610],
            [580, 570, 560],
        ]
    )
    return pd.DataFrame(data, columns=["algo1", "algo2", "algo3"])


@pytest.fixture
def dummy_features():
    data = np.array(
        [
            [10, 5, 1],
            [20, 10, 2],
            [15, 8, 1.5],
            [25, 12, 2.5],
            [30, 15, 3],
            [35, 18, 3.5],
            [40, 20, 4],
            [45, 22, 4.5],
            [50, 25, 5],
            [55, 28, 5.5],
            [60, 30, 6],
            [65, 32, 6.5],
            [70, 35, 7],
            [75, 38, 7.5],
            [80, 40, 8],
            [85, 42, 8.5],
            [90, 45, 9],
            [95, 48, 9.5],
            [100, 50, 10],
            [105, 52, 10.5],
        ]
    )
    return pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])


def validate_predictions(predictions):
    """
    Validates that predictions have the expected structure:
    - Length of predictions is 20.
    - Each value in predictions is a list.
    - Each list has a length of 1.
    """
    assert len(predictions) == 20, "Predictions length is not 20"
    assert all(isinstance(v, list) for v in predictions.values()), (
        "Not all predictions are lists"
    )
    assert all(len(v) == 1 for v in predictions.values()), (
        "Not all lists in predictions have length 1"
    )


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestClassifier,
        RandomForestClassifierWrapper,
        XGBoostClassifierWrapper,
        SVMClassifierWrapper,
        LinearClassifierWrapper,
        MLPClassifierWrapper,
    ],
)
def test_pairwise_classifier(dummy_performance, dummy_features, model_class):
    classifier = PairwiseClassifier(
        model_class=model_class, use_weights=model_class != MLPClassifierWrapper
    )
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestClassifier,
        RandomForestClassifierWrapper,
        XGBoostClassifierWrapper,
        SVMClassifierWrapper,
        LinearClassifierWrapper,
        MLPClassifierWrapper,
    ],
)
def test_multi_class_classifier(dummy_performance, dummy_features, model_class):
    classifier = MultiClassClassifier(model_class=model_class)
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestRegressorWrapper,
        XGBoostRegressorWrapper,
        SVMRegressorWrapper,
        LinearRegressorWrapper,
        EPMRandomForest,
        MLPRegressorWrapper,
        RegressionMLP,
    ],
)
def test_pairwise_regressor(dummy_performance, dummy_features, model_class):
    regressor = PairwiseRegressor(model_class=model_class)
    regressor.fit(dummy_features, dummy_performance)
    predictions = regressor.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestRegressorWrapper,
        XGBoostRegressorWrapper,
        SVMRegressorWrapper,
        LinearRegressorWrapper,
        EPMRandomForest,
        MLPRegressorWrapper,
        RegressionMLP,
    ],
)
def test_performance_model(dummy_performance, dummy_features, model_class):
    model = PerformanceModel(model_class=model_class)
    model.fit(dummy_features, dummy_performance)
    predictions = model.predict(dummy_features)
    validate_predictions(predictions)


def save_load(dummy_performance, dummy_features):
    model = PerformanceModel(model_class=RandomForestRegressor)
    model.fit(dummy_features, dummy_performance)
    model.save("model.pkl")
    loaded_model = PerformanceModel.load("model.pkl")
    predictions = loaded_model.predict(dummy_features)
    validate_predictions(predictions)


def test_simple_ranking(dummy_performance, dummy_features):
    selector = SimpleRanking(model_class=XGBRanker)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_joint_ranking(dummy_performance, dummy_features):
    selector = JointRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis(dummy_performance, dummy_features):
    selector = SurvivalAnalysisSelector(cutoff_time=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_selector_tuner(dummy_performance, dummy_features):
    # Keep runcount_limit and cv low for fast testing
    tuned_pipeline = tune_selector(
        X=dummy_features,
        y=dummy_performance,
        selector_class=[PairwiseClassifier, PairwiseRegressor],
        runcount_limit=2,
        cv=2,
        seed=42,
        output_dir="./smac_test_output",  # Use a test-specific output dir
        smac_scenario_kwargs={},
    )
    assert isinstance(tuned_pipeline, SelectorPipeline)
    # Fit the best pipeline found by the tuner
    tuned_pipeline.fit(dummy_features, dummy_performance)
    predictions = tuned_pipeline.predict(dummy_features)
    validate_predictions(predictions)

    # Clean up SMAC output directory if needed (optional)
    shutil.rmtree("./smac_test_output", ignore_errors=True)
