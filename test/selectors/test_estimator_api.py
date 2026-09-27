"""Configured estimators must remain templates across independent training tasks."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from asf.predictors import RandomForestRegressorWrapper, SklearnWrapper
from asf.selectors import (
    APPS,
    CollaborativeFilteringSelector,
    DyadRanking,
    MultiClassClassifier,
    PairwiseClassifier,
    PairwiseRegressor,
    PerformanceModel,
    SATzilla,
    SimpleRanking,
)
from asf.selectors.feature_generator import AbstractFeatureGenerator


@pytest.fixture
def training_data():
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
    y = pd.DataFrame(
        {
            "a": [1.0, 4.0, 2.0, 3.0],
            "b": [3.0, 1.0, 4.0, 2.0],
            "c": [2.0, 3.0, 1.0, 4.0],
        }
    )
    return X, y


@pytest.mark.parametrize(
    "selector_cls,estimator,attribute,kwargs",
    [
        (PairwiseClassifier, DecisionTreeClassifier(max_depth=2), "classifiers", {}),
        (PairwiseRegressor, DecisionTreeRegressor(max_depth=2), "regressors", {}),
        (MultiClassClassifier, DecisionTreeClassifier(max_depth=2), "classifier", {}),
        (PerformanceModel, DecisionTreeRegressor(max_depth=2), "regressors", {}),
        (
            PerformanceModel,
            DecisionTreeRegressor(max_depth=2),
            "regressors",
            {"use_multi_target": True},
        ),
    ],
)
def test_estimators_are_cloned_for_each_task_and_refit(
    training_data, selector_cls, estimator, attribute, kwargs, tmp_path
):
    X, y = training_data
    selector = selector_cls(estimator=estimator, budget=100, **kwargs)
    selector.fit(X, y)
    fitted = getattr(selector, attribute)
    models = fitted if isinstance(fitted, list) else [fitted]
    assert not hasattr(estimator, "tree_")
    assert len({id(model) for model in models}) == len(models)
    for model in models:
        assert model is not estimator
        assert model.max_depth == 2
        assert hasattr(model, "tree_")

    predictions = selector.predict(X)
    path = tmp_path / "selector.joblib"
    selector.save(path)
    restored = selector_cls.load(path)
    assert restored.predict(X) == predictions

    selector.fit(X, y)
    refitted = getattr(selector, attribute)
    new_models = refitted if isinstance(refitted, list) else [refitted]
    assert not {id(model) for model in models} & {id(model) for model in new_models}


@pytest.mark.parametrize("jackknife", [True, False])
def test_apps_clones_each_ensemble_member(training_data, jackknife):
    X, y = training_data
    estimator = DecisionTreeRegressor(max_depth=2)
    selector = APPS(
        estimator=estimator,
        use_jackknife=jackknife,
        n_jackknife_folds=2,
        n_estimators_for_std=2,
        budget=100,
    )
    selector.fit(X, y)
    models = [
        member[0] if jackknife else member
        for ensemble in selector.predictors
        for member in ensemble
    ]
    assert len({id(model) for model in models}) == 6
    assert all(model.max_depth == 2 for model in models)
    assert not hasattr(estimator, "tree_")
    selector.fit(X, y)
    assert len(selector.predictors) == 3


def test_unweighted_selector_accepts_pipeline_without_sample_weight(training_data):
    X, y = training_data
    estimator = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    selector = PairwiseClassifier(estimator=estimator, use_weights=False)
    selector.fit(X, y)
    assert len(selector.predict(X)) == len(X)
    assert not hasattr(estimator[0], "mean_")


@pytest.mark.parametrize(
    "wrapper",
    [
        SklearnWrapper(RandomForestRegressor, n_estimators=2, random_state=7),
        RandomForestRegressorWrapper(n_estimators=2, random_state=7),
    ],
)
def test_wrappers_clone_parameters_without_fitted_state(training_data, wrapper):
    X, y = training_data
    wrapper.fit(X, y["a"])
    copied = clone(wrapper)
    assert type(copied) is type(wrapper)
    assert copied.get_params() == wrapper.get_params()
    assert not hasattr(copied.model_class, "estimators_")
    selector = PairwiseRegressor(estimator=wrapper)
    selector.fit(X, y)
    assert all(model is not wrapper for model in selector.regressors)


def test_prefitted_estimator_is_not_retrained(training_data):
    X, y = training_data
    estimator = DecisionTreeRegressor().fit(X, np.full(len(X), 123.0))
    original_tree = estimator.tree_
    selector = PairwiseRegressor(estimator=estimator)
    selector.fit(X, y)
    assert estimator.tree_ is original_tree
    np.testing.assert_array_equal(estimator.predict(X), np.full(len(X), 123.0))
    assert all(model.tree_ is not original_tree for model in selector.regressors)


def test_joint_performance_model_clones_with_algorithm_features(training_data):
    X, y = training_data
    estimator = DecisionTreeRegressor()
    selector = PerformanceModel(estimator=estimator)
    algorithm_features = pd.DataFrame({"size": [1.0, 2.0, 3.0]}, index=y.columns)
    selector.fit(X, y, algorithm_features=algorithm_features)
    assert selector.regressors.n_features_in_ == 2
    assert len(selector.predict(X)) == len(X)
    assert not hasattr(estimator, "tree_")


def test_collaborative_filtering_clones_estimator(training_data):
    X, y = training_data
    estimator = DecisionTreeRegressor(max_depth=2)
    selector = CollaborativeFilteringSelector(
        estimator=estimator, n_iter=2, n_components=2
    )
    selector.fit(X, y)
    assert selector.model is not estimator
    assert selector.model.max_depth == 2
    assert not hasattr(estimator, "tree_")
    assert len(selector.predict(X)) == len(X)


def test_satzilla_clones_label_and_per_algorithm_models(training_data):
    X, y = training_data
    classifier = DecisionTreeClassifier()
    regressor = DecisionTreeRegressor()
    selector = SATzilla(estimator=classifier, epm_estimator=regressor, budget=100)
    selector.fit(X, y, labels=["sat", "unsat", "sat", "unsat"])
    assert not hasattr(classifier, "tree_")
    assert not hasattr(regressor, "tree_")
    assert selector.label_classifier is not classifier
    models = [
        epm.predictor
        for per_label in selector.epms.values()
        for epm in per_label.values()
    ]
    assert len(models) == 6
    assert len({id(model) for model in models}) == 6
    assert len(selector.predict(X)) == len(X)


def test_satzilla_uses_probabilities_from_configured_classifier():
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
    y = pd.DataFrame({"a": [1.0, 1.0, 1.0, 100.0], "b": [10.0, 10.0, 10.0, 1.0]})
    selector = SATzilla(
        estimator=DummyClassifier(strategy="prior"),
        epm_estimator=DummyRegressor(),
        budget=200,
    )
    selector.fit(X, y, labels=["sat", "sat", "sat", "unsat"])
    # Soft probabilities give b a lower expected runtime (7.75 versus 25.75).
    # A hard prediction of the majority label would incorrectly select a.
    assert all(schedule[0][0] == "b" for schedule in selector.predict(X).values())


def test_satzilla_hard_predictions_match_numeric_labels():
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
    y = pd.DataFrame({"a": [1.0, 1.0, 100.0, 100.0], "b": [100.0, 100.0, 1.0, 1.0]})
    selector = SATzilla(estimator=LinearSVC(dual=False), epm_estimator=DummyRegressor())
    selector.fit(X, y, labels=[0, 0, 1, 1])
    assert all(
        schedule[0][0] == "b" for schedule in selector.predict(X.iloc[2:]).values()
    )


class RankingTree(DecisionTreeRegressor):
    def fit(self, X, y, qid):
        self.qid_ = np.asarray(qid)
        return super().fit(X, y)


@pytest.mark.parametrize("selector_cls", [SimpleRanking, DyadRanking])
def test_ranking_selectors_clone_and_pass_query_groups(training_data, selector_cls):
    X, y = training_data
    estimator = RankingTree(max_depth=2)
    selector = selector_cls(estimator=estimator)
    selector.fit(X, y)
    assert selector.classifier is not estimator
    assert selector.classifier.max_depth == 2
    assert len(selector.classifier.qid_) > 0
    assert not hasattr(estimator, "tree_")
    assert len(selector.predict(X)) == len(X)


class NonCloneablePredictor:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(X))


def test_non_cloneable_estimator_has_actionable_error(training_data):
    X, y = training_data
    selector = MultiClassClassifier(estimator=NonCloneablePredictor())
    with pytest.raises(TypeError, match="must support sklearn.base.clone"):
        selector.fit(X, y)


@pytest.mark.parametrize("maximize,expected", [(False, "low"), (True, "high")])
def test_multiclass_respects_optimization_direction(maximize, expected):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"low": [1.0, 1.0, 1.0], "high": [10.0, 10.0, 10.0]})
    selector = MultiClassClassifier(
        estimator=DecisionTreeClassifier(), maximize=maximize
    )
    selector.fit(X, y)
    assert [schedule[0][0] for schedule in selector.predict(X).values()] == [
        expected
    ] * 3


class ExtraFeature(AbstractFeatureGenerator):
    def generate_features(self, base_features):
        return pd.DataFrame(
            {"extra": np.ones(len(base_features))}, index=base_features.index
        )


@pytest.mark.parametrize("numpy_training", [True, False])
def test_numpy_prediction_uses_raw_feature_names(training_data, numpy_training):
    X, y = training_data
    selector = MultiClassClassifier(
        estimator=DecisionTreeClassifier(), hierarchical_generator=ExtraFeature()
    )
    selector.fit(
        X.to_numpy() if numpy_training else X, y.to_numpy() if numpy_training else y
    )
    assert len(selector.features) == 2
    assert len(selector.input_features_) == 1
    expected = selector.predict(
        pd.DataFrame(X.to_numpy(), columns=selector.input_features_)
    )
    assert selector.predict(X.to_numpy()) == expected


def test_ensemble_clone_preserves_selector_configuration(training_data):
    from asf.selectors.ensembles import _clone_selector

    selector = PairwiseRegressor(
        estimator=DecisionTreeRegressor(max_depth=2), budget=7, maximize=True
    )
    cloned = _clone_selector(selector)
    assert isinstance(cloned, PairwiseRegressor)
    assert cloned.budget == 7
    assert cloned.maximize is True
    assert cloned.estimator.max_depth == 2
