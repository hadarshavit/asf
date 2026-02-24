"""Tests for baseline selectors (SBS and VBS)."""

import pandas as pd

from asf.selectors.baselines import SingleBestSolver, VirtualBestSolver


class TestSingleBestSolver:
    """Tests for SingleBestSolver."""

    def test_fit_finds_best_algorithm(self):
        """Test that fit identifies the algorithm with best average performance."""
        features = pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0]}, index=["inst1", "inst2", "inst3"]
        )
        performance = pd.DataFrame(
            {
                "algo1": [5.0, 5.0, 5.0],  # avg = 5
                "algo2": [1.0, 2.0, 3.0],  # avg = 2 (best)
                "algo3": [10.0, 10.0, 10.0],  # avg = 10
            },
            index=["inst1", "inst2", "inst3"],
        )

        selector = SingleBestSolver()
        selector.fit(features=features, performance=performance)

        assert selector.best_algorithm == "algo2"

    def test_fit_maximize_mode(self):
        """Test fit in maximize mode."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [5.0, 5.0, 5.0],
                "algo2": [1.0, 2.0, 3.0],
                "algo3": [10.0, 10.0, 10.0],  # best when maximizing
            },
            index=["i1", "i2", "i3"],
        )

        selector = SingleBestSolver(maximize=True)
        selector.fit(features=features, performance=performance)

        assert selector.best_algorithm == "algo3"

    def test_predict_returns_dictionary(self):
        """Test that predict returns a dictionary mapping instances to schedules."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [5.0, 5.0, 5.0],
                "algo2": [1.0, 2.0, 3.0],
            },
            index=["i1", "i2", "i3"],
        )

        selector = SingleBestSolver()
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features)

        assert isinstance(predictions, dict)
        assert len(predictions) == 3
        # Each prediction should be a list of (algorithm, time) tuples
        for inst_id, schedule in predictions.items():
            assert isinstance(schedule, list)
            assert len(schedule) > 0
            first_item = schedule[0]
            assert isinstance(first_item, tuple)
            algo, _ = first_item
            assert isinstance(algo, str)

    def test_predict_with_new_features(self):
        """Test predict with new features."""
        train_features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        train_performance = pd.DataFrame(
            {"algo1": [5.0, 5.0, 5.0], "algo2": [1.0, 2.0, 3.0]},
            index=["i1", "i2", "i3"],
        )

        selector = SingleBestSolver()
        selector.fit(features=train_features, performance=train_performance)

        test_features = pd.DataFrame({"f1": [4.0, 5.0]}, index=["t1", "t2"])
        predictions = selector.predict(features=test_features)

        assert len(predictions) == 2
        assert "t1" in predictions
        assert "t2" in predictions

    def test_define_hyperparameters(self):
        """Test hyperparameter definition."""
        hyperparams, conditions, forbiddens = SingleBestSolver._define_hyperparameters()

        assert hyperparams == []
        assert conditions == []
        assert forbiddens == []

    def test_with_budget(self):
        """Test with budget parameter."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [5.0, 5.0, 5.0],
                "algo2": [1.0, 2.0, 3.0],
            },
            index=["i1", "i2", "i3"],
        )

        selector = SingleBestSolver(budget=10)
        selector.fit(features=features, performance=performance)

        assert selector.best_algorithm is not None


class TestVirtualBestSolver:
    """Tests for VirtualBestSolver (Oracle)."""

    def test_fit_stores_performance(self):
        """Test that fit stores the performance data."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 5.0, 10.0],
                "algo2": [10.0, 1.0, 5.0],
            },
            index=["i1", "i2", "i3"],
        )

        selector = VirtualBestSolver()
        selector.fit(features=features, performance=performance)

        assert selector._performance is not None

    def test_predict_returns_dict_of_schedules(self):
        """Test that predict returns dictionary of schedules."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 5.0, 10.0],  # best for i1
                "algo2": [10.0, 1.0, 5.0],  # best for i2
                "algo3": [5.0, 10.0, 1.0],  # best for i3
            },
            index=["i1", "i2", "i3"],
        )

        selector = VirtualBestSolver()
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features, performance=performance)

        assert isinstance(predictions, dict)
        # Each prediction is a list of tuples
        assert predictions["i1"][0][0] == "algo1"
        assert predictions["i2"][0][0] == "algo2"
        assert predictions["i3"][0][0] == "algo3"

    def test_predict_maximize_mode(self):
        """Test predict in maximize mode."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {
                "algo1": [10.0, 1.0, 5.0],  # best for i1 when maximizing
                "algo2": [1.0, 10.0, 1.0],  # best for i2 when maximizing
                "algo3": [5.0, 5.0, 10.0],  # best for i3 when maximizing
            },
            index=["i1", "i2", "i3"],
        )

        selector = VirtualBestSolver(maximize=True)
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features, performance=performance)

        assert predictions["i1"][0][0] == "algo1"
        assert predictions["i2"][0][0] == "algo2"
        assert predictions["i3"][0][0] == "algo3"

    def test_predict_uses_train_performance_as_fallback(self):
        """Test that predict uses training performance if test performance not provided."""
        train_features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        train_performance = pd.DataFrame(
            {
                "algo1": [1.0, 5.0, 10.0],
                "algo2": [10.0, 1.0, 5.0],
            },
            index=["i1", "i2", "i3"],
        )

        selector = VirtualBestSolver()
        selector.fit(features=train_features, performance=train_performance)

        # Predict with same features but no performance
        predictions = selector.predict(features=train_features)

        assert isinstance(predictions, dict)
        assert len(predictions) == 3

    def test_define_hyperparameters(self):
        """Test hyperparameter definition."""
        hyperparams, conditions, forbiddens = (
            VirtualBestSolver._define_hyperparameters()
        )

        assert hyperparams == []
        assert conditions == []
        assert forbiddens == []
