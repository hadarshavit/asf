from asf.selectors.rpc_selector import RPCSelector


def test_rpc_selector_basic(dummy_performance, dummy_features, validate_predictions):
    """Test basic RPC functionality: fit and predict with pairwise voting."""
    from sklearn.ensemble import RandomForestClassifier

    selector = RPCSelector(
        classifier_class=RandomForestClassifier,
        n_estimators=20,
        random_state=42,
        budget=450.0,
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_rpc_selector_decision_tree(
    dummy_performance, dummy_features, validate_predictions
):
    """Test RPC with DecisionTreeClassifier instead of RandomForest."""
    from sklearn.tree import DecisionTreeClassifier

    selector = RPCSelector(
        classifier_class=DecisionTreeClassifier,
        classifier_kwargs={"max_depth": 5, "random_state": 42},
        budget=450.0,
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_rpc_selector_pairwise_structure(dummy_performance, dummy_features):
    """Test that RPC creates correct number of pairwise classifiers."""
    from sklearn.ensemble import RandomForestClassifier

    selector = RPCSelector(
        classifier_class=RandomForestClassifier,
        n_estimators=10,
        random_state=42,
        budget=450.0,
    )
    selector.fit(dummy_features, dummy_performance)

    n_algorithms = len(dummy_performance.columns)
    expected_pairs = n_algorithms * (n_algorithms - 1) // 2

    assert len(selector.pairs) == expected_pairs, (
        f"Expected {expected_pairs} pairs, got {len(selector.pairs)}"
    )
    assert len(selector.classifiers) == expected_pairs, (
        f"Expected {expected_pairs} classifiers, got {len(selector.classifiers)}"
    )
