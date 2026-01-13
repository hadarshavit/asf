from asf.selectors import ISAC


def test_isac_selector(dummy_performance, dummy_features, validate_predictions):
    selector = ISAC()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_isac_dbscan(dummy_performance, dummy_features, validate_predictions):
    from asf.clustering.wrappers import DBSCANWrapper

    selector = ISAC(
        clusterer=DBSCANWrapper, clusterer_kwargs={"eps": 0.5, "min_samples": 2}
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
