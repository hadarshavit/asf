from typing import cast

import pytest
from asf.selectors import MetaSelector, SingleBestSolver, SurvivalAnalysisScheduler


class _ConfiguredSingleBestSolver(SingleBestSolver):
    seen_markers: list[int] = []

    def __init__(self, marker: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.marker = marker

    def _fit(self, features, performance, **kwargs):
        type(self).seen_markers.append(self.marker)
        super()._fit(features, performance, **kwargs)


def test_meta_selector(dummy_performance, dummy_features, validate_predictions):
    # Test MetaSelector with base selectors and a meta_selector
    base_selectors = [SingleBestSolver(budget=3.0), SingleBestSolver(budget=3.0)]
    meta = SingleBestSolver(budget=3.0)
    # Pass them positionally
    selector = MetaSelector(base_selectors, meta, budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_meta_selector_preserves_base_selector_configuration(
    dummy_performance, dummy_features
):
    _ConfiguredSingleBestSolver.seen_markers = []

    base_selectors = [_ConfiguredSingleBestSolver(marker=7, budget=3.0)]
    meta = SingleBestSolver(budget=3.0)
    selector = MetaSelector(base_selectors, meta, budget=3.0)

    selector.fit(dummy_features, dummy_performance)

    assert selector.base_selectors_ is not None
    configured_selector = cast(_ConfiguredSingleBestSolver, selector.base_selectors_[0])
    assert isinstance(configured_selector, _ConfiguredSingleBestSolver)
    assert configured_selector.marker == 7
    assert _ConfiguredSingleBestSolver.seen_markers
    assert all(marker == 7 for marker in _ConfiguredSingleBestSolver.seen_markers)


def test_meta_selector_rejects_schedule_base(dummy_performance, dummy_features):
    # MetaSelector currently only supports base selectors that return a single algorithm
    base_selectors = [SurvivalAnalysisScheduler(budget=3.0)]
    meta = SingleBestSolver(budget=3.0)

    # This should fail during initialization due to RETURN_TYPE check
    with pytest.raises(ValueError, match="must have RETURN_TYPE 'single'"):
        MetaSelector(base_selectors, meta, budget=3.0)
