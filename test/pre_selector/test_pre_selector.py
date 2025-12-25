import numpy as np
import pytest
import pandas as pd
from asf.pre_selector import (
    OptimizePreSelection,
    MarginalContributionBasedPreSelector,
    SBSPreSelector,
)
from asf.metrics import virtual_best_solver
from functools import partial


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
    return pd.DataFrame(data, columns=pd.Index(["algo1", "algo2", "algo3"]))


def test_optimize_pre_selection(dummy_performance):
    # Create an instance of the OptimizePreSelection class
    pre_selector = OptimizePreSelection(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
        fmin_function="SLSQP",
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty


def test_marginal_contribution_based_pre_selector(dummy_performance):
    # Create an instance of the MarginalContributionBasedPreSelector class
    pre_selector = MarginalContributionBasedPreSelector(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty


def test_sbs_pre_selector(dummy_performance):
    # Create an instance of the SBSPreSelector class
    pre_selector = SBSPreSelector(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty
