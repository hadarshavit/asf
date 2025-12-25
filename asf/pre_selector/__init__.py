"""
This module provides various algorithms for algorithm pre-selection.
"""

from __future__ import annotations

from asf.pre_selector.marginal_contribution_based import (
    MarginalContributionBasedPreSelector,
)
from asf.pre_selector.optimize_pre_selection import OptimizePreSelection
from asf.pre_selector.sbs_pre_selection import SBSPreSelector
from asf.pre_selector.brute_force_pre_selection import BruteForcePreSelector
from asf.pre_selector.beam_search_pre_selection import BeamSearchPreSelector
from asf.pre_selector.knee_of_the_curve_pre_selector import KneeOfCurvePreSelector
from asf.pre_selector.random_local_search_pre_selection import (
    RandomLocalSearchPreSelector,
)
from asf.pre_selector.genetic_algorithm_pre_selection import (
    GeneticAlgorithmPreSelector,
)


__all__ = [
    "MarginalContributionBasedPreSelector",
    "OptimizePreSelection",
    "SBSPreSelector",
    "BruteForcePreSelector",
    "BeamSearchPreSelector",
    "KneeOfCurvePreSelector",
    "RandomLocalSearchPreSelector",
    "GeneticAlgorithmPreSelector",
]
