"""Comprehensive tests for selector configuration spaces."""

import pytest

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace


class TestSelectorConfigurationSpaces:
    """Test configuration spaces for selectors that have them."""

    def test_pairwise_classifier_config_space(self):
        """Test PairwiseClassifier configuration space."""
        from asf.selectors.pairwise_classifier import PairwiseClassifier

        cs = PairwiseClassifier.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have pairwise_classifier prefix and model_class, use_weights
        assert any("pairwise_classifier" in name for name in hp_names)
        assert any("model_class" in name for name in hp_names)
        assert any("use_weights" in name for name in hp_names)

    def test_pairwise_regressor_config_space(self):
        """Test PairwiseRegressor configuration space."""
        from asf.selectors.pairwise_regressor import PairwiseRegressor

        cs = PairwiseRegressor.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have pairwise_regressor prefix
        assert any("pairwise_regressor" in name for name in hp_names)

    def test_multi_class_classifier_config_space(self):
        """Test MultiClassClassifier configuration space."""
        from asf.selectors.multi_class import MultiClassClassifier

        cs = MultiClassClassifier.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have multi_class prefix and model_class
        assert any("multi_class" in name for name in hp_names)

    def test_performance_model_config_space(self):
        """Test PerformanceModel configuration space."""
        from asf.selectors.performance_model import PerformanceModel

        cs = PerformanceModel.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have performance_model prefix
        assert any("performance_model" in name for name in hp_names)

    def test_satzilla_config_space(self):
        """Test SATzilla configuration space."""
        from asf.selectors.satzilla import SATzilla

        cs = SATzilla.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have satzilla prefix
        assert any("satzilla" in name for name in hp_names)

    def test_isac_config_space(self):
        """Test ISAC configuration space."""
        from asf.selectors.isac import ISAC

        cs = ISAC.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have isac prefix and k_clusters
        assert any("isac" in name for name in hp_names)

    def test_snnap_config_space(self):
        """Test SNNAP configuration space."""
        from asf.selectors.snnap import SNNAP

        cs = SNNAP.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have snnap prefix
        assert any("snnap" in name for name in hp_names)

    def test_sunny_config_space(self):
        """Test SUNNY configuration space."""
        from asf.selectors.sunny import SUNNY

        cs = SUNNY.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Should have sunny prefix
        assert any("sunny" in name for name in hp_names)

    def test_collaborative_filtering_config_space(self):
        from asf.selectors.collaborative_filtering_selector import (
            CollaborativeFilteringSelector,
        )

        cs = CollaborativeFilteringSelector.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("collaborative_filtering" in name for name in hp_names)

    def test_cosine_config_space(self):
        from asf.selectors.cosine_selector import CosineSelector

        cs = CosineSelector.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("cosine" in name for name in hp_names)

    def test_cshc_config_space(self):
        from asf.selectors.cshc import CSHCSelector

        # CSHC requires candidate_selectors in definition if we want to test its full space
        # But ConfigurableMixin should handle partial definition gracefully or just use default.
        # But _define_hyperparameters expects candidate_selectors.
        # It handles None by returning empty.
        cs = CSHCSelector.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        # Without candidate_selectors, it might be empty or minimal.

    def test_isa_config_space(self):
        from asf.selectors.isa import ISA

        cs = ISA.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("isa" in name for name in hp_names)

    def test_joint_ranking_config_space(self):
        from asf.selectors.joint_ranking import JointRanking

        cs = JointRanking.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("joint_ranking" in name for name in hp_names)

    def test_meta_selector_config_space(self):
        from asf.selectors.meta_selector import MetaSelector

        cs = MetaSelector.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        # Expect empty if no candidates provided

    def test_osl_linear_config_space(self):
        from asf.selectors.osl_linear import OSLLinearSelector

        cs = OSLLinearSelector.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("osl_linear" in name for name in hp_names)

    def test_simple_ranking_config_space(self):
        from asf.selectors.simple_ranking import SimpleRanking

        cs = SimpleRanking.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("simple_ranking" in name for name in hp_names)

    def test_survival_analysis_config_space(self):
        pytest.importorskip("sksurv")
        from asf.selectors.survival_analysis import SurvivalAnalysis

        cs = SurvivalAnalysis.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)
        hp_names = {hp.name for hp in cs.values()}
        assert any("survival" in name for name in hp_names)

    def test_baseline_config_space(self):
        from asf.selectors.baselines import SingleBestSolver, VirtualBestSolver

        cs1 = SingleBestSolver.get_configuration_space()
        assert isinstance(cs1, ConfigurationSpace)
        cs2 = VirtualBestSolver.get_configuration_space()
        assert isinstance(cs2, ConfigurationSpace)


class TestSelectorConfigSpaceParams:
    """Test that selectors accept pre_prefix and parent_param correctly."""

    def test_pairwise_classifier_with_pre_prefix(self):
        """Test PairwiseClassifier configuration space with pre_prefix."""
        from asf.selectors.pairwise_classifier import PairwiseClassifier

        cs = PairwiseClassifier.get_configuration_space(pre_prefix="parent")

        hp_names = {hp.name for hp in cs.values()}
        # All names should start with parent:
        assert all(name.startswith("parent:") for name in hp_names)

    def test_pairwise_classifier_nested_model_config(self):
        """Test that nested model config spaces are included."""
        from asf.selectors.pairwise_classifier import PairwiseClassifier

        cs = PairwiseClassifier.get_configuration_space()

        hp_names = {hp.name for hp in cs.values()}
        # Should include nested random forest config
        rf_params = [name for name in hp_names if "rf_classifier" in name]
        assert len(rf_params) > 0, "Random forest hyperparameters should be included"
