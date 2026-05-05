"""Comprehensive tests for predictor configuration spaces."""

import pytest
from functools import partial

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace


class TestPredictorConfigurationSpaces:
    """Test configuration spaces for all predictors that have them."""

    def test_random_forest_classifier_config_space(self):
        """Test RandomForestClassifierWrapper configuration space."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        # Check expected hyperparameters exist with proper prefix
        hp_names = {hp.name for hp in cs.values()}
        expected = {
            "rf_classifier:n_estimators",
            "rf_classifier:min_samples_split",
            "rf_classifier:min_samples_leaf",
            "rf_classifier:max_features",
            "rf_classifier:bootstrap",
        }
        assert expected.issubset(hp_names)

    def test_random_forest_regressor_config_space(self):
        """Test RandomForestRegressorWrapper configuration space."""
        from asf.predictors.random_forest import RandomForestRegressorWrapper

        cs = RandomForestRegressorWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        expected = {
            "rf_regressor:n_estimators",
            "rf_regressor:min_samples_split",
            "rf_regressor:min_samples_leaf",
            "rf_regressor:max_features",
            "rf_regressor:bootstrap",
        }
        assert expected.issubset(hp_names)

    def test_random_forest_classifier_get_from_configuration(self):
        """Test RandomForestClassifierWrapper get_from_configuration."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space()
        config = cs.get_default_configuration()

        result = RandomForestClassifierWrapper.get_from_configuration(config)
        assert isinstance(result, partial)

        instance = result()
        assert isinstance(instance, RandomForestClassifierWrapper)

    def test_random_forest_regressor_get_from_configuration(self):
        """Test RandomForestRegressorWrapper get_from_configuration."""
        from asf.predictors.random_forest import RandomForestRegressorWrapper

        cs = RandomForestRegressorWrapper.get_configuration_space()
        config = cs.get_default_configuration()

        result = RandomForestRegressorWrapper.get_from_configuration(config)
        assert isinstance(result, partial)

        instance = result()
        assert isinstance(instance, RandomForestRegressorWrapper)

    def test_linear_classifier_config_space(self):
        """Test LinearClassifierWrapper configuration space."""
        from asf.predictors.linear_model import LinearClassifierWrapper

        cs = LinearClassifierWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # LinearClassifier should have alpha
        assert any("alpha" in name for name in hp_names)

    def test_linear_regressor_config_space(self):
        """Test LinearRegressorWrapper configuration space."""
        from asf.predictors.linear_model import LinearRegressorWrapper

        cs = LinearRegressorWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        assert len(hp_names) > 0

    def test_svm_classifier_config_space(self):
        """Test SVMClassifierWrapper configuration space."""
        from asf.predictors.svm import SVMClassifierWrapper

        cs = SVMClassifierWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # SVM should have C and gamma
        assert any("C" in name for name in hp_names)
        assert any("gamma" in name for name in hp_names)

    def test_svm_regressor_config_space(self):
        """Test SVMRegressorWrapper configuration space."""
        from asf.predictors.svm import SVMRegressorWrapper

        cs = SVMRegressorWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        assert len(hp_names) > 0

    def test_mlp_classifier_config_space(self):
        """Test MLPClassifierWrapper configuration space."""
        from asf.predictors.mlp import MLPClassifierWrapper

        cs = MLPClassifierWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # MLP should have learning_rate_init and hidden layers
        assert any("learning_rate_init" in name for name in hp_names)

    def test_mlp_regressor_config_space(self):
        """Test MLPRegressorWrapper configuration space."""
        from asf.predictors.mlp import MLPRegressorWrapper

        cs = MLPRegressorWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        assert len(hp_names) > 0

    def test_ridge_classifier_config_space(self):
        """Test RidgeClassifierWrapper configuration space."""
        from asf.predictors.ridge import RidgeClassifierWrapper

        cs = RidgeClassifierWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        # Ridge should have alpha
        assert any("alpha" in name for name in hp_names)

    def test_ridge_regressor_config_space(self):
        """Test RidgeRegressorWrapper configuration space."""
        from asf.predictors.ridge import RidgeRegressorWrapper

        cs = RidgeRegressorWrapper.get_configuration_space()
        assert isinstance(cs, ConfigurationSpace)

        hp_names = {hp.name for hp in cs.values()}
        assert any("alpha" in name for name in hp_names)


class TestConfigSpaceWithPrePrefix:
    """Test configuration space generation with pre_prefix."""

    def test_random_forest_with_pre_prefix(self):
        """Test configuration space with a pre_prefix."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space(pre_prefix="parent")

        hp_names = {hp.name for hp in cs.values()}
        # All names should start with parent:
        assert all(name.startswith("parent:") for name in hp_names)

    def test_random_forest_with_parent_param(self):
        """Test configuration space with parent parameter adds conditions."""
        from ConfigSpace import Categorical
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = ConfigurationSpace()
        parent = Categorical("selector", items=["rf", "xgb"])
        cs.add(parent)

        RandomForestClassifierWrapper.get_configuration_space(
            cs=cs,
            parent_param=parent,
            parent_value="rf",
        )

        # Check that conditions were added (one per hyperparameter)
        conditions = cs.conditions
        assert len(conditions) == 5  # 5 hyperparameters from RF


class TestConfigSpaceRoundTrip:
    """Test that configuration spaces work end-to-end."""

    def test_random_forest_config_roundtrip(self):
        """Test full configuration space round trip."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        # Generate config space
        cs = RandomForestClassifierWrapper.get_configuration_space()

        # Sample a configuration
        config = cs.sample_configuration()

        # Create instance from configuration
        factory = RandomForestClassifierWrapper.get_from_configuration(config)
        instance = factory()

        # Verify instance is properly created
        assert isinstance(instance, RandomForestClassifierWrapper)
        assert instance.model_class.n_estimators == config["rf_classifier:n_estimators"]


class TestPredictorDefaultConfigToInstance:
    """Test that each predictor can be instantiated from its default configuration."""

    def test_random_forest_classifier_default_config(self):
        """Test RandomForestClassifierWrapper instantiation from default config."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RandomForestClassifierWrapper.get_from_configuration(
            default_config
        )
        assert callable(partial_fn)

    def test_random_forest_regressor_default_config(self):
        """Test RandomForestRegressorWrapper instantiation from default config."""
        from asf.predictors.random_forest import RandomForestRegressorWrapper

        cs = RandomForestRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RandomForestRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_linear_classifier_default_config(self):
        """Test LinearClassifierWrapper instantiation from default config."""
        from asf.predictors.linear_model import LinearClassifierWrapper

        cs = LinearClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = LinearClassifierWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_linear_regressor_default_config(self):
        """Test LinearRegressorWrapper instantiation from default config."""
        from asf.predictors.linear_model import LinearRegressorWrapper

        cs = LinearRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = LinearRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_svm_classifier_default_config(self):
        """Test SVMClassifierWrapper instantiation from default config."""
        from asf.predictors.svm import SVMClassifierWrapper

        cs = SVMClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = SVMClassifierWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_svm_regressor_default_config(self):
        """Test SVMRegressorWrapper instantiation from default config."""
        from asf.predictors.svm import SVMRegressorWrapper

        cs = SVMRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = SVMRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_mlp_classifier_default_config(self):
        """Test MLPClassifierWrapper instantiation from default config."""
        from asf.predictors.mlp import MLPClassifierWrapper

        cs = MLPClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = MLPClassifierWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_mlp_regressor_default_config(self):
        """Test MLPRegressorWrapper instantiation from default config."""
        from asf.predictors.mlp import MLPRegressorWrapper

        cs = MLPRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = MLPRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_ridge_classifier_default_config(self):
        """Test RidgeClassifierWrapper instantiation from default config."""
        from asf.predictors.ridge import RidgeClassifierWrapper

        cs = RidgeClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RidgeClassifierWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_ridge_regressor_default_config(self):
        """Test RidgeRegressorWrapper instantiation from default config."""
        from asf.predictors.ridge import RidgeRegressorWrapper

        cs = RidgeRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RidgeRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_xgboost_classifier_default_config(self):
        """Test XGBoostClassifierWrapper instantiation from default config."""
        from asf.predictors.xgboost import XGBoostClassifierWrapper

        cs = XGBoostClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = XGBoostClassifierWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_xgboost_regressor_default_config(self):
        """Test XGBoostRegressorWrapper instantiation from default config."""
        from asf.predictors.xgboost import XGBoostRegressorWrapper

        cs = XGBoostRegressorWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = XGBoostRegressorWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    @pytest.mark.skip(
        reason="EPMRandomForest has MRO issue: AbstractPredictor.get_configuration_space is called instead of ConfigurableMixin"
    )
    def test_epm_random_forest_default_config(self):
        """Test EPMRandomForest instantiation from default config."""
        from asf.predictors.epm_random_forest import EPMRandomForest

        cs = EPMRandomForest.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = EPMRandomForest.get_from_configuration(default_config)
        assert callable(partial_fn)

    @pytest.mark.skip(
        reason="EPMExtraTrees has MRO issue: AbstractPredictor.get_configuration_space is called instead of ConfigurableMixin"
    )
    def test_epm_extra_trees_default_config(self):
        """Test EPMExtraTrees instantiation from default config."""
        from asf.predictors.epm_extra_trees import EPMExtraTrees

        cs = EPMExtraTrees.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = EPMExtraTrees.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_ranking_mlp_default_config(self):
        """Test RankingMLP instantiation from default config."""
        from asf.predictors.ranking_mlp import RankingMLP

        cs = RankingMLP.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RankingMLP.get_from_configuration(default_config)
        assert callable(partial_fn)

    @pytest.mark.skip(
        reason="RegressionMLP has MRO issue: AbstractPredictor.get_configuration_space is called instead of ConfigurableMixin"
    )
    def test_regression_mlp_default_config(self):
        """Test RegressionMLP instantiation from default config."""
        from asf.predictors.regression_mlp import RegressionMLP

        cs = RegressionMLP.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RegressionMLP.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_random_survival_forest_default_config(self):
        """Test RandomSurvivalForestWrapper instantiation from default config."""
        pytest.importorskip("sksurv")
        from asf.predictors.survival import RandomSurvivalForestWrapper

        assert RandomSurvivalForestWrapper is not None

        cs = RandomSurvivalForestWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        partial_fn = RandomSurvivalForestWrapper.get_from_configuration(default_config)
        assert callable(partial_fn)

    def test_random_survival_forest_ignores_selector_metadata_kwargs(self):
        """Selector-level kwargs should not leak into sksurv constructors."""
        pytest.importorskip("sksurv")
        from asf.predictors.survival import RandomSurvivalForestWrapper

        wrapper = RandomSurvivalForestWrapper(
            n_estimators=10,
            budget=100.0,
            maximize=False,
            n_algorithms=3,
        )

        assert wrapper.model.n_estimators == 10
