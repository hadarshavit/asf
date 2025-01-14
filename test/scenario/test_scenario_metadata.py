import pytest

from asf.scenario.scenario_metadata import ScenarioMetadata

@pytest.fixture
def dummy_metadata():
    return ScenarioMetadata(
        algorithms=["algo1", "algo2", "algo3"],
        features=["feature1", "feature2", "feature3"],
        performance_metric="time",
        maximize=False,
        budget=None,
    )

def test_scenario_metadata(dummy_metadata):
    """Test the ScenarioMetadata class."""
    metadata_dict = dummy_metadata.to_dict()
    # Check data to dict conversion
    assert metadata_dict == {'algorithms': ['algo1', 'algo2', 'algo3'],
                             'budget': None,
                             'features': ['feature1', 'feature2', 'feature3'],
                             'maximize': False,
                             'performance_metric': 'time'}
    # Check dict to data conversion
    metadata = ScenarioMetadata(**metadata_dict)
    assert metadata == dummy_metadata
