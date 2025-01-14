from dataclasses import dataclass


@dataclass
class ScenarioMetadata:
    algorithms: list[str]
    features: list[str]
    performance_metric: str | list[str]
    maximize: bool
    budget: int | None

    def to_dict(self):
        """Converts the metadata into a dictionary format."""
        return {
            "algorithms": self.algorithms,
            "features": self.features,
            "performance_metric": self.performance_metric,
            "maximize": self.maximize,
            "budget": self.budget,
        }
