from asf.selectors import PairwiseClassifier
from sklearn.ensemble import RandomForestClassifier
import asf.scenario.aslib_reader as aslib_reader


if __name__ == "__main__":
    scenario_path = "path/to/your/scenario"
    # Load the data
    features, performance, features_running_time, cv, feature_groups, maximize, budget = (
        aslib_reader.read_aslib_scenario(scenario_path)
    )

    for i in range(10):
        X_train, X_test = features[cv["fold"] != i + 1], features[cv["fold"] == i + 1]
        y_train, y_test = (
            performance[cv["fold"] != i + 1],
            performance[cv["fold"] == i + 1],
        )

        selector = PairwiseClassifier(model_class=RandomForestClassifier)

        selector.fit(X_train, y_train)

        predictions = selector.predict(X_test)

        print(predictions)
