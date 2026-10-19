from typing import Any
from asf.epm import EPM
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def get_data() -> Any:
    performance = np.array(
        [
            120,
            140,
            180,
            160,
            250,
            230,
            300,
            280,
            350,
            330,
            400,
            380,
            450,
            430,
            500,
            480,
            550,
            530,
            600,
            580,
        ]
    )

    features = np.array(
        [
            [10, 5, 1],
            [20, 10, 2],
            [15, 8, 1.5],
            [25, 12, 2.5],
            [30, 15, 3],
            [35, 18, 3.5],
            [40, 20, 4],
            [45, 22, 4.5],
            [50, 25, 5],
            [55, 28, 5.5],
            [60, 30, 6],
            [65, 32, 6.5],
            [70, 35, 7],
            [75, 38, 7.5],
            [80, 40, 8],
            [85, 42, 8.5],
            [90, 45, 9],
            [95, 48, 9.5],
            [100, 50, 10],
            [105, 52, 10.5],
        ]
    )

    return features, performance


if __name__ == "__main__":
    # Load the data
    features, performance = get_data()

    # Initialize the selector
    epm = EPM(predictor_class=RandomForestRegressor)

    # Fit the selector to the data
    epm.fit(features, performance)

    predictions = epm.predict(features)

    # Print the predictions
    print(predictions)
