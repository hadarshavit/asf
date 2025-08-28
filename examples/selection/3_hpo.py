from asf.selectors import PairwiseClassifier, PairwiseRegressor
from asf.predictors import SVMRegressorWrapper, SVMClassifierWrapper
from asf.selectors import tune_selector
import pandas as pd
import numpy as np


def get_data():
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
    performance = pd.DataFrame(data, columns=["algo1", "algo2", "algo3"])

    data = np.array(
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
    features = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])

    return features, performance


if __name__ == "__main__":
    # Load the data
    features, performance = get_data()

    selector = tune_selector(
        features,
        performance,
        selector_class=[PairwiseClassifier, PairwiseRegressor],
        budget=5000,
        runcount_limit=10,
    )

    # Fit the selector to the data
    selector.fit(features, performance)

    predictions = selector.predict(features)

    # Print the predictions
    print(predictions)

    # Load the data
    features, performance = get_data()

    # Setting configuration space manually
    selector = tune_selector(
        features,
        performance,
        selector_class=[
            (PairwiseClassifier, {"model_class": [SVMClassifierWrapper]}),
            (PairwiseRegressor, {"model_class": [SVMRegressorWrapper]}),
        ],
        budget=5000,
        runcount_limit=10,
    )

    # Fit the selector to the data
    selector.fit(features, performance)

    predictions = selector.predict(features)

    # Print the predictions
    print(predictions)
