---
hide:
  - navigation
  - toc
---

# Quick start

The first step is to define a the data. It can be either NumPy array or Pandas DataFrame.
The data contains of (at least) two matrices. The first defines the instance features with a row for every instance and each column defines one feature.
The second is the performance data, which for which every row describes an instance and each column the performance of a single algorithm.

Here, we define some toy data on three instances, three features and three algorithms.

```python
data = np.array(
    [
        [10, 5, 1],
        [20, 10, 2],
        [15, 8, 1.5],
    ]
)
features = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
performance = np.array(
    [
        [120, 100, 110],
        [140, 150, 130],
        [180, 170, 190],
    ]
)
performance = pd.DataFrame(data, columns=["algo1", "algo2", "algo3"])

```

We can then define a selector:
```python
from asf.selectors import PairwiseClassifier
from sklearn.ensemble import RandomForestClassifier

selector = PairwiseClassifier(
    estimator=RandomForestClassifier(n_estimators=200, random_state=42)
)

selector.fit(features, performance)
```

Next, we can use the selector to predict on unseen dta:
```
selector.predict(features)
```
Currently, ASF always returns the prediction in the ASlib format: a dictionary which has the instance id (row index, in case of a numpy array or the index of the row for a pandas dataframe) as keys and an array of tuples (predicted algorithm, budget).
The selectors has only one tuple in the array, which is the selected algorithm. 
An example output is:
```
{
    0: [('algo2', None)], 
    1: [('algo3', None)], 
    2: [('algo2', None)]
}
```

The budget is set by default to None. To change the budget, you can pass it as an argument for the selector initialisation.
Similarly, ASF minimises the performance by default. To change it, pass `maxmimize=True` to the selector.

