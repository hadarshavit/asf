[![PyPI - Version](https://img.shields.io/pypi/v/asf-lib)](https://pypi.org/project/asf-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/asf-lib)](https://pypi.org/project/asf-lib/)
[![License](https://img.shields.io/pypi/l/asf-lib?color=informational)](LICENSE)
[![Python application](https://github.com/hadarshavit/asf/actions/workflows/tests.yml/badge.svg)](https://github.com/hadarshavit/asf/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14957286.svg)](https://doi.org/10.5281/zenodo.14957286)
[![codecov](https://codecov.io/gh/hadarshavit/asf/graph/badge.svg?token=WOQ37XYZWG)](https://codecov.io/gh/hadarshavit/asf)

# Algorithm Selection Framework (ASF)

ASF is a lightweight yet powerful Python library for algorithm selection and empirical performance prediction. 
It implements various algorithm selection methods, along with algorithm pre-selection, pre-solving schedules and more features to easily create algorithm selection pipeline.
ASF is a modular framework that allows easy extensions to tailor made an algorithm selector for every use-case.
While ASF includes several built-in machine learning models through scikit-learn and XGBoost, it supports every model that complies with the scikit-learn API.
ASF also implements empirical performance prediction, allowing to use different performance scalings.

ASF is written in Python 3 and is intended to use with Python 3.10+. It requires only scikit-learn, NumPy and Pandas as basic requirements. More advanced features (such as hyperparameter optimisation) requires additional dependencies. 

You can find full documentation in: https://hadarshavit.github.io/asf/
## Installation

ASF is written in Python3 and requires Python version 3.10+.
The basic installation is lightweight and requires only NumPy, Pandas and scikit-learn.

ASF is currently tested on Linux machines. Mac and Windows (official) support will be released in the near future.

To install the base version run 
```bash
pip install asf
```

### Additional options

Additional options include:

- XGBoost model suppot `pip install asf[xgb]`
- PyTorch-based models `pip install asf[nn]`
- ASlib scenarios reading `pip install asf[aslib]`

## Quick start

The first step is to define a the data. It can be either NumPy array or Pandas DataFrame.
The data contains of (at least) two matrices. The first defines the instance features with a row for every instance and each column defines one feature.
The second is the performance data, for which every row describes an instance and each column the performance of a single algorithm.

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

selector = PairwiseClassifier(model_class=RandomForestClassifier)

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
Similarly, ASF minimises the performance by default. To change it, pass `maximize=True` to the selector.




## Cite Us

If you use ASF, please cite the Zenodo DOI. We are currently working on publishing a paper on ASF, but by then a Zenodo citation will do it. 

```bibtex
@software{ASF,
	author = {Hadar Shavit and Holger Hoos},
	doi = {10.5281/zenodo.15288151},
	title = {ASF: Algorithm Selection Framework},
	url = {https://doi.org/10.5281/zenodo.15288151},
	year = {in progress},
}
```
