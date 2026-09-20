# Empirical performance prediction

To configure an EPM directly, pass an estimator instance:

```python
from asf.epm import EPM
from sklearn.ensemble import RandomForestRegressor

epm = EPM(estimator=RandomForestRegressor(n_estimators=200, random_state=42))
```

EPM clones the estimator on each fit and leaves the supplied instance untouched.
See [Configuring base models](base_models.md) for cloning requirements, including
custom and PyTorch models.

ASF allows to easily tune and train EPMs. For example, to tune an EPM:

```python
features, performance = get_data()

# Initialize the selector
epm = tune_epm(
    features,
    performance,
    model_class=RandomForestRegressorWrapper,
    features_preprocessing=None,
)

# Fit the selector to the data
epm.fit(features, performance)

predictions = epm.predict(features)

# Print the predictions
print(predictions)
```

By default, ASF uses log scaling of the performance. Other performance scaling methods include standarization, and inverse sigmoid. For more details check the [API](../api/preprocessing.md)
