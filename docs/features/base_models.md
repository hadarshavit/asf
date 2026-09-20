# Configuring base models

Pass a configured model instance through `estimator=` to model-based selectors
and EPM. Model parameters belong in the estimator constructor; selector
parameters such as `budget` and `maximize` belong in the selector constructor.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from asf.selectors import PairwiseClassifier, PerformanceModel
from asf.epm import EPM

selector = PairwiseClassifier(
    estimator=RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    budget=100,
)

performance_selector = PerformanceModel(
    estimator=RandomForestRegressor(n_estimators=200, random_state=42),
    budget=100,
)

epm = EPM(estimator=RandomForestRegressor(n_estimators=200, random_state=42))
```

## Cloning and model lifetime

ASF uses [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
to create a fresh model for each training task: each algorithm pair, individual
algorithm, ensemble member, or EPM fit. Repeating `fit` creates new models.
The supplied estimator is a configuration template and is never fitted by ASF.
For standard scikit-learn estimators, cloning copies constructor parameters
without copying learned state, even if the supplied estimator has already been
fitted. Passing a fitted estimator does not enable inference without training or
warm-start training in ASF.

An integer `random_state` is preserved in every clone. ASF does not automatically
assign different seeds to different models.

## Compatible estimators

The estimator must support `sklearn.base.clone` and provide the methods required
by the selector. Standard scikit-learn estimators, compatible pipelines, and
ASF's `SklearnWrapper` subclasses support cloning. Custom estimators can implement
the scikit-learn `get_params` constructor contract (usually via `BaseEstimator`)
or `__sklearn_clone__` on scikit-learn 1.3 or later. A custom clone implementation
must return a fresh, unfitted model without sharing mutable training state.
ASF does not fall back to deep-copying non-cloneable estimators.

Cloning does not make every estimator suitable for every selector:

- Classification selectors need classifiers, and regression selectors need regressors.
- `PairwiseClassifier` uses sample weights by default. Its estimator must accept
  `sample_weight` in `fit`, or use `use_weights=False` to disable weighting.
- Multi-target performance models need a regressor supporting multiple target columns.
- `SimpleRanking` and `DyadRanking` require ranking-specific fit arguments such
  as `qid`. `JointRanking` uses the matrix and algorithm-feature interface of
  `RankingMLP`. Survival selectors require survival prediction methods.

Raw PyTorch modules are not accepted: they do not provide the required
`fit`/`predict` and cloning interface. Use a cloneable training wrapper that
constructs its network and optimizer during `fit`, inferring dimensions from
the training data. ASF's `RegressionMLP` and `RankingMLP` support cloning through
their constructor parameters. Their trained networks and optimizer state are
not retained in a clone. When supplying a custom network with `model=`, its
initial weights are part of the configuration: each wrapper trains a separate
copy and leaves that supplied network untouched. `RankingMLP` can infer its
input dimension during `fit` when neither a network nor `input_size` is supplied.
Custom network and loss objects must support deep copying, as required for these
constructor parameters by scikit-learn's cloning mechanism.

```python
from asf.predictors import RegressionMLP, RankingMLP
from asf.selectors import PairwiseRegressor, JointRanking

regression_selector = PairwiseRegressor(
    estimator=RegressionMLP(epochs=100, compile_model=False),
)
ranking_selector = JointRanking(
    estimator=RankingMLP(epochs=100, compile=False),
)
```

### Built-in predictor compatibility

| Predictor family | Cloning | Additional requirements |
| --- | --- | --- |
| Scikit-learn wrappers, including XGBoost and its ranker | Supported | Use the appropriate classification, regression, or ranking selector; install optional XGBoost dependencies when needed. |
| `EPMRandomForest` | Supported, including forest and EPM parameters | Use `return_var=False` for selectors expecting one prediction per row. |
| `EPMExtraTrees` | Supported, including `log` and forest parameters | Always returns `(means, variances)`; requires a consumer or adapter that handles that output. |
| `RandomSurvivalForestWrapper` | Supported | Requires scikit-survival and structured survival targets; use `SurvivalAnalysis`. |
| `RegressionMLP` | Supported | Requires PyTorch; use regression targets and no sample weights. |
| `RankingMLP` | Supported | Requires PyTorch and algorithm features; use `JointRanking`. |

Cloneability alone does not adapt prediction shapes or training interfaces.
In particular, a `(means, variances)` predictor is not a drop-in replacement
for a scalar regressor in EPM or the ordinary regression selectors.

For SATzilla, `estimator=` configures the instance-label classifier and
`epm_estimator=` configures the per-algorithm regressors.

## Existing code and tuning

Existing `model_class=` selector arguments and EPM's `predictor_class=`,
`predictor_kwargs=`, and `predictor_config=` remain supported. These create
models through their existing class/factory path. ConfigSpace-based tuning
continues to use that path, including nested model hyperparameters.

When `estimator=` is supplied, it takes precedence over the class argument.
In EPM, combining `estimator=` with `predictor_config` or nonempty
`predictor_kwargs` raises an error; put the parameters on the estimator itself.
For new manually configured models, use `estimator=` instead of a class or
`functools.partial`.
