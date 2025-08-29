# SNNAP and ISAC Algorithm Examples

This file demonstrates how to use the newly implemented SNNAP and ISAC algorithms in ASF.

## SNNAP (SATzilla-like Nearest Neighbor Algorithm Portfolio)

SNNAP uses k-nearest neighbors to select algorithms based on the performance of algorithms on similar instances.

```python
from asf.selectors import SNNAP
import pandas as pd

# Load your data
features = pd.DataFrame(...)  # Instance features
performance = pd.DataFrame(...)  # Algorithm performance data

# Create and configure SNNAP selector
snnap = SNNAP(
    k=5,  # Number of nearest neighbors
    weight_strategy="distance",  # "uniform" or "distance"
    algorithm_selection_strategy="voting",  # "voting" or "best_performance"
    maximize=False  # Set to True for maximization problems
)

# Fit the selector
snnap.fit(features, performance)

# Make predictions
predictions = snnap.predict(features)
```

## ISAC (Instance-Specific Algorithm Configuration)

ISAC provides instance-specific algorithm selection using clustering or regression approaches.

### Clustering Mode

```python
from asf.selectors import ISAC

# Create ISAC selector in clustering mode
isac = ISAC(
    mode="clustering",
    n_clusters=10,  # Number of clusters
    distance_threshold=1.0,  # Distance threshold for cluster assignment
    fallback_strategy="best_overall",  # "best_overall" or "random"
    maximize=False
)

# Fit and predict
isac.fit(features, performance)
predictions = isac.predict(features)
```

### Regression Mode

```python
# Create ISAC selector in regression mode
isac = ISAC(
    mode="regression",
    fallback_strategy="best_overall",
    maximize=False
)

# Fit and predict
isac.fit(features, performance)
predictions = isac.predict(features)
```

## Integration with ASF Pipeline

Both algorithms can be used within the ASF pipeline system:

```python
from asf.selectors import SelectorPipeline, SNNAP, ISAC

# Create pipeline with SNNAP
pipeline = SelectorPipeline(
    selector=SNNAP(k=5),
    preprocessor=None,  # Add preprocessing if needed
    budget=100.0,
    maximize=False
)

# Or with ISAC
pipeline = SelectorPipeline(
    selector=ISAC(mode="clustering", n_clusters=8),
    budget=100.0,
    maximize=False
)
```