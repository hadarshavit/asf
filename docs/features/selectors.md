# Algorithm Selectors Overview

This page provides an overview of all available algorithm selectors in ASF. Click on any selector name below to jump to its description.

## Quick Index

| | | | |
|---|---|---|---|
| [APPS](#apps) | [Collaborative Filtering](#collaborative-filtering) | [Cosine](#cosine) | [CSHC](#cshc) |
| [Dyad Ranking](#dyad-ranking) | [Hybrid Decision Tree](#hybrid-decision-tree) | [ISA](#isa) | [ISAC](#isac) |
| [Joint Ranking](#joint-ranking) | [Meta Selector](#meta-selector) | [Multi-class](#multi-class) | [OSL Linear](#osl-linear) |
| [Pairwise Classifier](#pairwise-classifier) | [Pairwise Regressor](#pairwise-regressor) | [Performance Model](#performance-model) | [RPC Selector](#rpc-selector) |
| [SATzilla](#satzilla) | [Simple Ranking](#simple-ranking) | [SNNAP](#snnap) |
| [SUNNY](#sunny) | [Survival Analysis](#survival-analysis) | | |

---

## APPS

**Automatic Parallel Portfolio Selector**

APPS predicts runtime distribution for each algorithm and selects a parallel portfolio based on overlap with the best algorithm's predicted distribution.

- **Type**: Performance distribution-based
- **Returns**: Parallel Algorithm Portfolio
- **References**: [Kashgarani & Kotthoff (2020)](https://par.nsf.gov/biblio/10470673)

**Key Parameters**

- **`p_intersection`** (default: 0.82): Overlap threshold for portfolio inclusion. Higher values select smaller portfolios with only very similar performers. Lower values include more diverse algorithms.
- **`use_jackknife`** (default: True): Whether to use Jackknife method for uncertainty estimation. If False, uses bootstrap sampling instead.
- **`n_jackknife_folds`** (default: None): Number of folds for Jackknife. If None, uses leave-one-out (one fold per instance, training one model per instance).

**How It Works**

1. Trains ensemble of performance prediction models for each algorithm
2. Predicts mean runtime and uncertainty (std dev) for each algorithm on each instance
3. Identifies best predicted algorithm for each instance
4. Computes overlap between each algorithm's distribution and the best's distribution
5. Includes algorithms exceeding the `p_intersection` threshold in the portfolio

---

## Collaborative Filtering

**Collaborative Filtering Selector**

This selector uses matrix factorization to handle sparse performance data, treating algorithm selection as a collaborative filtering (recommendation) problem. It learns latent factors for both instances and algorithms.

- **Type**: Instance and performance-based
- **Returns**: Single
- **Key Features**: Sparse performance handling, warm/cold start support
- **References**: [Misir & Sebag (2016)](https://www.sciencedirect.com/science/article/pii/S0004370216301436)

**Key Parameters**

- **`n_components`** (default: 10): Number of latent factors for the matrix factorization. Higher values capture more complex instance-algorithm relationships but may overfit.
- **`n_iter`** (default: 100): Number of SGD iterations during training. More iterations lead to better convergence but longer training time.
- **`lr`** (default: 0.001): Learning rate for stochastic gradient descent. Controls step size during optimization.
- **`reg`** (default: 0.1): Regularization strength. Higher values prevent overfitting but may underfit the model.

**How It Works**

1. Decomposes the performance matrix into instance latent factors (U) and algorithm latent factors (V)
2. Uses stochastic gradient descent to optimize latent factors based on observed performance values
3. Learns instance and algorithm biases to capture global preferences
4. For cold-start instances, trains a regressor to predict latent factors from features
5. Predicts algorithm performance as: `mu + bias_instance + bias_algorithm + U·V^T`
6. Selects algorithm with lowest predicted performance

---

## Cosine

**Cosine Similarity-based Selector (AS-LLM)**

A neural network-based selector that learns embeddings for both instances and algorithms using LSTM and cosine similarity in the embedding space for selection. Requires algorithm features in addition to instance features.

- **Type**: Neural network-based
- **Returns**: Single
- **Key Features**: Dense embeddings, configurable neural architecture, LSTM fusion
- **References**: [Wu et al. (2023)](https://arxiv.org/abs/2311.13184)

**Key Parameters**

- **`embed_size`** (default: 50): Dimensionality of algorithm embeddings. Larger values capture more complex patterns but increase memory usage.
- **`num_hiddens`** (default: 50): Hidden units in LSTM. Controls capacity of the sequence modeling in algorithm encoding.
- **`num_layers`** (default: 2): Number of LSTM layers. More layers can capture hierarchical patterns.
- **`alpha`** (default: 0.9): Weight for learned LSTM features in fusion. Controls balance between learned and provided algorithm features.
- **`beta`** (default: 0.1): Weight for algorithm features in fusion.
- **`lr`** (default: 0.001): Learning rate for training. Controls optimization step size.

**How It Works**

1. Requires algorithm features (`algorithm_features`) to be set before fitting
2. Preprocesses instance features with optional normalization and imputation
3. Generates pairwise training data: for each instance, creates pairs (instance, algorithm) with labels indicating best algorithm
4. Builds instance embedding via MLP from normalized features
5. Builds algorithm embedding via:
   - LSTM processing of algorithm embeddings across all algorithms
   - Fusion of LSTM output with provided algorithm features using alpha/beta weighting
   - MLP projection to final embedding dimension
6. Computes cosine similarity between instance and algorithm embeddings
7. Concatenates embeddings + similarity through final MLP for binary classification
8. Selects algorithm with highest match probability

---

## CSHC

**Confidence-Switching Hybrid Constructor**

CSHC is a meta-selector that combines a primary selector with guardian models. Guardian models predict the success probability of the primary selector's choice and switch to a backup selector when confidence is low.

- **Type**: Meta-selector with confidence estimation
- **Returns**: Single
- **Key Features**: Confidence-based switching, ensemble design, guardian models
- **References**: [Malitsky et al. (2021)](https://link.springer.com/chapter/10.1007/978-3-642-44973-4_17)

**Key Parameters**

- **`primary_selector`** (required): The main algorithm selector to use.
- **`backup_selector`** (default: None): Optional backup selector used when primary's confidence is below threshold. If None, uses the highest-confidence algorithm globally.
- **`n_estimators`** (default: 100): Number of trees in each guardian RandomForest model. Higher values improve stability but increase training time.
- **`threshold_grid`** (default: linspace(0.01, 0.99, 99)): Grid of confidence thresholds to evaluate for switching. Finer grids = better threshold but slower.

**How It Works**

1. Trains the primary selector on the full training data
2. Performs cross-validation to generate out-of-fold predictions and true algorithm success labels
3. For each fold, trains guardian models (RandomForest) to predict which algorithms will succeed
4. Evaluates different confidence thresholds on OOF predictions to minimize ratio fn / tn
5. At prediction time, gets primary selector's choice and queries the guardian for confidence
6. Switches to backup (or best highest-confidence one) if confidence is below threshold
7. Otherwise uses primary selector's choice

**When to Use**

- When you have a good primary selector but want safety nets for uncertain decisions
- When backup selector is available and complementary to primary
- When false positives are acceptable but false negatives should be minimized
- Budget allows training multiple models (primary + guardians + optional backup)

---

## Dyad Ranking

**Dyad Ranking Selector**

Treats algorithm selection as learning dyadic (instance-algorithm pair) rankings. Creates training dyads by combining instance features with algorithm features, then trains a ranker to predict which algorithms are better for each instance.

- **Type**: Pairwise ranking-based
- **Returns**: Single
- **Key Features**: Algorithm features support, pairwise learning, explicit algorithm characterization
- **References**: [Tornede et al. (2019)](https://ris.uni-paderborn.de/download/15011/17060/ci_workshop_tornede.pdf)

**Key Parameters**

- **`model_class`** (default: XGBoostRankerWrapper): The ranking model to use. XGBoost is recommended for learning-to-rank tasks. Controls prediction quality and training speed.
- **`algorithm_features`** (required for best performance): DataFrame of algorithm features (n_algorithms × n_features). Each column represents an algorithm characteristic (e.g., configuration, structural property). If None, falls back to one-hot encoding (not recommended).
- **`n_pairs_per_instance`** (default: 10): Number of algorithm pairs to sample per training instance for pairwise learning. Higher values = more training data per instance but slower training.
- **`random_state`** (default: 42): Random seed for reproducible pairwise sampling.

**How It Works**

1. Requires algorithm features to characterize each algorithm (not just one-hot encoding)
2. Creates "dyads" by concatenating instance features with algorithm features
3. For each instance, samples algorithm pairs with different performance values
4. Creates training dyads: each dyad is (instance_features || algorithm_features, pairwise_rank)
5. Trains XGBoost ranker to predict relative rankings of algorithm dyads within each instance
6. At prediction time, scores all dyads for the query instance
7. Selects algorithm with lowest predicted score (best rank)

**When to Use**

- When you have meaningful algorithm features (parameters, structural properties)
- When pairwise comparisons between algorithms are more informative than absolute performance

---

## Hybrid Decision Tree

**Hybrid Decision Tree Selector (HARRIS)**

A forest of hybrid decision trees that combines regression and ranking losses for interpretable and efficient algorithm selection. Each tree makes splits optimizing both performance prediction (MSE) and algorithm ranking quality (Spearman correlation).

- **Type**: Tree-based ensemble with hybrid loss
- **Returns**: Single
- **Key Features**: Interpretability, no algorithm features needed, explicit ranking optimization
- **Paper**: [Fehring et al. (2022)](https://arxiv.org/pdf/2210.17341)

**Key Parameters**

- **`n_estimators`** (default: 100): Number of trees in the ensemble. More trees improve stability and accuracy but increase training time and memory.
- **`max_depth`** (default: 10): Maximum depth of each tree. Controls complexity; deeper trees can overfit, shallower trees may underfit.
- **`min_samples_split`** (default: 2): Minimum samples required to split a node. Higher values create simpler trees, lower values allow more complex patterns.
- **`lambda_param`** (default: 0.5): Balance between regression (λ=1) and ranking (λ=0) losses. λ=1 focuses on performance prediction, λ=0 focuses on ranking quality. 0.5 balances both.

**How It Works**

1. Bootstrap samples instances (with replacement) for each tree
2. Randomly selects subset of features for each tree
3. For each node, finds best split threshold that minimizes **hybrid loss**:
   - Regression component: MSE between predicted and actual performance
   - Ranking component: 1 - (Spearman correlation between predicted and true algorithm rankings)
   - Combined: λ × MSE + (1-λ) × Ranking_loss
4. Splits recursively until max_depth or min_samples_split reached
5. Stores regression label (mean performance) at each leaf node
6. At prediction time: ensemble averages predictions across all trees
7. Selects algorithm with lowest predicted runtime (or highest if maximizing)

**When to Use**

- When interpretability is important (can inspect decision tree splits)
- When you want ranking-aware selection (performance ranking matters, not just absolute values)
- When you need both good average performance prediction and good algorithm ranking

---

## ISA

**Instance-Specific Aspeed**

ISA uses k-nearest neighbors to find similar training instances, then applies Aspeed to compute optimal algorithm schedules for each neighborhood. Supports optional k-tuning via cross-validation.

- **Type**: k-NN based with schedule optimization
- **Returns**: Schedule
- **Key Features**: Instance-specific scheduling, Aspeed integration, optional k-tuning
- **Return Type**: Schedule
- **Dependencies**: clingo (ASP solver for Aspeed)
- **Paper**: [Lindauer et al. (2016)](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/16-LION-ASschedules.pdf)

**Key Parameters**

- **`k`** (default: 10): Number of nearest neighbors to use. Controls how many similar instances influence the schedule. Larger k = broader consensus, smaller k = more instance-specific.
- **`use_k_tuning`** (default: True): Enable automatic k-tuning via cross-validation to find optimal neighborhood size. Increases training time but often improves performance.
- **`n_folds`** (default: 5): Number of cross-validation folds for tuning k. More folds = better tuning but slower.
- **`k_candidates`** (default: [3, 5, 10, 15, 20]): Candidate k values to evaluate during tuning. Can be list, or "small"/"medium"/"broad" for predefined ranges.
- **`aspeed_cutoff`** (default: 30): Time limit (seconds) for internal Aspeed solver. Controls schedule optimization quality vs computation time.
- **`cores`** (default: 1): Number of cores for Aspeed solver. More cores speed up schedule optimization for large portfolios.
- **`random_state`** (default: 42): Random seed for reproducibility in k-NN and cross-validation.

**How It Works**

1. Filters out "trivial" instances: all algorithms solve in budget or all timeout (no learning signal)
2. Stores reduced set of "interesting" instances with k-NN index
3. If `use_k_tuning=True`: Cross-validates across candidate k values to find k that minimizes total runtime
4. For each query instance:
   - Finds k nearest neighbors in instance feature space
   - Extracts performance of algorithms on those k neighbors
   - Runs Aspeed to compute optimal schedule for that subset
   - Returns schedule with algorithm timing allocations
5. Aspeed optimizes: given neighbor performance matrix, compute time allocations across algorithms

**When to Use**

- When optimal portfolio allocation is important (not just selecting one algorithm)
- When you have computational budget for Aspeed optimization during prediction
- When you want a more sophisticated version of SUNNY with schedule optimization

---

## ISAC

**Instance-Specific Algorithm Configuration**

ISAC clusters problem instances in feature space and assigns each cluster its best-performing algorithm. This creates instance groups with tailored selections, enabling simple and fast per-cluster selection. (Original implementation also configures the solver for the given cluster)

- **Type**: Clustering-based
- **Returns**: Single
- **Key Features**: Unsupervised instance grouping, per-cluster optimization
- **References**: [Kadioglu et al. (2010)](https://www.researchgate.net/profile/Yuri-Malitsky/publication/220837402_ISAC_-_Instance-Specific_Algorithm_Configuration/links/02e7e52738cb135ccc000000/ISAC-Instance-Specific-Algorithm-Configuration.pdf)

**Key Parameters**

- **`clusterer`** (default: GMeansWrapper): The clustering algorithm to use. Can be a class, partial function, or instance. Options include KMeansWrapper, DBSCANWrapper, AgglomerativeClusteringWrapper, GMeansWrapper.
- **`clusterer_kwargs`** (default: None): Additional keyword arguments passed to the clusterer. Depends on chosen clustering algorithm (e.g., `n_clusters` for K-means, `eps` for DBSCAN).

**How It Works**

1. On training: Clusters instances using chosen clustering algorithm in feature space
2. For each cluster: Identifies the algorithm with lowest mean performance (best algorithm)
3. Stores mapping: cluster_id → best_algorithm
4. At prediction time: Assigns each query instance to nearest cluster
5. Returns the best algorithm associated with that cluster
6. If query instance doesn't fit any cluster (e.g., DBSCAN outlier), falls back to global best algorithm

**When to Use**

- When problem instances have natural groupings in feature space
- When you want interpretable selection (can inspect cluster characteristics)
- When fast prediction is critical (no model inference needed)
- For portfolios where per-cluster tuning is acceptable instead of per-instance
- G-means is good default (automatic cluster detection), otherwise try K-means with domain knowledge

---

## Joint Ranking

**Joint Instance and Algorithm Ranking**

A neural network-based selector that jointly learns rankings for both instances and algorithms. Uses a RankingMLP to predict algorithm performance while treating instance and algorithm embeddings symmetrically.

- **Type**: Joint ranking-based with neural networks
- **Returns**: Single
- **Key Features**: Symmetric instance-algorithm learning, ranking-focused neural architecture

**Key Parameters**

- **`model`** (default: RankingMLP): The neural network model for joint ranking. RankingMLP is the recommended choice; can be class or callable factory.
- **No other direct parameters**: Configuration is via the RankingMLP model kwargs (hidden layers, activation functions, etc.).

**How It Works**

1. Creates algorithm features via one-hot encoding if not provided (falls back)
2. Initializes RankingMLP model with input size = instance features + number of algorithms
3. For training: concatenates instance features with each algorithm's features to create (instance || algorithm) dyads
4. Trains RankingMLP to predict algorithm performance (ranking) for each instance
5. At prediction time: scores each algorithm by concatenating its features with query instance features
6. RankingMLP predicts scores for all algorithms
7. Selects algorithm with lowest predicted score (best ranking)

**When to Use**

- When ranking relationships between algorithms within an instance matter
- When you can afford neural network training overhead

---

## Meta Selector

**Meta-Selector (Ensemble of Selectors)**

An ensemble meta-learner that combines multiple base selectors. Uses out-of-fold cross-validation to measure each base selector's performance, then trains a meta-selector to choose the best base selector for each instance.

- **Type**: Meta-learning ensemble
- **Returns**: Single
- **Key Features**: Multiple base selectors, learned selector weighting, ensemble robustness

**Key Parameters**

- **`base_selectors`** (required): List of base selector instances or callable factory returning list. Each must have RETURN_TYPE='single'. Core component of the ensemble.
- **`meta_selector`** (required): The meta-selector instance used to choose among base selectors. Must have RETURN_TYPE='single'. Typically a simple model like Performance Model selector.


**How It Works**

1. Creates meta-performance dataset: measure each base selector's actual runtime on each training instance
2. Performs cross-validation:
   - For each fold: trains base selectors on training split
   - Gets predictions on validation split
   - Records actual runtime for predicted algorithm
   - Applies penalty (par_factor × budget) for failures/timeouts
3. Trains meta-selector to predict which base selector achieves best runtime for each instance
4. Retrains all base selectors on full training data
5. At prediction time:
   - Meta-selector predicts best base selector for query instance
   - Delegates prediction to that base selector
   - Returns the algorithm chosen by the selected base selector

**When to Use**

- When you have multiple diverse selectors and want automatic algorithm selection
- When combining selectors trained with different paradigms (k-NN, tree-based, neural, etc.)
- When you want automatic selector switching per instance
- When you can afford training overhead (trains N+1 selectors where N = base selectors)

---

## Multi-class

**Multi-class Classification Selector**

Treats algorithm selection as a multi-class classification problem where each algorithm is a distinct class. Trains a classifier to predict which algorithm is best (lowest performance) for each instance.

- **Type**: Classification-based (multi-class)
- **Returns**: Single
- **Key Features**: Direct classification, probabilistic outputs, simple training
- **Common Models**: Random Forest Classifier, XGBoost Classifier

**Key Parameters**

- **`model_class`** (default: RandomForestClassifierWrapper): The classifier class to use. Options include RandomForestClassifierWrapper (fast, robust) or XGBoostClassifierWrapper (more tunable). Controls prediction quality and interpretability.
- **No algorithm features used**: One-hot encoding or learned features not needed; each algorithm is just a class label.

**How It Works**

1. Creates classification labels: for each instance, label = index of best algorithm (lowest performance value)
2. Trains multi-class classifier (e.g., Random Forest or XGBoost) on (features → algorithm_class) mapping
3. At prediction time: Selects algorithm with highest predicted probability (best class)

**When to Use**

- When you want fast, simple baseline (no complex model, just standard classifier)
- When algorithm features are not available or not needed
- For quick prototyping and comparison with other methods

---

## OSL Linear

**Optimistic Superset Loss with Linear Models**

Uses linear regression with Optimistic Superset Loss (OSL) to handle censored runtime data (timeouts). Trains per-algorithm linear models that predict runtimes while treating timeout instances as censored observations.

- **Type**: Linear regression-based with censored data handling
- **Returns**: Single
- **Key Features**: Censored data handling, per-algorithm models, customizable optimization
- **References**: [Hanselle et al. (2021)](https://epub.ub.uni-muenchen.de/91670/1/_PAKDD__Superset_Learning_for_Algorithm_Selection_with_Right_Censored_Data.pdf)

**Key Parameters**

- **`reg`** (default: 0.0): L2 regularization strength for parameter shrinkage. Higher values prevent overfitting by penalizing large coefficients.
- **`optimizer_method`** (default: "L-BFGS-B"): Optimization algorithm from scipy.optimize.minimize. Options: "L-BFGS-B", "CG", "BFGS", "TNC", "SLSQP". Controls convergence speed and robustness.

**How It Works**

1. For each algorithm, fits a linear model: `f(x) = w·x + b` using instance features x
2. Handles censored data (timeouts)
3. Total loss = regression_loss + (L2_regularization if reg > 0)
4. Optimizes loss using scipy.optimize.minimize with specified method
5. Initialization: tries least squares on uncensored data, falls back to zeros
6. At prediction time: predicts runtime for each algorithm on query instance
7. Selects algorithm with lowest predicted runtime

**When to Use**

- For datasets with many timeouts (censored observations)
- When runtime prediction is more important than timeout probability

---

## Pairwise Classifier

**Pairwise Classification Selector**

Reduces algorithm selection to a series of pairwise binary classification tasks. For each pair of algorithms, trains a classifier to predict which one is better, then uses tournament-style selection to find the best algorithm.

- **Type**: Pairwise classification-based
- **Returns**: Single
- **Key Features**: Tournament-style selection, binary comparisons, pairwise learning
- **Common Models**: Logistic Regression, RBF SVM, Binary Random Forest

**Key Parameters**

- **`model_class`** (default: LogisticRegression or RandomForestClassifier): The binary classifier for pairwise comparisons. Must support binary classification. Controls learning capacity.
- **`use_weight`** (default: True): Instances are weighed by performance difference between the two algorithms in the pair.

**How It Works**

1. Creates pairwise comparison training data: for each instance, generates pairs of algorithms with different performance
2. For each pair (algo_i, algo_j), creates binary label: 1 if algo_i is better, 0 if algo_j is better
3. Trains a binary classifier per algorithm pair
4. At prediction time: for query instance, scores each pair of algorithms
5. Score all pairs and aggregate votes to find most-wins algorithm

**When to Use**

- When binary classification is simpler/faster than multi-class for your problem
- When you want explainable pairwise decisions (which algorithm beats which)

---

## Pairwise Regressor

**Pairwise Regression Selector**

Uses regression to predict relative performance differences between algorithm pairs. For each pair, predicts which algorithm is better by modeling the performance difference (algo_i - algo_j).

- **Type**: Pairwise regression-based
- **Returns**: Single
- **Key Features**: Performance difference prediction, tournament-based
- **Common Models**: Ridge Regression, Gradient Boosting, SVR

**Key Parameters**

- **`model_class`** (default: RandomForestRegressor): The regression model for pairwise differences. Predicts performance_i - performance_j. Controls smoothness vs complexity.
- **No direct hyperparameters**: Configuration through the chosen model_class and its parameters.

**How It Works**

1. Creates pairwise training data: for each instance, generates all pairs of algorithms
2. For each pair (algo_i, algo_j), creates regression target: performance[algo_i] - performance[algo_j]
3. Trains binary regression model for each algorithm pair to predict performance difference
4. At prediction time: for query instance, scores all algorithm pairings
5. Score all pairs, aggregate pairwise outcomes to get ranking

**When to Use**

- When you want to model fine-grained algorithm performance differences
- When regression is more natural than classification for your problem
- For portfolios where performance margins matter (not just relative ranking)
- When you want to predict not just which algorithm is better, but by how much

---

## Performance Model

**Performance Model-based Selector**

A base selector that learns performance prediction models and selects the algorithm with the lowest predicted runtime (or highest if maximizing).

- **Type**: Model-based prediction
- **Returns**: Single
- **Key Features**: Flexible regressors, optional multi-target, optional normalization
- **Superclass**: AbstractModelBasedSelector

**Key Parameters**

- **`model_class`** (default: RandomForestRegressorWrapper): Regression model wrapper used for performance prediction.
- **`use_multi_target`** (default: False): If True, fits one multi-target regressor for all algorithms. If False, fits per-algorithm models (or a joint regressor when algorithm features are provided).


**How It Works**

1. Optionally normalizes the performance matrix.
2. If `use_multi_target=True`, fits one regressor that predicts all algorithm runtimes at once.
3. Otherwise:
   - If no algorithm features are provided, fits one regressor per algorithm.
   - If algorithm features are provided, fits a joint regressor on concatenated (instance + algorithm) features.
4. Predicts a runtime for each algorithm on each query instance.
5. Selects the algorithm with the best predicted runtime (min or max depending on `maximize`).

**When to Use**

- When straightforward performance prediction is sufficient.
- As a baseline for comparing with more complex selectors.
- When algorithm features are available and you want a joint model.

---

## RPC Selector

**Ranking by Pairwise Comparison (RPC)**

Trains one binary classifier per algorithm pair, then aggregates pairwise wins using Copeland scores to rank algorithms for each instance.

- **Type**: Pairwise classification-based ranking
- **Returns**: Single (or Parallel when `top_n` > 1)
- **Key Features**: Pairwise models, Copeland aggregation, optional parallel portfolios

**Key Parameters**

- **`classifier_class`** (default: RandomForestClassifierWrapper): Base classifier class used for each algorithm pair.
- **`top_n`** (default: 1): Number of top algorithms to return. If > 1, returns a parallel portfolio with equal time slices.

**How It Works**

1. Creates all algorithm pairs.
2. For each pair, trains a binary classifier on instance features with labels indicating which algorithm is faster.
3. At prediction time, predicts pairwise winners for each instance.
4. Computes Copeland scores (number of pairwise wins) per algorithm.
5. Selects the top `top_n` algorithms by score.
6. If `top_n` > 1, splits the budget evenly across the selected algorithms.

**When to Use**

- When pairwise comparisons are a good fit for your data.
- When you want a simple voting-style aggregation across algorithms.
- When you want optional parallel portfolios via `top_n`.

---

## SATzilla

**SATzilla-like Selector**

Trains per-algorithm empirical performance models (EPM) that predict runtime. Optionally conditions predictions on instance labels (e.g., SAT/UNSAT). Uses iterative imputation to handle censored (timeout) runtimes.

- **Type**: EPM-based with optional label conditioning
- **Returns**: Single
- **Key Features**: Censored runtime handling, label-conditioned models, separate label classifier and EPM regressors
- **References**: [Xu et al. (2008)](https://www.jair.org/index.php/jair/article/view/10556)

**Key Parameters**

- **`label_classifier_model`** (default: RandomForestClassifierWrapper): The classifier class for predicting instance labels (e.g., SAT/UNSAT). Only used when instance labels are provided at training time. Ignored if no labels are given.
- **`epm_regressor_model`** (default: RidgeRegressorWrapper): The regressor class used internally by each EPM for per-algorithm runtime prediction. Controls the performance prediction model used for selection.

**How It Works**

1. If labels are provided: trains a label classifier to predict instance labels
2. Creates EPM (Empirical Performance Model) for each algorithm
3. If labels are provided: trains separate EPMs per algorithm-label combination on filtered data
4. At prediction time: 
   - If label classifier exists, predicts instance label(s)
   - Predicts runtime for each algorithm using the appropriate EPM
   - Combines predictions: `total_prediction = sum(label_probability[i] * epm_prediction[i])`
5. Selects algorithm with lowest predicted runtime

**When to Use**

- When you have optionally available instance labels (SAT/UNSAT, easy/hard, etc.)
- When handling incomplete performance data (censored runtimes)
- When per-algorithm runtime prediction with a specific regressor (e.g., Ridge) is preferred

---

## Simple Ranking

**Simple Ranking Selector**

A ranking-based selector that trains a learning-to-rank model combining instance features with algorithm features. Predicts algorithm rankings for each instance and selects the top-ranked algorithm.

- **Type**: Learning-to-rank with combined instance-algorithm features
- **Returns**: Single
- **Key Features**: XGBoost ranking model, algorithm feature support, pairwise ranking loss

**Key Parameters**

- **`model_class`** (default: XGBoostRankerWrapper): The ranking model to use. Controls how algorithm rankings are learned and predicted.
- **`algorithm_features`** (optional): DataFrame of algorithm features (n_algorithms × n_features). If None, automatically generates one-hot encoding of algorithm names.

**How It Works**

1. Creates a cross-product of instance features and algorithm features (or auto-generated one-hot encoding)
2. Calculates rank labels per instance: algorithms ranked by performance (1 = best, higher = worse)
3. Trains the ranking model to predict ranks for instance-algorithm pairs
4. At prediction time:
   - Creates cross-product of query instance features with all algorithm features
   - Gets rank predictions from the model for each instance-algorithm pair
   - Selects algorithm with the lowest predicted rank
5. Returns the selected algorithm

**When to Use**

- When you have explicit algorithm features describing their characteristics
- When pairwise ranking between algorithms is more informative than absolute performance

---

## SNNAP

**Solver-based Nearest Neighbor for Algorithm Portfolio**

Uses per-algorithm regression models trained on z-score normalized performance, then leverages Jaccard distance between predicted and actual algorithm rankings to find similar instances. Selects the best algorithm from the k most similar neighbors.

- **Type**: Regression + Jaccard-based instance similarity
- **Returns**: Single
- **Key Features**: Per-algorithm models, z-score normalization, Jaccard distance on top-n algorithms
- **References**: [Collautti et al. (2013)](https://link.springer.com/chapter/10.1007/978-3-642-40994-3_28)

**Key Parameters**

- **`k`** (default: 5): Number of nearest neighbors (similar instances) to consider. Determines neighborhood size for aggregating algorithm performance.
- **`top_n`** (default: 3): Number of top algorithms to use for computing Jaccard distance. Controls specificity of similarity metric: smaller values focus on strongest algorithms, larger values include broader preferences.

**How It Works**

1. During training:
   - Z-score normalizes performance per instance (row-wise normalization across algorithms)
   - Trains one RandomForest regressor per algorithm on (instance_features → scaled_runtime)

2. At prediction time:
   - Uses per-algorithm models to predict scaled runtimes for the query instance
   - Identifies the query's predicted top-n algorithms
   - For each training instance, gets its actual top-n algorithms from scaled performance
   - Computes Jaccard distance: 1 - |intersection| / |union| between query's top-n and each training instance's top-n
   - Selects k training instances with smallest Jaccard distance

3. From the k neighbors:
   - Finds the algorithm that was best (lowest actual runtime) across the most neighbors
   - Selects that algorithm using mean actual (unscaled) runtime as tie-breaker

**When to Use**

- When you want algorithm similarity based on predicted preferences, not raw features
- When per-algorithm models can capture instance-specific performance patterns
- When Jaccard distance on algorithm subsets better captures instance similarity than feature distance
- When you have diverse algorithm portfolio and want to leverage neighbor rankings
- For a hybrid approach combining model predictions with instance-based selection

---

## SUNNY

**SUNNY: Successive Nearest Neighbor Algorithm Portfolio**

A k-nearest neighbor-based selector that constructs algorithm schedules by finding similar training instances and greedily building portfolios from their best algorithms. Supports optional k-tuning (SUNNY-AS2) and schedule length tuning (TSunny).

- **Type**: k-NN based portfolio construction
- **Returns**: Schedule (ordered list of algorithms with time allocations)
- **Key Features**: Instance similarity-based, greedy set cover portfolio construction, optional k-tuning, budget-aware schedule allocation
- **References**: [Liu et al. (2022)](https://dl.acm.org/doi/10.1613/jair.1.13116)

**Key Parameters**

- **`k`** (default: 10): Number of nearest neighbors to use. Controls neighborhood size: smaller k = more instance-specific, larger k = more generalizing.
- **`use_v2`** (default: False): Enable SUNNY-AS2: tuned k via cross-validation. Searches across k_candidates to find k minimizing validation cost.
- **`n_folds`** (default: 5): Number of cross-validation folds for k-tuning. More folds = better tuning but slower.
- **`k_candidates`** (default: [3, 5, 7, 10, 20, 50]): List of k values to evaluate, or string "small"/"medium"/"broad" for predefined ranges.
- **`use_tsunny`** (default: False): Enable TSunny: tune the maximum schedule length via cross-validation.

**How It Works**

1. Builds a k-NN index on training instance features (Euclidean distance)
2. If `use_v2=True`: Cross-validates to find optimal k that minimizes total runtime on validation set
3. If `use_tsunny=True`: Cross-validates to find optimal maximum schedule length
4. For each query instance:
   - Finds k nearest neighbors in feature space
   - Applies greedy set cover algorithm on neighbors' performance data:
     - Recursively selects algorithm that covers most uncovered instances
     - Continues until budget exhausted or all instances covered
   - Allocates budget proportionally to how many instances each algorithm solves
   - Sorts algorithms by mean runtime in neighborhood
   - Returns schedule with (algorithm, time_allocation) tuples
5. Respects `algorithm_limit` or tuned limit to cap schedule length

**When to Use**

- When you want instance-based selection without training complex models
- When algorithm schedules (portfolios) are preferred over single selection
- When k-tuning could improve performance (set `use_v2=True`)
- When you want simple, interpretable baselines
- When computational budget for k-NN is acceptable
- For combining complementary algorithms: weak individually but strong together

---

## Survival Analysis

**Survival Analysis-based Algorithm Selector**

Uses survival analysis (Random Survival Forest) to model algorithm completion probability under budget constraints. Treats timeout as censored data and predicts which algorithms are most likely to complete within the budget.

- **Type**: Survival analysis with random forests
- **Returns**: Single algorithm or Schedule (depending on `use_schedule`)
- **Key Features**: Timeout-aware probability modeling, censored data handling, optional schedule optimization
- **References**: [Tornede et al. (2020)](https://arxiv.org/abs/2007.02816)

**Key Parameters**

- **`model_class`** (default: RandomSurvivalForestWrapper): The survival model class. Trains forests to predict completion probability.
- **`use_schedule`** (default: False): If True, returns optimized schedule maximizing completion probability. If False, returns best single algorithm.
- **`popsize`** (default: 20): Population size for differential evolution optimization (only used if use_schedule=True).
- **`maxiter`** (default: 150): Maximum iterations for schedule optimization (only used if use_schedule=True).
- **`dominance_resolution`** (default: 100): Resolution for dominance analysis grid when identifying non-dominated algorithms.

**How It Works**

1. **Training** (creates structured survival data):
   - For each (instance, algorithm) pair, creates row: [instance_features, algorithm_one_hot, status, runtime]
   - status=1 if algorithm completed within budget, 0 if timeout (censored)
   - runtime=actual_runtime if completed, runtime=budget if timeout
   - Trains RandomSurvivalForest to predict P(algorithm completes by time t)

2. **Single algorithm mode** (`use_schedule=False`):
   - For each algorithm, predicts survival probability at query instance
   - Selects algorithm with highest P(completes within budget)

3. **Schedule mode** (`use_schedule=True`):
   - Performs dominance analysis: identifies non-dominated algorithms
   - Uses differential evolution to optimize schedule:
     - Decision variables: which algorithms to include + time allocation for each
     - Objective: maximize P(at least one algorithm completes)
   - Returns schedule with (algorithm, time_allocation) tuples
   - Algorithms tried in order; first to complete satisfies the job

**When to Use**

- When timeout/budget constraints are critical and should be explicit in selection
- When you have significant censored data (timeouts are common)
- When completion probability is more meaningful than predicted runtime
- For modeling "time-to-completion" as a stochastic event

---

## Selecting the Right Selector

Here's a quick guide to choosing which selector fits your use case:

| Your Situation | Recommended Selectors |
|---|---|
| Fast, simple baseline | ISAC, Multi-class |
| Instance similarity matters (single) | ISAC, SNNAP |
| Instance similarity matters (schedule) | SUNNY, ISA |
| Performance predictions available | SATzilla, Performance Model |
| Have algorithm features | Dyad Ranking, Simple Ranking, Cosine, Joint Ranking |
| Limited/sparse data | Collaborative Filtering |
| Combine Selectors | Meta Selector, CSHC |
| Need interpretability | Hybrid Decision Tree |
| Budget/timeout constraints | Survival Analysis, OSL Linear |
| Portfolio/Schedule | SUNNY, ISA, APPS, Survival Analysis |

---

## API Reference

For detailed API documentation on each selector, visit the [Selectors API Reference](../api/selectors.md).
