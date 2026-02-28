# JoinBoost Source Changes

This document summarises the changes made to JoinBoost and the adapter to enable incremental (constant-time) model updates
and PostgreSQL support.

## Summary of Changes

### JoinBoost Core (`eval/JoinBoost/src/joinboost/`)

**1. `joingraph.py` - Annotation Support**
- Added `self.annotations = {}` to `JoinGraph.__init__`
- Added `add_annotation()`, `get_annotations()`, `get_all_annotations()` methods to base `JoinGraph` class

**2. `cjt.py` - Inherit Annotations from JoinGraph**
- Changed `CJT.annotations` to `copy.deepcopy(join_graph.annotations)` instead of accepting annotations parameter
- Removed duplicate annotation methods (now inherited from `JoinGraph`)
- Added `select_conds=self.get_annotations(self.target_relation)` to `lift()` SPJA query

**3. `app.py` - Warm Start & Filtering**

*`DummyModel._fit()`*:
- Added `warm_start` and `filter_expression` parameters
- Applies `filter_expression` as annotation to target relation
- For warm_start: computes residuals (target - prediction) instead of raw target
- Passes `select_conds` to SPJA query to filter gradient/hessian calculation

*`DecisionTree._fit()`*:
- Propagates `warm_start` and `filter_expression` to parent
- Reuses preprocessor on warm_start (doesn't recreate)
- For warm_start: subtracts predicted values (from all existing trees) instead of constant mean

*`GradientBoosting._fit()`*:
- Propagates `warm_start` and `filter_expression` to parent
- Trains `incremental_estimators` trees, instead of `n_estimators` to improve performance
  and provide a hyperparameter allowing the user to choose between accuracy and train time

*`GradientBoosting._update_error()`*:
- **Critical optimisation**: Changed from per-leaf UPDATE queries to single CASE expression UPDATE
- Reduces MVCC table bloat (was causing linear slowdown from 400+ updates per batch)
- Updates semi-ring to reflect learning rate decay

**4. `executor.py` - Minor Fixes for PostgreSQL support**
- Added `DISTINCT_IDENTITY` aggregator support
- Fixed `COUNT` for multi-column args

**5. `aggregator.py` - CASE Expression Syntax Fix**
- Added default `1=1` condition to `Aggregator.CASE` when the condition list is empty.
- Prevents `psycopg2.errors.SyntaxError: syntax error at or near "THEN"` in PostgreSQL when a leaf node has no annotations (e.g., during initial training or specific tree structures).
- Add `__lt__(self, other)` for `QualifiedAttribute` to prevent bug where this can't be added to the priority queue

---

## Reasoning Behind Each Change

### JoinGraph Annotation System (`joingraph.py`)

**Problem**: JoinBoost had no way to filter which rows to process during training. Every `fit()` scanned the entire target table.

**Solution**: Added annotation system to `JoinGraph`:
```python
self.annotations = {}  # Maps relation -> list of SelectionExpressions
```

**Why here?** `JoinGraph` is the base class that both CJT and user code interact with. Putting annotations here means:
- CJT can inherit and use them in queries
- User code (adapter) can add filters before calling fit
- It's the natural place for "query metadata"

---

### CJT Inheriting Annotations (`cjt.py`)

**Problem**: CJT was accepting annotations as a constructor parameter, but we needed it to copy from the JoinGraph so filters propagate automatically.

**Solution**: 
```python
self.annotations = copy.deepcopy(join_graph.annotations)
```

**Why deepcopy?** Each CJT (one per tree node) needs its own copy because:
- Different nodes may add different filters (e.g., "feature < 5")
- We don't want a filter in one node to affect another

**Removed duplicate methods**: CJT had its own `add_annotation`, `get_annotations`, `get_all_annotations` - these are now inherited from `JoinGraph` (DRY principle).

---

### Filtering in `lift()` (`cjt.py`)

**Problem**: `lift()` materializes the target relation into a temp table. Without filtering, it copied ALL rows every time.

**Solution**:
```python
spja_data = SPJAData(
    ...,
    select_conds=self.get_annotations(self.target_relation)
)
```

**Why critical?** This is where the actual filtering happens. The `lift` operation:
1. Creates a temp table with `g` (gradient/residual) and `h` (hessian/count) columns
2. Without filtering, this table contains ALL historical data
3. With filtering, only new rows (matching `date > last_processed`) are included

This is the key to constant-time updates - the temp table size stays bounded by batch size, not total data.

---

### DummyModel Warm Start (`app.py`)

**Problem**: Even with filtering, `DummyModel._fit` was computing gradient/hessian on the entire unfiltered table.

**Solution**:
```python
select_conds = jg.get_annotations(jg.target_relation)
spja_data = SPJAData(..., select_conds=select_conds)
```

**Warm start residual calculation**:
```python
if warm_start and hasattr(self, "constant_"):
    pred_agg = self.get_prediction_aggregate()  # All existing tree predictions
    residual_agg = AggExpression(Aggregator.SUB, (jg.target_var, pred_agg))
```

**Why this matters**: In gradient boosting, we fit each tree to residuals (actual - predicted). For incremental updates:
- First batch: residuals = actual - mean (constant prediction)
- Subsequent batches: residuals = actual - (mean + all existing tree predictions)

Without computing residuals correctly, the new trees would be wrong.

---

### DecisionTree Warm Start (`app.py`)

**Preprocessor reuse**:
```python
if not warm_start:
    self.preprocessor = Preprocessor()  # Only create once
    self.preprocessor.add_step(RenameStep(...))
```

**Why?** The preprocessor handles column renaming for reserved words. We don't need to recreate it each batch - just re-run it on the (filtered) data.

**lift() with residuals**:
```python
if warm_start:
    exp = AggExpression(Aggregator.SUB, (self.cjt.target_var, pred_agg))
else:
    exp = AggExpression(Aggregator.SUB, (self.cjt.target_var, str(self.constant_)))
```

This computes the initial residuals that go into the temp table. For warm start, we subtract the full model prediction, not just the mean.

---

## GradientBoosting._fit `incremental_estimators` Parameter

Controls how many trees are added per incremental update.

### Purpose

Without this parameter, each incremental update adds `n_estimators` trees (e.g., 50). After 10 batches, the model would have 500 trees, leading to:
- Increasing prediction latency
- Larger model size
- More complex residual calculations

With `incremental_estimators=5`, each update adds only 5 trees:
- Batch 1: 50 trees (initial training uses `n_estimators`)
- Batch 2: 55 trees (adds 5)
- Batch 3: 60 trees (adds 5)

---

### GradientBoosting._update_error optimisation (`app.py`)

**This was the critical performance fix.**

**Original problem**: One UPDATE query per leaf node:
```python
for cur_cjt in self.leaf_nodes:
    self.cjt.exe.update_query(f"{g_col}={g_col}-({pred})", ...)
```

With 8 leaves x 50 trees = 400 UPDATEs per batch. Each UPDATE creates new row versions in Postgres MVCC. After 10 batches:
- 10,000 rows x 400 versions = 4 million row versions
- Sequential scans on bloated table become slower each batch
- Fit times: 9s -> 12s -> 15s -> 21s -> 78s -> 94s -> 116s -> 145s

**Solution**: Single UPDATE with CASE:
```python
case_conditions = []
for cur_cjt in self.leaf_nodes:
    case_conditions.append((pred, all_conds))

update_exp = f"{g_col} = {g_col} - (CASE WHEN ... THEN pred ELSE 0 END)"
self.cjt.exe._execute_query(f"UPDATE {target_relation} SET {update_exp}")
```

**Result**: Only 50 UPDATEs per batch (one per tree), reducing row versions 8x. Fit times became nearly constant: 7-8s across all batches.

---

### CASE Expression Syntax Fix (`aggregator.py`)

**Problem**: When `GradientBoosting._update_error` generated a `CASE` expression for a tree where a leaf had no conditions (common in the first tree or specific splits), it produced `WHEN THEN ...`, which is invalid SQL in PostgreSQL.

**Solution**:
```python
if not conds:
    conds = "1=1"
```

**Why?** In PostgreSQL, `CASE` requires a condition after `WHEN`. If JoinBoost determines a leaf applies to all rows (empty annotations), `1=1` ensures the `THEN` clause is executed correctly while maintaining valid syntax.

---

### Test Framework Robustness (`performance_tests.py`)

**Problem**: Multivariate datasets (like `wind_farms_minutely`) contain `'NULL'` strings for missing values. The benchmark runner's loss calculation failed with `ValueError: could not convert string to float: 'NULL'`.

**Solution**: Added explicit checks for `'NULL'` strings in `get_forecast_value_from_response` to treat them as `0.0`. This prevents crashes during long-running performance benchmarks when encountering sparse data.
