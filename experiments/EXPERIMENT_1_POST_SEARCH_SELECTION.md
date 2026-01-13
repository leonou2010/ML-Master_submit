# Experiment 1: Post-Search Selection for ML-Master

**Priority**: Phase 1 (Implement First)
**Risk Level**: Low
**Expected Impact**: High
**Estimated Time**: 2-3 days

## Overview

Test whether robust post-search selection strategies can reduce overfitting in ML-Master's MCTS-generated solutions. This is the **lowest-risk, highest-priority** experiment because it:
- Requires minimal changes to core MCTS logic
- Is pure post-processing after search completes
- Has clean separation of concerns
- Easy to A/B test

## Motivation

ML-Master's MCTS optimizes for validation metric during search. The final node selection uses `journal.get_best_node()` which returns `max(nodes, key=lambda n: n.metric)`. This greedy selection may:
1. Choose solutions that overfit to the validation set
2. Ignore cross-validation variance (solution stability)
3. Miss solutions with better worst-case performance
4. Not consider generalization indicators

## Design

### Core Principle: Minimal Intervention

**Control = Current ML-Master**:
- MCTS search unchanged
- UCT-based node selection during search unchanged
- Final selection: `best_valid` (highest validation metric)

**Treatment = Post-Search Selection Only**:
- MCTS search identical (same seeds, same UCT, same budgets)
- Only difference: final node selection strategy
- No changes to search rewards or expansion

### Treatment Selectors

#### 1. maximin (Conservative)
```python
def select_maximin(journal: Journal) -> MCTSNode:
    """Select node with best worst-case CV fold"""
    candidates = [n for n in journal.good_nodes if n.cv_folds]
    if not candidates:
        return journal.get_best_node()  # Fallback
    return max(candidates, key=lambda n: min(n.cv_folds))
```

**Intuition**: Choose the solution that performs best in its worst fold. Robustness over peak performance.

#### 2. stat_maximin (Statistically-Constrained Maximin)
```python
def select_stat_maximin(journal: Journal, z_threshold=0.5) -> MCTSNode:
    """Filter to elite set, then apply maximin"""
    candidates = [n for n in journal.good_nodes if n.cv_mean and n.cv_folds]
    if not candidates:
        return journal.get_best_node()

    # Population statistics
    cv_means = [n.cv_mean for n in candidates]
    mean = np.mean(cv_means)
    std = np.std(cv_means)

    # Elite filter: cv_mean >= population_mean + z * population_std
    elite = [n for n in candidates if n.cv_mean >= mean + z_threshold * std]
    if not elite:
        elite = candidates  # Fallback if filter too strict

    # Among elite, select maximin
    return max(elite, key=lambda n: min(n.cv_folds))
```

**Intuition**: Combine performance filtering (must be in top tier) with robustness (best worst-case among top tier).

#### 3. mean_minus_k_std (Lower Confidence Bound)
```python
def select_mean_minus_k_std(journal: Journal, k=1.0) -> MCTSNode:
    """Select by cv_mean - k * cv_std"""
    candidates = [n for n in journal.good_nodes if n.cv_mean and n.cv_std]
    if not candidates:
        return journal.get_best_node()

    return max(candidates, key=lambda n: n.cv_mean - k * n.cv_std)
```

**Intuition**: Penalize high within-solution variance. Prefer stable solutions over volatile high-performers.

---

## Implementation Plan

### Step 1: Enhance MCTSNode with CV Metrics

**File**: `search/mcts_node.py`

```python
@dataclass(eq=False)
class MCTSNode(Node):
    # Existing fields...
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    # ... other existing fields

    # NEW: Cross-validation metrics (for post-search selection)
    valid_metric: float | None = field(default=None, kw_only=True)  # Explicit validation metric
    train_metric: float | None = field(default=None, kw_only=True)  # Training metric (if printed)
    test_metric: float | None = field(default=None, kw_only=True)   # Test metric (if available)
    cv_mean: float | None = field(default=None, kw_only=True)       # Mean across CV folds
    cv_std: float | None = field(default=None, kw_only=True)        # Std dev across CV folds
    cv_folds: list[float] | None = field(default=None, kw_only=True) # Individual fold scores

    # NEW: Submission tracking
    submission_csv_path: str | None = field(default=None, kw_only=True)
    submission_csv_sha256: str | None = field(default=None, kw_only=True)
```

**Why these fields?**
- `cv_mean`, `cv_std`, `cv_folds`: Required for all 3 selectors
- `train_metric`, `test_metric`: For generalization gap analysis
- `submission_csv_*`: For tracking which solutions differ

**Impact on MCTS**: None - these are write-only during execution, read-only during post-search selection.

### Step 2: Parse CV Metrics from Execution Output

**File**: `utils/metric.py`

Create new function:
```python
import re
import numpy as np

def parse_cv_metrics(term_out: str) -> dict:
    """
    Extract cross-validation metrics from execution output.

    Looks for patterns like:
    - "CV scores: [0.85, 0.87, 0.83, 0.88, 0.86]"
    - "Cross-validation scores: 0.85, 0.87, 0.83, 0.88, 0.86"
    - "Mean CV: 0.858, Std CV: 0.018"
    - "Train: 0.95, Valid: 0.86, Test: 0.84"
    - "Fold 1: 0.85, Fold 2: 0.87, Fold 3: 0.83, Fold 4: 0.88, Fold 5: 0.86"

    Returns:
        dict with keys: cv_mean, cv_std, cv_folds, train_metric, valid_metric, test_metric
    """
    result = {}

    # Pattern 1: List format [score1, score2, ...]
    cv_list_pattern = r'(?:CV|cv|cross.validation).*?scores?.*?\[([\d\.,\s]+)\]'
    match = re.search(cv_list_pattern, term_out, re.IGNORECASE)
    if match:
        folds_str = match.group(1)
        folds = [float(x.strip()) for x in folds_str.split(',')]
        result['cv_folds'] = folds
        result['cv_mean'] = np.mean(folds)
        result['cv_std'] = np.std(folds)

    # Pattern 2: Comma-separated format
    cv_comma_pattern = r'(?:CV|cv|cross.validation).*?scores?:\s*([\d\.,\s]+)(?:\n|$)'
    match = re.search(cv_comma_pattern, term_out, re.IGNORECASE)
    if match and 'cv_folds' not in result:
        folds_str = match.group(1)
        folds = [float(x.strip()) for x in folds_str.split(',') if x.strip()]
        if len(folds) >= 2:  # Minimum 2 folds to be valid
            result['cv_folds'] = folds
            result['cv_mean'] = np.mean(folds)
            result['cv_std'] = np.std(folds)

    # Pattern 3: Mean and Std explicitly stated
    mean_pattern = r'(?:mean|average).*?(?:CV|cv):\s*([\d\.]+)'
    std_pattern = r'(?:std|standard.deviation).*?(?:CV|cv):\s*([\d\.]+)'
    mean_match = re.search(mean_pattern, term_out, re.IGNORECASE)
    std_match = re.search(std_pattern, term_out, re.IGNORECASE)
    if mean_match:
        result['cv_mean'] = float(mean_match.group(1))
    if std_match:
        result['cv_std'] = float(std_match.group(1))

    # Pattern 4: Train/Valid/Test metrics
    train_pattern = r'(?:train|training).*?metric.*?:\s*([\d\.]+)'
    valid_pattern = r'(?:valid|validation).*?metric.*?:\s*([\d\.]+)'
    test_pattern = r'(?:test).*?metric.*?:\s*([\d\.]+)'

    train_match = re.search(train_pattern, term_out, re.IGNORECASE)
    valid_match = re.search(valid_pattern, term_out, re.IGNORECASE)
    test_match = re.search(test_pattern, term_out, re.IGNORECASE)

    if train_match:
        result['train_metric'] = float(train_match.group(1))
    if valid_match:
        result['valid_metric'] = float(valid_match.group(1))
    if test_match:
        result['test_metric'] = float(test_match.group(1))

    # Pattern 5: Fold-by-fold format
    fold_pattern = r'Fold\s+\d+:\s*([\d\.]+)'
    fold_matches = re.findall(fold_pattern, term_out, re.IGNORECASE)
    if fold_matches and 'cv_folds' not in result:
        folds = [float(x) for x in fold_matches]
        result['cv_folds'] = folds
        result['cv_mean'] = np.mean(folds)
        result['cv_std'] = np.std(folds)

    return result
```

### Step 3: Populate CV Metrics in Agent

**File**: `agent/mcts_agent.py`

In the execution result processing (after `_review()` is called):

```python
def step(self, exec_callback: ExecCallbackType) -> MCTSNode:
    # ... existing code ...

    # After execution
    exec_result = exec_callback(node.code, True)
    node._term_out = exec_result.output

    # Existing review call
    review_dict = self._review(node)

    # NEW: Parse CV metrics
    from utils.metric import parse_cv_metrics
    cv_metrics = parse_cv_metrics(node.term_out)

    # Populate node with CV data
    node.cv_mean = cv_metrics.get('cv_mean')
    node.cv_std = cv_metrics.get('cv_std')
    node.cv_folds = cv_metrics.get('cv_folds')
    node.train_metric = cv_metrics.get('train_metric')
    node.valid_metric = cv_metrics.get('valid_metric')
    node.test_metric = cv_metrics.get('test_metric')

    # Track submission file
    if review_dict.get('has_csv_submission'):
        submission_path = os.path.join(self.cfg.workspace_dir, "submission", "submission.csv")
        if os.path.exists(submission_path):
            node.submission_csv_path = submission_path
            # Compute hash for tracking uniqueness
            import hashlib
            with open(submission_path, 'rb') as f:
                node.submission_csv_sha256 = hashlib.sha256(f.read()).hexdigest()

    # ... rest of existing code ...
```

**Key Points**:
- Parsing happens after execution, before MCTS backpropagation
- Does NOT modify `node.metric` (MCTS reward signal unchanged)
- Only populates new CV fields for post-search use

### Step 4: Implement Post-Search Selectors

**File**: `utils/post_search.py` (new file)

```python
"""
Post-search selection strategies for final node selection.

All strategies consume the same journal produced by MCTS search.
The control strategy (best_valid) replicates current ML-Master behavior.
Treatment strategies use CV statistics to select more robust solutions.
"""

import numpy as np
from typing import Literal
from search.journal import Journal
from search.mcts_node import MCTSNode
from utils.config_mcts import Config
import logging

logger = logging.getLogger("ml-master")


def select_final_node(journal: Journal, cfg: Config) -> MCTSNode:
    """
    Select the final node based on the configured strategy.

    Args:
        journal: Completed search journal
        cfg: Configuration with post_search.selection strategy

    Returns:
        Selected MCTSNode
    """
    strategy = cfg.post_search.selection
    logger.info(f"Selecting final node using strategy: {strategy}")

    if strategy == "best_valid":
        return _select_best_valid(journal)
    elif strategy == "maximin":
        return _select_maximin(journal)
    elif strategy == "stat_maximin":
        return _select_stat_maximin(journal, cfg.post_search.z_threshold)
    elif strategy == "mean_minus_k_std":
        return _select_mean_minus_k_std(journal, cfg.post_search.k)
    else:
        logger.warning(f"Unknown selection strategy: {strategy}, falling back to best_valid")
        return _select_best_valid(journal)


def _select_best_valid(journal: Journal) -> MCTSNode:
    """Control strategy: highest validation metric (current ML-Master)"""
    return journal.get_best_node(only_good=True)


def _select_maximin(journal: Journal) -> MCTSNode:
    """Select node with best worst-case CV fold"""
    candidates = [n for n in journal.good_nodes if n.cv_folds and len(n.cv_folds) > 0]

    if not candidates:
        logger.warning("No nodes with cv_folds, falling back to best_valid")
        return _select_best_valid(journal)

    selected = max(candidates, key=lambda n: min(n.cv_folds))
    logger.info(f"Selected node {selected.id} with min_fold={min(selected.cv_folds):.4f}")
    return selected


def _select_stat_maximin(journal: Journal, z_threshold: float = 0.5) -> MCTSNode:
    """Statistically-constrained maximin selection"""
    candidates = [n for n in journal.good_nodes
                 if n.cv_mean is not None and n.cv_folds and len(n.cv_folds) > 0]

    if not candidates:
        logger.warning("No nodes with cv_mean and cv_folds, falling back to best_valid")
        return _select_best_valid(journal)

    # Population statistics
    cv_means = [n.cv_mean for n in candidates]
    mean = np.mean(cv_means)
    std = np.std(cv_means)

    logger.info(f"Population: mean={mean:.4f}, std={std:.4f}")

    # Elite filter
    threshold = mean + z_threshold * std
    elite = [n for n in candidates if n.cv_mean >= threshold]

    if not elite:
        logger.warning(f"No nodes pass elite threshold {threshold:.4f}, using all candidates")
        elite = candidates

    # Among elite, select maximin
    selected = max(elite, key=lambda n: min(n.cv_folds))
    logger.info(f"Selected node {selected.id} from {len(elite)} elite nodes, "
               f"cv_mean={selected.cv_mean:.4f}, min_fold={min(selected.cv_folds):.4f}")
    return selected


def _select_mean_minus_k_std(journal: Journal, k: float = 1.0) -> MCTSNode:
    """Select by cv_mean - k * cv_std (lower confidence bound)"""
    candidates = [n for n in journal.good_nodes
                 if n.cv_mean is not None and n.cv_std is not None]

    if not candidates:
        logger.warning("No nodes with cv_mean and cv_std, falling back to best_valid")
        return _select_best_valid(journal)

    selected = max(candidates, key=lambda n: n.cv_mean - k * n.cv_std)
    lcb = selected.cv_mean - k * selected.cv_std
    logger.info(f"Selected node {selected.id} with LCB(k={k})={lcb:.4f} "
               f"(mean={selected.cv_mean:.4f}, std={selected.cv_std:.4f})")
    return selected


def compute_selection_summary(journal: Journal, selected_node: MCTSNode, cfg: Config) -> dict:
    """
    Compute summary statistics for logging and analysis.

    Returns dict with:
    - strategy: which selector was used
    - selected_node_id, selected_metric, selected_cv_mean, etc.
    - population_stats: aggregates over all good nodes
    - generalization_gaps: if train/test metrics available
    """
    good_nodes = journal.good_nodes

    summary = {
        "strategy": cfg.post_search.selection,
        "selected_node_id": selected_node.id,
        "selected_node_step": selected_node.step,
        "selected_metric": selected_node.metric.value if selected_node.metric else None,
        "selected_cv_mean": selected_node.cv_mean,
        "selected_cv_std": selected_node.cv_std,
        "selected_cv_folds": selected_node.cv_folds,
        "selected_train_metric": selected_node.train_metric,
        "selected_valid_metric": selected_node.valid_metric,
        "selected_test_metric": selected_node.test_metric,
    }

    # Population statistics
    cv_means = [n.cv_mean for n in good_nodes if n.cv_mean is not None]
    cv_stds = [n.cv_std for n in good_nodes if n.cv_std is not None]
    metrics = [n.metric.value for n in good_nodes if n.metric and n.metric.value is not None]

    if cv_means:
        summary["population_cv_mean"] = float(np.mean(cv_means))
        summary["population_cv_std"] = float(np.std(cv_means))
        summary["population_cv_min"] = float(np.min(cv_means))
        summary["population_cv_max"] = float(np.max(cv_means))

    if cv_stds:
        summary["population_within_solution_std_mean"] = float(np.mean(cv_stds))
        summary["population_within_solution_std_std"] = float(np.std(cv_stds))

    if metrics:
        summary["population_metric_mean"] = float(np.mean(metrics))
        summary["population_metric_std"] = float(np.std(metrics))

    # Generalization gaps
    if selected_node.train_metric and selected_node.valid_metric:
        summary["gap_train_valid"] = selected_node.train_metric - selected_node.valid_metric

    if selected_node.train_metric and selected_node.test_metric:
        summary["gap_train_test"] = selected_node.train_metric - selected_node.test_metric

    return summary
```

### Step 5: Update Configuration

**File**: `utils/config_mcts.yaml`

```yaml
# ... existing config ...

# NEW: Post-search selection configuration
post_search:
  # Selection strategy: best_valid | maximin | stat_maximin | mean_minus_k_std
  selection: best_valid  # Control arm uses this

  # Parameters for stat_maximin
  z_threshold: 0.5  # Elite filter: cv_mean >= population_mean + z * population_std

  # Parameters for mean_minus_k_std
  k: 1.0  # Lower confidence bound: cv_mean - k * cv_std

  # Export detailed selection info
  export_selection_summary: true
```

**File**: `utils/config_mcts.py`

```python
@dataclass
class PostSearchConfig:
    selection: Literal["best_valid", "maximin", "stat_maximin", "mean_minus_k_std"] = "best_valid"
    z_threshold: float = 0.5
    k: float = 1.0
    export_selection_summary: bool = True


@dataclass
class Config:
    # ... existing fields ...
    post_search: PostSearchConfig = field(default_factory=PostSearchConfig)
```

### Step 6: Integrate into Main Loop

**File**: `main_mcts.py`

```python
def run():
    # ... existing setup ...

    # After MCTS search completes
    agent.search()

    # NEW: Post-search selection
    from utils.post_search import select_final_node, compute_selection_summary

    final_node = select_final_node(journal, cfg)

    # Log selection
    logger.info(f"Final node selected: {final_node.id} (step {final_node.step})")
    logger.info(f"Final metric: {final_node.metric.value if final_node.metric else None}")

    # Export selection summary
    if cfg.post_search.export_selection_summary:
        selection_summary = compute_selection_summary(journal, final_node, cfg)

        summary_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}", "final_selection.json")
        with open(summary_path, 'w') as f:
            json.dump(selection_summary, f, indent=2)

        logger.info(f"Selection summary saved to {summary_path}")

    # Copy final submission
    if final_node.submission_csv_path and os.path.exists(final_node.submission_csv_path):
        final_submission_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}", "final_submission.csv")
        shutil.copy(final_node.submission_csv_path, final_submission_path)
        logger.info(f"Final submission copied to {final_submission_path}")

    # ... existing save_run and cleanup ...
```

---

## A/B Test Design

### Experimental Arms

1. **Control**: `post_search.selection=best_valid`
   - Current ML-Master behavior
   - Highest validation metric

2. **Treatment 1**: `post_search.selection=maximin`
   - Conservative: best worst-case fold

3. **Treatment 2**: `post_search.selection=stat_maximin`
   - Balanced: elite filter + maximin
   - Parameters: `z_threshold=0.5`

4. **Treatment 3**: `post_search.selection=mean_minus_k_std`
   - Lower confidence bound
   - Parameters: `k=1.0`

### Datasets

Use MLE-Bench datasets (same as AIDE experiments where applicable):
- **Primary** (overlap with AIDE): house_prices, spaceship-titanic, wine_quality
- **MLE-Bench specific**: plant-pathology-2020-fgvc7, new-york-city-taxi-fare-prediction, etc.

**Target**: 10 datasets × 3 seeds = 30 runs per arm = 120 runs total (4 arms)

### Configuration Template

Control arm (`configs/control.yaml`):
```yaml
post_search:
  selection: best_valid
```

Treatment arms:
```yaml
# Treatment 1
post_search:
  selection: maximin

# Treatment 2
post_search:
  selection: stat_maximin
  z_threshold: 0.5

# Treatment 3
post_search:
  selection: mean_minus_k_std
  k: 1.0
```

### Metrics

**Primary**:
- Generalization gap: `gap_train_valid`, `gap_train_test`
- Final test performance (when available)

**Secondary**:
- CV variance: `selected_cv_std`
- Population statistics: `population_cv_mean`, `population_cv_std`
- Agreement with baseline: How often does treatment select same node as control?

**Analysis**:
- Paired t-tests (same dataset + seed)
- Bootstrap 95% CIs
- Effect sizes (Cohen's d)

---

## Runner Script

**File**: `experiments/run_post_search_ab.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR=/home/ka3094/ML-Master_submit
DATASET_DIR=/path/to/mle-bench
OUT_ROOT=/home/ka3094/ML-Master_submit/experiments/post_search_ab

# Experiment settings
STEPS=500
SEEDS="0,1,2"
DATASETS="plant-pathology-2020-fgvc7,house-prices,..."

# Arms
ARMS=(
    "best_valid"
    "maximin"
    "stat_maximin"
    "mean_minus_k_std"
)

cd $AGENT_DIR

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"

for arm in "${ARMS[@]}"; do
    for ds in "${DATASET_ARR[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
            echo "== ARM=${arm} | DATASET=${ds} | SEED=${seed} =="

            OUT_DIR="${OUT_ROOT}/${arm}/${ds}_seed${seed}"
            mkdir -p "$OUT_DIR"

            python main_mcts.py \
                dataset_dir="${DATASET_DIR}/${ds}" \
                exp_name="${arm}-${ds}-seed${seed}" \
                agent.steps=$STEPS \
                log_dir="${OUT_DIR}/logs" \
                workspace_dir="${OUT_DIR}/workspace" \
                post_search.selection=$arm \
                post_search.z_threshold=0.5 \
                post_search.k=1.0 \
                agent.code.model=deepseek-r1 \
                agent.feedback.model=gpt-4o-2024-08-06 \
                || true  # Continue on failure

            echo "✓ Completed: $arm / $ds / seed$seed"
        done
    done
done

echo "=== A/B test complete ==="
```

---

## Analysis Script

**File**: `experiments/analyze_post_search.py`

```python
"""Analyze post-search selection A/B test results"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats

def load_results(exp_root: str) -> pd.DataFrame:
    """Load all final_selection.json files"""
    results = []

    for arm_dir in Path(exp_root).iterdir():
        if not arm_dir.is_dir():
            continue

        arm = arm_dir.name

        for run_dir in arm_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Parse dataset and seed from directory name
            parts = run_dir.name.split('_seed')
            dataset = parts[0]
            seed = int(parts[1])

            # Load selection summary
            summary_path = run_dir / "logs" / f"{arm}-{dataset}-seed{seed}" / "final_selection.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                data = json.load(f)

            data['arm'] = arm
            data['dataset'] = dataset
            data['seed'] = seed
            results.append(data)

    return pd.DataFrame(results)


def paired_comparison(df: pd.DataFrame, control_arm: str, treatment_arm: str, metric: str):
    """Compute paired t-test for control vs treatment"""
    control = df[df['arm'] == control_arm].sort_values(['dataset', 'seed'])
    treatment = df[df['arm'] == treatment_arm].sort_values(['dataset', 'seed'])

    # Ensure same dataset+seed pairing
    assert (control['dataset'].values == treatment['dataset'].values).all()
    assert (control['seed'].values == treatment['seed'].values).all()

    control_vals = control[metric].values
    treatment_vals = treatment[metric].values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(treatment_vals, control_vals)

    # Effect size (Cohen's d for paired samples)
    diff = treatment_vals - control_vals
    effect_size = np.mean(diff) / np.std(diff)

    # Bootstrap 95% CI
    n_bootstrap = 10000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_means.append(np.mean(diff[sample_idx]))

    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        'mean_diff': np.mean(diff),
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'ci_low': ci_low,
        'ci_high': ci_high,
    }


if __name__ == "__main__":
    exp_root = "/home/ka3094/ML-Master_submit/experiments/post_search_ab"
    df = load_results(exp_root)

    print(f"Loaded {len(df)} results")
    print(df.groupby('arm').size())

    # Primary analysis: generalization gap
    for treatment in ['maximin', 'stat_maximin', 'mean_minus_k_std']:
        print(f"\n=== Control vs {treatment} ===")

        if 'gap_train_valid' in df.columns:
            gap_result = paired_comparison(df, 'best_valid', treatment, 'gap_train_valid')
            print(f"Generalization gap (train-valid):")
            print(f"  Mean diff: {gap_result['mean_diff']:.4f}")
            print(f"  95% CI: [{gap_result['ci_low']:.4f}, {gap_result['ci_high']:.4f}]")
            print(f"  p-value: {gap_result['p_value']:.4f}")
            print(f"  Effect size: {gap_result['effect_size']:.4f}")

        # CV variance
        if 'selected_cv_std' in df.columns:
            cv_result = paired_comparison(df, 'best_valid', treatment, 'selected_cv_std')
            print(f"CV std (lower is better):")
            print(f"  Mean diff: {cv_result['mean_diff']:.4f}")
            print(f"  p-value: {cv_result['p_value']:.4f}")
```

---

## Validation Checklist

Before running full A/B test:

- [ ] Control arm produces identical results to current ML-Master
- [ ] CV parsing works on sample outputs
- [ ] All 3 treatment selectors run without errors
- [ ] final_selection.json is correctly formatted
- [ ] Fallback logic triggers when CV data missing
- [ ] Run 1 dataset with 1 seed end-to-end for all 4 arms
- [ ] Verify MCTS search is identical across arms (same journal up to final selection)

---

## Expected Outcomes

**If robust selection works**:
- **Generalization gap**: Smaller gap (train-valid, train-test) in treatment arms
- **CV variance**: Lower `selected_cv_std` in treatment arms
- **Test performance**: Better or comparable despite potentially lower validation metrics
- **Stability**: Treatment arms select more stable solutions

**Trade-offs**:
- **Validation metric**: Treatment arms may have slightly lower validation metrics
- **Compute**: No additional compute (pure post-processing)
- **Complexity**: Minimal code complexity added

---

## Timeline

1. **Day 1**: Implement Steps 1-4 (node enhancement, parsing, selectors)
2. **Day 2**: Implement Steps 5-6 (config, integration), write tests
3. **Day 3**: Pilot run on 2 datasets, debug, prepare for full A/B

**Total**: 2-3 days to implementation + testing, then launch full A/B experiment.

---

**Next**: After validating Experiment 1, proceed to Experiment 2 (Plan Constraints) and Experiment 3 (Bug Consultant).

---

## Extension: Per-Step Grading for Generalization Gap Analysis

**Motivation**: While post-search selection reduces overfitting in the FINAL solution, we want to understand:
1. **How does the generalization gap evolve during MCTS search?**
2. **Do different selection methods choose solutions from different stages of search?**
3. **Can we track overfitting onset in real-time?**

### Design

**Core Idea**: At each MCTS step, grade ALL 4 selection methods' current choices using MLE-bench ground truth.

```
Step 1: Generate solution_1
  → Grade: best_valid(step1), maximin(step1), elite_maximin(step1), mean_minus_k_std(step1)
Step 2: Generate solution_2
  → Grade: best_valid(step2), maximin(step2), elite_maximin(step2), mean_minus_k_std(step2)
...
Step 50: Generate solution_50
  → Grade: best_valid(step50), maximin(step50), elite_maximin(step50), mean_minus_k_std(step50)
```

**Result**: For each selection method, we have a time series of (validation_score, test_score) pairs.

### Implementation

#### Step 1: Add Per-Step Selection Tracking

**File**: `utils/post_search.py`

Add function to select current best for each method:

```python
def select_all_methods_at_step(
    journal: Journal,
    step: int,
    methods: list[str] = None
) -> dict[str, MCTSNode | None]:
    """
    For each selection method, return the node it would select
    given only nodes up to this step.

    Args:
        journal: Full journal
        step: Current step number
        methods: List of method names (default: all 4 methods)

    Returns:
        Dict mapping method_name -> selected_node (or None if no valid selection)
    """
    if methods is None:
        methods = ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"]

    # Filter journal to only include nodes up to current step
    nodes_up_to_step = [n for n in journal.all_nodes if n.step <= step]

    # Create a temporary journal view
    temp_journal = Journal()
    temp_journal._nodes = nodes_up_to_step

    selections = {}
    for method in methods:
        try:
            node = _select_by_method(temp_journal, method)
            selections[method] = node
        except Exception as e:
            logger.warning(f"Failed to select for method {method} at step {step}: {e}")
            selections[method] = None

    return selections


def _select_by_method(journal: Journal, method: str, **kwargs) -> MCTSNode:
    """Helper to select by method name"""
    if method == "best_valid":
        return _select_best_valid(journal)
    elif method == "maximin":
        return _select_maximin(journal)
    elif method == "elite_maximin":
        return _select_stat_maximin(journal, z_threshold=kwargs.get('z_threshold', 0.5))
    elif method == "mean_minus_k_std":
        return _select_mean_minus_k_std(journal, k=kwargs.get('k', 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")
```

#### Step 2: Add MLE-bench Grading Integration

**File**: `utils/mlebench_grading.py` (new file)

```python
"""
Integration with MLE-bench for per-step grading.
Allows grading submissions during search to track generalization gap.
"""

import logging
from pathlib import Path
from typing import Optional
import json

logger = logging.getLogger("ml-master")


def grade_submission_with_mlebench(
    submission_path: Path,
    competition_id: str,
    data_dir: Path,
) -> dict:
    """
    Grade a submission using MLE-bench.

    Args:
        submission_path: Path to submission CSV
        competition_id: MLE-bench competition ID
        data_dir: MLE-bench data directory

    Returns:
        dict with keys: score, percentile, error (if failed)
    """
    try:
        # Import MLE-bench grading
        from mlebench.grade import grade_csv
        from mlebench.registry import Registry

        registry = Registry().set_data_dir(data_dir)
        competition = registry.get_competition(competition_id)

        report = grade_csv(submission_path, competition)

        return {
            "score": report.score,
            "percentile": getattr(report, 'percentile', None),
            "error": None,
        }
    except Exception as e:
        logger.warning(f"MLE-bench grading failed: {e}")
        return {
            "score": None,
            "percentile": None,
            "error": str(e),
        }


def setup_per_step_grading(cfg, competition_id: str = None):
    """
    Setup per-step grading if enabled in config.

    Returns:
        GradingCallback or None
    """
    if not getattr(cfg, 'per_step_grading', None) or not cfg.per_step_grading.enabled:
        return None

    if competition_id is None:
        logger.warning("Per-step grading enabled but no competition_id provided")
        return None

    data_dir = Path(cfg.per_step_grading.mlebench_data_dir)
    if not data_dir.exists():
        logger.warning(f"MLE-bench data dir not found: {data_dir}")
        return None

    logger.info(f"Per-step grading ENABLED for competition: {competition_id}")

    return GradingCallback(
        competition_id=competition_id,
        data_dir=data_dir,
        output_dir=Path(cfg.log_dir) / cfg.exp_name / "per_step_grading",
        methods=cfg.per_step_grading.methods,
    )


class GradingCallback:
    """
    Callback to grade all selection methods at each step.
    """

    def __init__(
        self,
        competition_id: str,
        data_dir: Path,
        output_dir: Path,
        methods: list[str] = None,
    ):
        self.competition_id = competition_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.methods = methods or ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track grading history
        self.grading_history = {method: [] for method in self.methods}

    def on_step_complete(self, journal, step: int, workspace_dir: Path):
        """
        Called after each MCTS step completes.
        Grades all selection methods' current choices.
        """
        from utils.post_search import select_all_methods_at_step

        logger.info(f"[Per-step grading] Step {step}: selecting and grading all methods")

        # Select current best for each method
        selections = select_all_methods_at_step(journal, step, methods=self.methods)

        # Grade each selection
        for method, node in selections.items():
            if node is None:
                logger.warning(f"[Per-step grading] Step {step}, method {method}: no selection")
                self.grading_history[method].append({
                    "step": step,
                    "node_id": None,
                    "validation_score": None,
                    "test_score": None,
                    "error": "No selection",
                })
                continue

            # Get submission path for this node
            submission_path = self._get_submission_path(node, workspace_dir)
            if submission_path is None or not submission_path.exists():
                logger.warning(f"[Per-step grading] Step {step}, method {method}: submission not found")
                self.grading_history[method].append({
                    "step": step,
                    "node_id": node.id,
                    "validation_score": node.metric.value if node.metric else None,
                    "test_score": None,
                    "error": "Submission file not found",
                })
                continue

            # Grade with MLE-bench
            grade_result = grade_submission_with_mlebench(
                submission_path,
                self.competition_id,
                self.data_dir,
            )

            # Record result
            self.grading_history[method].append({
                "step": step,
                "node_id": node.id,
                "validation_score": node.metric.value if node.metric else None,
                "cv_mean": node.cv_mean,
                "cv_std": node.cv_std,
                "test_score": grade_result["score"],
                "test_percentile": grade_result.get("percentile"),
                "error": grade_result.get("error"),
            })

            logger.info(
                f"[Per-step grading] Step {step}, method {method}: "
                f"node {node.id}, validation={node.metric.value if node.metric else None:.4f}, "
                f"test={grade_result['score']:.4f if grade_result['score'] is not None else 'N/A'}"
            )

    def save_results(self):
        """Save grading history to JSON"""
        output_file = self.output_dir / "grading_history.json"
        with open(output_file, 'w') as f:
            json.dump(self.grading_history, f, indent=2)

        logger.info(f"[Per-step grading] Saved grading history to {output_file}")

        # Also save CSV for easy analysis
        self._save_as_csv()

    def _save_as_csv(self):
        """Save grading history as CSV for easy plotting"""
        import pandas as pd

        rows = []
        for method, history in self.grading_history.items():
            for record in history:
                rows.append({
                    "method": method,
                    **record
                })

        df = pd.DataFrame(rows)
        csv_file = self.output_dir / "grading_history.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"[Per-step grading] Saved grading history CSV to {csv_file}")

    def _get_submission_path(self, node, workspace_dir: Path) -> Path | None:
        """Get submission file path for a node"""
        if hasattr(node, 'submission_csv_path') and node.submission_csv_path:
            return Path(node.submission_csv_path)

        # Fallback: look for submission_{node_id}.csv
        submission_path = workspace_dir / "submission" / f"submission_{node.id}.csv"
        if submission_path.exists():
            return submission_path

        return None
```

#### Step 3: Integrate into Main Loop

**File**: `main_mcts.py`

```python
def run():
    # ... existing setup ...

    # NEW: Setup per-step grading
    from utils.mlebench_grading import setup_per_step_grading

    competition_id = os.getenv("COMPETITION_ID") or cfg.get("competition_id")
    grading_callback = setup_per_step_grading(cfg, competition_id)

    # Main MCTS loop
    for step in range(cfg.agent.steps):
        agent.step()

        # NEW: Per-step grading callback
        if grading_callback:
            grading_callback.on_step_complete(
                journal=journal,
                step=step + 1,
                workspace_dir=cfg.workspace_dir,
            )

    # After search completes
    if grading_callback:
        grading_callback.save_results()

    # ... rest of existing code ...
```

#### Step 4: Add Configuration

**File**: `utils/config_mcts.yaml`

```yaml
# Per-step grading configuration (for generalization gap experiments)
per_step_grading:
  enabled: false  # Set to true for experiments
  mlebench_data_dir: /home/ka3094/mle-bench/data/competitions
  methods:
    - best_valid
    - mean_minus_k_std
    - maximin
    - elite_maximin

  # Optional: Grade every N steps instead of every step (for efficiency)
  grade_every_n_steps: 1
```

**File**: `utils/config_mcts.py`

```python
@dataclass
class PerStepGradingConfig:
    enabled: bool = False
    mlebench_data_dir: str = "/home/ka3094/mle-bench/data/competitions"
    methods: list[str] = field(default_factory=lambda: ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"])
    grade_every_n_steps: int = 1


@dataclass
class Config:
    # ... existing fields ...
    per_step_grading: PerStepGradingConfig = field(default_factory=PerStepGradingConfig)
```

### Analysis: Generalization Gap Evolution

**File**: `experiments/analyze_per_step_grading.py`

```python
"""
Analyze per-step grading results to understand generalization gap evolution.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_grading_history(run_dir: Path) -> pd.DataFrame:
    """Load grading history from a run"""
    history_file = run_dir / "per_step_grading" / "grading_history.csv"
    if history_file.exists():
        return pd.read_csv(history_file)

    # Fallback: load from JSON
    json_file = run_dir / "per_step_grading" / "grading_history.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)

        rows = []
        for method, history in data.items():
            for record in history:
                rows.append({"method": method, **record})
        return pd.DataFrame(rows)

    raise FileNotFoundError(f"No grading history found in {run_dir}")


def plot_generalization_gap(df: pd.DataFrame, output_path: Path = None):
    """
    Plot generalization gap evolution for all methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Generalization Gap Evolution: All Selection Methods', fontsize=16)

    methods = df['method'].unique()

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        method_df = df[df['method'] == method].dropna(subset=['validation_score', 'test_score'])

        if len(method_df) == 0:
            ax.text(0.5, 0.5, f'No data for {method}', ha='center', va='center')
            ax.set_title(method)
            continue

        steps = method_df['step']
        val_scores = method_df['validation_score']
        test_scores = method_df['test_score']
        gap = val_scores - test_scores

        # Plot validation and test scores
        ax.plot(steps, val_scores, label='Validation', marker='o', alpha=0.7)
        ax.plot(steps, test_scores, label='Test (Ground Truth)', marker='s', alpha=0.7)

        # Plot gap
        ax2 = ax.twinx()
        ax2.plot(steps, gap, label='Gap (Val - Test)', color='red', linestyle='--', marker='x', alpha=0.7)
        ax2.set_ylabel('Generalization Gap', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_xlabel('MCTS Step')
        ax.set_ylabel('Score')
        ax.set_title(f'{method}')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def compute_gap_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute generalization gap statistics for each method.
    """
    results = []

    for method in df['method'].unique():
        method_df = df[df['method'] == method].dropna(subset=['validation_score', 'test_score'])

        if len(method_df) == 0:
            continue

        gap = method_df['validation_score'] - method_df['test_score']

        results.append({
            'method': method,
            'mean_gap': gap.mean(),
            'std_gap': gap.std(),
            'max_gap': gap.max(),
            'min_gap': gap.min(),
            'final_gap': gap.iloc[-1] if len(gap) > 0 else None,
            'final_validation': method_df['validation_score'].iloc[-1] if len(method_df) > 0 else None,
            'final_test': method_df['test_score'].iloc[-1] if len(method_df) > 0 else None,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze per-step grading results")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--output", type=Path, help="Output path for plot")
    args = parser.parse_args()

    df = load_grading_history(args.run_dir)

    print("=" * 80)
    print("Generalization Gap Statistics")
    print("=" * 80)
    stats = compute_gap_statistics(df)
    print(stats.to_string(index=False))
    print()

    output_path = args.output or args.run_dir / "per_step_grading" / "gap_evolution.png"
    plot_generalization_gap(df, output_path)
```

### Running Per-Step Grading Experiments

**Example: MLE-bench Native Run with Per-Step Grading**

```bash
# Export competition ID for grading
export COMPETITION_ID=playground_series_s5e8

# Run with per-step grading enabled
python main_mcts.py \
    data_dir=/home/ka3094/dataset_submit/playground_series_s5e8 \
    desc_file=/home/ka3094/dataset_submit/playground_series_s5e8/task_description.txt \
    agent.steps=10 \
    agent.k_fold_validation=5 \
    exp_name=gap-experiment \
    per_step_grading.enabled=true \
    per_step_grading.mlebench_data_dir=/home/ka3094/mle-bench/data/competitions \
    competition_id=playground_series_s5e8

# Analyze results
python experiments/analyze_per_step_grading.py \
    ./logs/gap-experiment \
    --output ./logs/gap-experiment/gap_plot.png
```

### Expected Insights

1. **Gap Evolution**: Does generalization gap increase monotonically or plateau?
2. **Method Comparison**: Do robust methods (maximin, elite_maximin) select solutions with smaller gaps?
3. **Overfitting Onset**: At what step does validation score peak but test score decline?
4. **Solution Stability**: Do robust methods select from different regions of the search tree?

### Output Structure

```
logs/
└── gap-experiment/
    ├── per_step_grading/
    │   ├── grading_history.json       # Full grading history
    │   ├── grading_history.csv        # CSV for plotting
    │   └── gap_evolution.png          # Generated plot
    ├── journal.json                   # MCTS journal
    ├── final_selection.json           # Post-search selection
    └── ml-master.log                  # Main log
```

### Integration with Existing Experiment Plan

This extends the base experiment plan by:
1. **No changes to MCTS search** - still pure post-processing
2. **Additional grading calls** - uses MLE-bench ground truth
3. **Temporal analysis** - tracks generalization gap evolution
4. **Validates hypothesis** - shows IF robust selection reduces generalization gap in practice

**Use Cases**:
- **Full experiments**: Enable per-step grading on 1-2 datasets to visualize gap evolution
- **Quick tests**: Disable for faster iteration during development
- **Production runs**: Optional - only needed for analysis, not required for selection to work
