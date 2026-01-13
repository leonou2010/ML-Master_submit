# Phase 1 Implementation Summary: Post-Search Selection

**Date**: January 11, 2026
**Status**: ✅ IMPLEMENTATION COMPLETE - READY FOR TESTING

---

## Overview

Successfully implemented post-search selection for ML-Master by copying AIDE's exact implementation. This enables robust final solution selection using cross-validation statistics instead of greedy "best validation" metric.

## What Was Implemented

### 1. CV Metrics Support in MCTSNode

**File**: `search/mcts_node.py`

Added 6 new fields to track cross-validation statistics:

```python
# Post-search selection: CV metrics (copied from AIDE)
cv_mean: Optional[float] = field(default=None, kw_only=True)
cv_std: Optional[float] = field(default=None, kw_only=True)
cv_folds: Optional[list[float]] = field(default=None, kw_only=True)
valid_metric: Optional[float] = field(default=None, kw_only=True)
train_metric: Optional[float] = field(default=None, kw_only=True)
test_metric: Optional[float] = field(default=None, kw_only=True)
```

### 2. Metrics Parsing Module

**File**: `utils/metrics_io.py` (NEW - 97 lines, exact copy from AIDE)

Functions:
- `parse_aide_metrics(term_out)` - Extracts metrics from terminal output
- `normalize_metrics(metrics)` - Coerces and validates metric types
- `_coerce_float()`, `_coerce_float_list()` - Type conversion helpers

Expected format in generated code:
```python
print(f'AIDE_METRICS_JSON={json.dumps({
    "valid": 0.85,
    "lower_is_better": False,
    "cv_mean": 0.87,
    "cv_std": 0.03,
    "cv_folds": [0.85, 0.88, 0.89, 0.86, 0.87]
})}')
```

### 3. Post-Search Selection Module

**File**: `utils/post_search.py` (NEW - 307 lines, exact copy from AIDE)

**Available Strategies**:

1. **`best_valid`** (Baseline)
   - Selects node with highest validation metric
   - Current ML-Master behavior
   - Fast, simple, but prone to overfitting

2. **`maximin`**
   - Selects node with best worst-case CV fold
   - `max(nodes, key=lambda n: min(n.cv_folds))`
   - Maximizes robustness

3. **`elite_maximin`**
   - Statistical filtering + maximin
   - Filters to elite set: `max(top_k, ratio%, best±k_std)`
   - Then applies maximin on elite set
   - Balances performance and robustness

4. **`mean_minus_k_std`**
   - Lower confidence bound selection
   - `max(nodes, key=lambda n: n.cv_mean - k*n.cv_std)`
   - Conservative, risk-averse

5. **`maximin_no_filter`**
   - Maximin on all nodes (no top-k filtering)
   - Most conservative

**Key Functions**:
- `select_final_node()` - Main entry point
- `select_final_node_with_info()` - Returns node + metadata dict
- `_topk()`, `_best_by()`, `_worst_fold()` - Helper functions

### 4. Configuration Updates

**File**: `utils/config_mcts.py`

Added `PostSearchConfig` dataclass:

```python
@dataclass
class PostSearchConfig:
    selection: str = "best_valid"
    top_k: int = 20
    k_std: float = 2.0
    z_threshold: float = 2.0
    guard_std: float = 2.0
    elite_top_k: int = 3
    elite_ratio: float = 0.05
    elite_k_std: float = 2.0
```

**File**: `utils/config_mcts.yaml`

Added configuration section:

```yaml
post_search:
  selection: best_valid
  top_k: 20
  k_std: 2.0
  elite_top_k: 3
  elite_ratio: 0.05
  elite_k_std: 2.0
```

### 5. Agent Modifications

**File**: `agent/mcts_agent.py`

**Changes**:

1. **Imports**: Added `metrics_io`, `numpy`

2. **Review Function Spec**: Added `cv_folds` field
   ```python
   "cv_folds": {
       "type": "array",
       "items": {"type": "number"},
       "description": "If CV was used, report all fold scores..."
   }
   ```

3. **Implementation Guidelines**: Added instructions
   - Print metrics in `AIDE_METRICS_JSON=` format
   - Use mandatory CV when `k_fold_validation > 1`
   - Report ALL fold scores (not just mean/std)

4. **CV Metrics Parsing** (in `parse_exec_result()`):
   - Two-layer parsing:
     - Layer 1: JSON parsing (fast, structured)
     - Layer 2: LLM extraction (fallback)
   - CV fold validation:
     - Detect all zeros/ones (placeholder bugs)
     - Check for variance across folds
   - Calculate `cv_mean` and `cv_std` from `cv_folds`
   - Mark node as buggy if CV validation fails

### 6. Main Loop Integration

**File**: `main_mcts.py`

Added post-search selection after MCTS completes:

```python
from utils.post_search import select_final_node_with_info
import json

final_node, selection_info = select_final_node_with_info(
    journal,
    selection=cfg.post_search.selection,
    top_k=cfg.post_search.top_k,
    k_std=cfg.post_search.k_std,
    # ... all parameters
    only_good=True,
)

# Save final selection metadata
selection_file = Path(cfg.log_dir) / cfg.exp_name / "final_selection.json"
selection_data = {
    **selection_info,
    "selected_node_id": final_node.id,
    "selected_metric": final_node.metric.value,
    "selected_cv_mean": final_node.cv_mean,
    "selected_cv_std": final_node.cv_std,
    "selected_cv_folds": final_node.cv_folds,
}
with open(selection_file, 'w') as f:
    json.dump(selection_data, f, indent=2)
```

### 7. Experiment Runner Scripts

**File**: `experiments/run_post_search_ab.sh` (Bash script - 115 lines)

Features:
- Runs all strategies × seeds combinations
- Generates `summary.csv` with aggregated results
- Uses `jq` to parse `final_selection.json` files

**File**: `experiments/run_post_search_experiment.py` (Python script - 172 lines)

Features:
- More flexible configuration
- pandas-based analysis
- Computes mean ± std across seeds
- Prints comparison statistics

---

## Files Created/Modified

### New Files (3)
1. `utils/metrics_io.py` - 97 lines (copied from AIDE)
2. `utils/post_search.py` - 307 lines (copied from AIDE)
3. `experiments/run_post_search_ab.sh` - 115 lines
4. `experiments/run_post_search_experiment.py` - 172 lines

### Modified Files (5)
1. `search/mcts_node.py` - Added 6 CV metric fields
2. `utils/config_mcts.py` - Added PostSearchConfig dataclass
3. `utils/config_mcts.yaml` - Added post_search section
4. `agent/mcts_agent.py` - Added CV parsing logic (~120 lines)
5. `main_mcts.py` - Added post-search selection call (~40 lines)

**Total**: 4 new files, 5 modified files, ~850 lines of new code

---

## How It Works

### Execution Flow

1. **MCTS Search** (existing)
   - Generates N solutions (nodes)
   - Each node has: code, plan, metric, etc.

2. **CV Metrics Parsing** (NEW)
   - For each executed node:
     - Parse terminal output for `AIDE_METRICS_JSON=`
     - Extract `cv_folds` from JSON or LLM
     - Validate CV folds (no placeholders)
     - Calculate `cv_mean` and `cv_std`
     - Store in node fields

3. **Post-Search Selection** (NEW)
   - After MCTS completes:
     - Call `select_final_node_with_info(journal, ...)`
     - Apply selected strategy (e.g., `elite_maximin`)
     - Return final node + metadata
     - Save `final_selection.json`

4. **Output**
   - `best_solution.py` - Selected solution code
   - `final_selection.json` - Selection metadata
   - `ml-master.log` - Full execution logs

### Example: elite_maximin Strategy

```
1. Filter to top-20 nodes by cv_mean
2. Compute elite set: max(top_3, 5%, mean±2std)
   → Elite set size: 8 nodes
3. Apply maximin on elite set:
   → Select node with best worst-case CV fold
4. Return: Node #42 (cv_mean=0.87, worst_fold=0.83)
```

---

## Testing Instructions

### Quick Test (Single Run)

```bash
cd /home/ka3094/ML-Master_submit

python main_mcts.py \
    data_dir=/path/to/task \
    desc_file=/path/to/task.md \
    agent.steps=10 \
    agent.k_fold_validation=5 \
    post_search.selection=elite_maximin

# Check output:
cat logs/run/<exp_name>/final_selection.json
```

### A/B Testing (Multiple Strategies)

```bash
# Option 1: Bash script
bash experiments/run_post_search_ab.sh \
    /path/to/task \
    /path/to/task.md \
    "0,1,2"

# Option 2: Python script
python experiments/run_post_search_experiment.py \
    --data-dir /path/to/task \
    --desc-file /path/to/task.md \
    --strategies "best_valid,maximin,elite_maximin" \
    --seeds "0,1,2" \
    --steps 50 \
    --k-fold 5
```

### Expected Output

```
experiments/results/post_search_ab_<timestamp>/
├── best_valid_seed0/
│   ├── final_selection.json
│   ├── best_solution.py
│   └── ml-master.log
├── maximin_seed0/
│   └── ...
└── summary.csv
```

`summary.csv` format:
```
strategy,seed,selected_node_id,selected_metric,selected_cv_mean,selected_cv_std
best_valid,0,node_45,0.8523,0.8612,0.0234
maximin,0,node_38,0.8401,0.8456,0.0189
elite_maximin,0,node_42,0.8478,0.8571,0.0198
```

---

## Design Decisions

### 1. Exact Copy from AIDE

**Why**: To ensure fair A/B comparison, we copied AIDE's implementation exactly rather than reimplementing. This eliminates implementation differences as a confounding variable.

**What was copied**:
- `utils/metrics_io.py` - Line-by-line copy
- `utils/post_search.py` - Line-by-line copy
- Only changed: imports to match ML-Master's structure

### 2. Two-Layer Metrics Parsing

**Why**: Robustness against parsing failures

**How**:
1. JSON parsing (fast, structured, preferred)
2. LLM extraction (fallback, handles any format)

### 3. Strict CV Enforcement (when enabled)

**Why**: Prevent accidentally selecting solutions without valid CV data

**How**:
- If `k_fold_validation > 1`, CV folds are **mandatory**
- Nodes without valid CV folds are marked as buggy
- Placeholder detection (all zeros/ones, identical values)

### 4. Backward Compatibility

**Why**: Don't break existing workflows

**How**:
- If `k_fold_validation = 1`, CV parsing is optional
- Default `post_search.selection = "best_valid"` (no change in behavior)
- All new fields default to `None`

---

## Limitations & Future Work

### Current Limitations

1. **No train/test metrics**: Only parsing CV metrics, not train/test
   - Reason: Simpler implementation, focus on CV robustness
   - Future: Add train/test parsing if needed

2. **No last_good strategy**: AIDE has `last_good`, we skipped it
   - Reason: Not used in AIDE's experiments
   - Future: Add if requested

3. **Requires jq for bash script**: Summary generation uses `jq`
   - Workaround: Use Python script instead
   - Future: Add pure bash fallback

### Future Enhancements

1. **Submission path tracking**: Parse submission file paths from output
2. **Multi-objective selection**: Balance multiple metrics
3. **Visualization**: Plot CV distribution for each node
4. **Auto-tuning**: Learn best k_std, elite_ratio from data

---

## Next Steps

### Immediate (Testing Phase)

1. **Pilot Test** (2 datasets × 4 arms = 8 runs)
   - Validate implementation works end-to-end
   - Check CV metrics are parsed correctly
   - Verify all strategies run without errors

2. **Debug Issues** (if any)
   - Fix parsing failures
   - Handle edge cases
   - Improve error messages

3. **Full A/B Test** (10 datasets × 3 seeds × 4 arms = 120 runs)
   - Run on diverse tasks
   - Compare strategies statistically
   - Analyze generalization gap

### Later (Analysis Phase)

4. **Results Analysis**
   - Does robust selection reduce overfitting?
   - Which strategy works best on average?
   - Are results consistent with AIDE?

5. **Paper Writing**
   - Document findings
   - Compare AIDE vs ML-Master
   - Prove generality of improvement

---

## Success Criteria

Phase 1 is considered **successful** if:

✅ CV metrics parsed correctly in ≥80% of runs
✅ All 5 selection strategies run without errors
✅ `final_selection.json` generated for all runs
✅ Robust strategies (maximin, elite_maximin) reduce generalization gap vs. best_valid
✅ Results are consistent with AIDE's findings

---

## Contact & Support

For questions or issues:
- Check `experiments/EXPERIMENT_1_POST_SEARCH_SELECTION.md` for detailed design
- Check `experiments/IMPLEMENTATION_STATUS.md` for current status
- Review AIDE's implementation: `/home/ka3094/aideml_submit/aide/utils/post_search.py`

---

**Implementation by**: Claude Code
**Last Updated**: January 11, 2026
**Status**: ✅ READY FOR TESTING
