# ML-Master Experiments: Testing 3 AIDE Improvements

**Goal**: Test if 3 improvements from AIDE also work on ML-Master

## The 3 Improvements

### 1. Bug Consultant (RAG + RL + Summarization Debugging)

**What it does**:
- Learns from past bugs to avoid repeating mistakes
- Retrieves similar bugs when debugging ("I've seen this before...")
- Tracks failed strategies to prevent repetition
- Accumulates debugging knowledge in a world model

**Impact on AIDE**: Improved bug resolution rate, fewer wasted debugging iterations

**Test on ML-Master**: Does intelligent debugging also help ML-Master's MCTS search?

---

### 2. Plan Constraints (Plan-First Debugging)

**What it does**:
- Forces diagnostic planning before code generation
- Two-phase: (1) Diagnose root cause → (2) Implement fix
- Prevents "quick fix syndrome"

**Impact on AIDE**: Better debugging success rate, deeper root cause analysis

**Test on ML-Master**: Does structured debugging improve ML-Master's fix quality?

---

### 3. Post-Search Selection (Robust Final Selection)

**What it does**:
- Selects final solution using robust strategies instead of greedy "best validation"
- Options: maximin (best worst-case), stat-maximin (elite + maximin), mean-minus-k-std
- Uses CV fold statistics to assess solution stability

**Impact on AIDE**: Reduced overfitting, better generalization, more stable solutions

**Test on ML-Master**: Does robust selection also reduce overfitting in MCTS-found solutions?

---

## Implementation Priority

### Phase 1: Post-Search Selection (EASIEST)
- Pure post-processing after search completes
- No changes to core search logic
- **Timeline**: 2-3 days

### Phase 2: Plan Constraints (MEDIUM)
- Modifies debugging method only
- Localized changes
- **Timeline**: 3-4 days

### Phase 3: Bug Consultant (HARDEST)
- Most complex: RAG, LLM summarization, retrieval
- Requires careful integration
- **Timeline**: 5-7 days

**Total**: ~2-4 weeks for all 3

---

## Experiment Design

For each improvement:
- **Control**: Current ML-Master (no improvement)
- **Treatment**: With improvement enabled
- **Datasets**: 10 datasets × 3 seeds = 30 runs per arm
- **Metrics**: Specific to each improvement (see individual docs)
- **Analysis**: Paired comparisons, bootstrap CIs

---

## Files

- `EXPERIMENT_1_POST_SEARCH_SELECTION.md` - Implementation plan for improvement #3
- `EXPERIMENT_2_PLAN_CONSTRAINTS.md` - Implementation plan for improvement #2
- `EXPERIMENT_3_BUG_CONSULTANT.md` - Implementation plan for improvement #1
- `IMPLEMENTATION_STATUS.md` - Implementation tracking checklist

---

## Quick Start

### Running Post-Search Selection Experiments

**Option 1: Using the bash script**

```bash
cd /home/ka3094/ML-Master_submit
bash experiments/run_post_search_ab.sh <data_dir> <desc_file> [seeds]

# Example:
bash experiments/run_post_search_ab.sh \
    /path/to/kaggle/task \
    /path/to/task.md \
    "0,1,2"
```

**Option 2: Using the Python script (more flexible)**

```bash
python3 experiments/run_post_search_experiment.py \
    --data-dir /path/to/kaggle/task \
    --desc-file /path/to/task.md \
    --strategies "best_valid,maximin,elite_maximin,mean_minus_k_std" \
    --seeds "0,1,2" \
    --steps 50 \
    --k-fold 5

# This will:
# 1. Run ML-Master with each strategy × seed combination
# 2. Generate a summary.csv with results
# 3. Print comparison statistics
```

**Available strategies:**
- `best_valid` - Baseline: best validation metric (default)
- `maximin` - Best worst-case CV fold
- `elite_maximin` - Statistical filtering + maximin
- `mean_minus_k_std` - Lower confidence bound
- `maximin_no_filter` - Maximin without top-k filtering

### Results Structure

```
experiments/results/post_search_ab_<timestamp>/
├── best_valid_seed0/
│   ├── final_selection.json    # Selected node metadata
│   ├── best_solution.py        # Selected solution code
│   └── ml-master.log          # Execution logs
├── maximin_seed0/
│   └── ...
└── summary.csv                 # Aggregated results
```

### Analyzing Results

The `summary.csv` contains:
- `strategy`: Selection strategy used
- `seed`: Random seed
- `selected_metric`: Final validation metric
- `selected_cv_mean`: Mean CV score
- `selected_cv_std`: CV standard deviation

Compare:
- Mean performance across seeds
- Variance (robustness)
- Worst-case performance (safety)

---

### Running Plan Constraint Experiments

```bash
bash experiments/run_plan_constraints_ab.sh <data_dir> <desc_file> [seeds]

# Example:
bash experiments/run_plan_constraints_ab.sh \
    /path/to/kaggle/task \
    /path/to/task.md \
    "0,1,2"
```

### Running Bug Consultant Experiments

```bash
bash experiments/run_bug_consultant_ab.sh <data_dir> <desc_file> [seeds]

# Example:
bash experiments/run_bug_consultant_ab.sh \
    /path/to/kaggle/task \
    /path/to/task.md \
    "0,1,2"
```

## Success = All 3 Improvements Work on Both AIDE and ML-Master

If successful, we prove these are **general AutoML improvements**, not specific to one framework.
