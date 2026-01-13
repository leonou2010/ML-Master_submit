# Implementation Status - 3 AIDE Improvements for ML-Master

**Goal**: Implement and test 3 improvements from AIDE on ML-Master

---

## Phase 1: Post-Search Selection ⬅️ START HERE

**What**: Use robust selection strategies instead of greedy "best validation"

**Implementation Steps**:

```
[✓] 1. Add CV metrics to MCTSNode
    File: search/mcts_node.py
    Add fields: cv_mean, cv_std, cv_folds, train_metric, valid_metric, test_metric
    Status: COMPLETE

[✓] 2. Parse CV statistics from execution output
    File: utils/metrics_io.py (NEW - copied from AIDE)
    Functions: parse_aide_metrics(), normalize_metrics()
    Status: COMPLETE

[✓] 3. Populate CV fields after execution
    File: agent/mcts_agent.py
    - Updated review_func_spec to include cv_folds
    - Added CV metrics parsing in parse_exec_result()
    - Two-layer parsing: JSON first, then LLM fallback
    - CV fold validation (detect placeholder bugs)
    - Calculate cv_mean and cv_std from cv_folds
    - Mark nodes as buggy if CV validation fails
    Status: COMPLETE

[✓] 4. Create post-search selector module
    File: utils/post_search.py (NEW - copied from AIDE)
    Functions: select_final_node(), select_final_node_with_info()
    Strategies: best_valid, maximin, elite_maximin, mean_minus_k_std, maximin_no_filter
    Status: COMPLETE

[✓] 5. Update configuration
    Files: utils/config_mcts.yaml, utils/config_mcts.py
    Add: PostSearchConfig dataclass and post_search section
    Status: COMPLETE

[✓] 6. Integrate into main loop
    File: main_mcts.py
    After MCTS completes: call select_final_node_with_info()
    Export: final_selection.json with all metadata
    Status: COMPLETE

[✓] 7. Create runner scripts
    Files: experiments/run_post_search_ab.sh (bash)
           experiments/run_post_search_experiment.py (python)
    Status: COMPLETE

[ ] 8. Pilot test (2 datasets × 4 arms = 8 runs)
    Status: PENDING

[ ] 9. Full A/B (10 datasets × 3 seeds × 4 arms = 120 runs)
    Status: PENDING

[ ] 10. Analyze results
    Status: PENDING
```

**Time**: 2-3 days

---

## Phase 2: Plan Constraints

**What**: Prompt-only plan/sketch length constraints (no truncation)

**Implementation Steps**:

```
[✓] 1. Add plan_constraints config
    Files: utils/config_mcts.yaml, utils/config_mcts.py
    Fields: enabled, max_sentences
    Status: COMPLETE

[✓] 2. Apply constraints in prompts
    File: agent/mcts_agent.py
    - Updates response format when enabled
    - Adds a length guideline to draft/improve/debug sketch instructions
    Status: COMPLETE

[✓] 3. Create runner script
    File: experiments/run_plan_constraints_ab.sh
    Status: COMPLETE

[ ] 4. Pilot test (2 datasets × 2 arms = 4 runs)
    Status: PENDING

[ ] 5. Full A/B (10 datasets × 3 seeds × 2 arms = 60 runs)
    Status: PENDING

[ ] 6. Analyze results
    Status: PENDING
```

**Time**: 1-2 days

---

## Phase 3: Bug Consultant

**What**: RAG + RL + Summarization for intelligent debugging

**Implementation Steps**:

```
[✓] 1. Create bug consultant module
    File: utils/bug_consultant.py (NEW)
    Status: COMPLETE

[✓] 2. Implement LLM summarization functions
    File: utils/bug_consultant.py
    Status: COMPLETE

[✓] 3. Implement BugConsultant class methods
    - start_bug_record(), record_trial(), complete_bug_record()
    - retrieve_relevant_context() + formatting helpers
    - learn_from_bug() + ingest_journal()
    Status: COMPLETE

[✓] 4. Add bug_consultant config
    Files: utils/config_mcts.yaml, utils/config_mcts.py
    Add: agent.search.use_bug_consultant and parameters
    Status: COMPLETE

[✓] 5. Initialize consultant in agent
    File: agent/mcts_agent.py
    In __init__: create BugConsultant if enabled
    Status: COMPLETE

[✓] 6. Integrate into prompting + learning
    File: agent/mcts_agent.py
    - Injects consultant context into _draft/_debug prompts
    - Calls learn_from_bug() after each evaluated node
    Status: COMPLETE

[✓] 7. Create runner script
    File: experiments/run_bug_consultant_ab.sh
    Status: COMPLETE

[ ] 8. Pilot test (2 datasets × 2 arms = 4 runs)
    Status: PENDING

[ ] 9. Full A/B (10 datasets × 3 seeds × 2 arms = 60 runs)
    Status: PENDING

[ ] 10. Analyze results
    Status: PENDING
```

**Time**: 2-4 days

---

## Quick Start - Phase 1

```bash
cd /home/ka3094/ML-Master_submit

# Step 1: Edit search/mcts_node.py
# Add these fields to MCTSNode class:
#   cv_mean: float | None = field(default=None, kw_only=True)
#   cv_std: float | None = field(default=None, kw_only=True)
#   cv_folds: list[float] | None = field(default=None, kw_only=True)
#   train_metric: float | None = field(default=None, kw_only=True)
#   valid_metric: float | None = field(default=None, kw_only=True)

# Step 2: Create utils/metric.py enhancement (see EXPERIMENT_1 doc for code)

# Step 3: Create utils/post_search.py (see EXPERIMENT_1 doc for code)

# Continue with remaining steps...
```

---

## Success Criteria

### Phase 1 Success
✅ CV metrics parsed from ≥80% of runs
✅ All 4 selectors run without errors
✅ Generalization gap reduced in treatment arms

### Phase 2 Success
✅ Constraints appear in prompts when enabled
✅ Treatment arm changes sketch length distribution

### Phase 3 Success
✅ Bug records saved correctly
✅ RAG retrieval finds relevant bugs
✅ Bug resolution rate > control

---

## Current Status

📋 **Planning**: COMPLETE
✅ **Phase 1 Implementation**: COMPLETE (Steps 1-7)
✅ **Phase 2 Implementation**: COMPLETE (Steps 1-3)
✅ **Phase 3 Implementation**: COMPLETE (Steps 1-7)
🧪 **Testing**: READY TO START (pilot tests)
👉 **Next Action**: Run pilot tests for all 3 improvements

---

## Timeline

- **Phase 1**: 2-3 days
- **Phase 2**: 3-4 days
- **Phase 3**: 5-7 days
- **Total**: ~2-4 weeks

---

**Last Updated**: January 11, 2026
