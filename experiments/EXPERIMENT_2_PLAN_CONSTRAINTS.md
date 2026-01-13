# Experiment 2: Plan Constraints for ML-Master

**Priority**: Phase 2 (Implement Second)
**Risk Level**: Medium
**Expected Impact**: Medium-High
**Estimated Time**: 3-4 days

## Overview

Test whether forcing diagnostic planning before code generation improves debugging success in ML-Master's MCTS search. This builds on AIDE's plan-first debugging approach but adapts it to MCTS's parallel search architecture.

## Motivation

ML-Master's current `_debug()` method generates a brief plan + code in a single LLM call:
```python
prompt = "Your previous solution had a bug... fix it."
plan, code = plan_and_code_query(prompt)
```

Problems:
1. **Quick fix syndrome**: LLM may generate superficial fixes without deep diagnosis
2. **Mixed objectives**: Planning and implementation compete for tokens/attention
3. **No forced reasoning**: Brief plan may skip root cause analysis

**AIDE's solution**: Two-phase debugging
1. **Diagnose**: Analyze root cause (3-5 sentences, no code)
2. **Implement**: Write fix based on diagnosis

## Design

### Core Principle

**Control = Current ML-Master**:
- Single-phase: plan + code in one call
- Brief plan precedes code block

**Treatment = Plan-First Debugging**:
- Two-phase when `plan_constraints.enabled=true`
- Phase 1: Diagnostic analysis (text only, no code)
- Phase 2: Implementation based on diagnosis
- MCTS search otherwise unchanged

### Two-Phase Flow

```
Buggy Node → Phase 1: Diagnose → Phase 2: Implement → Debug Node
              (text only)         (code based on diagnosis)
```

**Phase 1: Diagnostic Planning**
```python
def _debug_diagnose(self, parent_node: MCTSNode) -> str:
    """Generate diagnostic plan analyzing the bug"""
    prompt = {
        "Introduction": "Analyze the bug, do NOT write code yet",
        "Task": self.task_desc,
        "Buggy Code": parent_node.code,
        "Error Output": parent_node.term_out,
        "Previous Analysis": parent_node.analysis,  # From reviewer
        "Instructions": [
            "1. Identify the root cause (not just symptoms)",
            "2. Explain why the bug occurred",
            "3. Outline the fix strategy (3-5 sentences)",
            "4. Do NOT write any code in this response"
        ]
    }
    return query(prompt, model=self.acfg.code.model)
```

**Phase 2: Implementation**
```python
def _debug_implement(self, parent_node: MCTSNode, diagnosis: str) -> tuple[str, str]:
    """Generate code fix based on diagnosis"""
    prompt = {
        "Introduction": "Implement the fix based on the diagnosis",
        "Task": self.task_desc,
        "Diagnosis": diagnosis,
        "Previous Code": parent_node.code,
        "Instructions": self._prompt_impl_guideline
    }
    return plan_and_code_query(prompt)  # Returns (plan, code)
```

---

## Implementation Plan

### Step 1: Add Configuration

**File**: `utils/config_mcts.yaml`

```yaml
# NEW: Plan constraints configuration
plan_constraints:
  enabled: false  # Control: false, Treatment: true
  debug_plan_first: true  # Two-phase debugging
  plan_max_sentences: 5  # Max sentences for diagnostic plan
  require_root_cause: true  # Enforce root cause analysis
```

**File**: `utils/config_mcts.py`

```python
@dataclass
class PlanConstraintsConfig:
    enabled: bool = False
    debug_plan_first: bool = True
    plan_max_sentences: int = 5
    require_root_cause: bool = True


@dataclass
class Config:
    # ... existing ...
    plan_constraints: PlanConstraintsConfig = field(default_factory=PlanConstraintsConfig)
```

### Step 2: Implement Diagnostic Planning

**File**: `agent/mcts_agent.py`

Add new method:
```python
def _debug_diagnose(self, parent_node: MCTSNode) -> str:
    """
    Phase 1: Diagnostic planning.
    Generate a diagnostic analysis without code implementation.
    """
    logger.info(f"Diagnostic planning for buggy node {parent_node.id}")

    introduction = (
        "You are a Kaggle grandmaster analyzing a buggy solution. "
        "Your task is to diagnose the root cause of the bug. "
        "Do NOT write any code yet - focus only on understanding WHY the bug occurred."
    )

    prompt: Any = {
        "Introduction": introduction,
        "Task description": self.task_desc,
        "Data preview": self.data_preview,
        "Buggy code": wrap_code(parent_node.code),
        "Execution output": wrap_code(parent_node.term_out, lang=""),
        "Instructions": {
            "Diagnostic Analysis Guidelines": [
                "1. Identify the specific error type (e.g., TypeError, ValueError, API misuse, logic error)",
                "2. Explain the root cause - WHY did this error occur? (Not just WHAT the error is)",
                "3. Outline a fix strategy in 3-5 sentences (but do NOT write code yet)",
                "4. If the error is due to API misuse, explain the correct API usage",
                "5. Your response should be ONLY text - no code blocks"
            ]
        }
    }

    # Build prompt based on model
    if "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning:
        user_prompt = f"""
# Task description
{prompt['Task description']}

# Data preview
{prompt['Data preview']}

# Buggy code
{prompt['Buggy code']}

# Execution output
{prompt['Execution output']}

# Instructions
{compile_prompt_to_md(prompt['Instructions'], 2)}
"""
        prompt_complete = f"<｜begin▁of▁sentence｜>{prompt['Introduction']}<｜User｜>{user_prompt}<｜Assistant｜><think>\nI will carefully analyze this bug to understand the root cause.\n"

    elif "gpt-5" in self.acfg.code.model or not self.acfg.steerable_reasoning:
        user_prompt = f"""
# Task description
{prompt['Task description']}

# Data preview
{prompt['Data preview']}

# Buggy code
{prompt['Buggy code']}

# Execution output
{prompt['Execution output']}

# Instructions
{compile_prompt_to_md(prompt['Instructions'], 2)}
"""
        prompt_complete = [
            {"role": "system", "content": prompt['Introduction']},
            {"role": "user", "content": user_prompt}
        ]

    # Query LLM for diagnosis
    if "gpt-5" in self.acfg.code.model:
        diagnosis = gpt_query(
            prompt=prompt_complete,
            temperature=self.acfg.code.temp,
            model=self.acfg.code.model,
            cfg=self.cfg
        )
    else:
        diagnosis = r1_query(
            prompt=prompt_complete,
            temperature=self.acfg.code.temp,
            model=self.acfg.code.model,
            cfg=self.cfg
        )

    # Strip any code blocks if accidentally generated
    diagnosis = extract_text_up_to_code(diagnosis)

    logger.info(f"Diagnosis generated: {diagnosis[:200]}...")
    return diagnosis


def _debug_implement(self, parent_node: MCTSNode, diagnosis: str) -> tuple[str, str]:
    """
    Phase 2: Implementation based on diagnosis.
    Generate code fix given the diagnostic analysis.
    """
    logger.info(f"Implementing fix for buggy node {parent_node.id} based on diagnosis")

    introduction = (
        "You are a Kaggle grandmaster implementing a bug fix. "
        "A diagnostic analysis has been provided. Your task is to implement the fix."
    )

    prompt: Any = {
        "Introduction": introduction,
        "Task description": self.task_desc,
        "Diagnostic Analysis": diagnosis,
        "Previous (buggy) code": wrap_code(parent_node.code),
        "Instructions": {}
    }

    prompt["Instructions"] |= self._prompt_resp_fmt
    prompt["Instructions"] |= {
        "Implementation Guidelines": [
            "Based on the diagnostic analysis above, implement the bug fix",
            "Your response should contain a brief implementation plan (2-3 sentences)",
            "followed by a single Python code block that fixes the bug",
        ]
    }
    prompt["Instructions"] |= self._prompt_impl_guideline

    # Build prompt
    if "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning:
        user_prompt = f"""
# Task description
{prompt['Task description']}

# Diagnostic Analysis
{prompt['Diagnostic Analysis']}

# Previous (buggy) code
{prompt['Previous (buggy) code']}

# Instructions
{compile_prompt_to_md(prompt['Instructions'], 2)}
"""
        prompt_complete = f"<｜begin▁of▁sentence｜>{prompt['Introduction']}<｜User｜>{user_prompt}<｜Assistant｜><think>\nBased on the diagnosis, I will now implement the fix.\n"

    elif "gpt-5" in self.acfg.code.model or not self.acfg.steerable_reasoning:
        user_prompt = f"""
# Task description
{prompt['Task description']}

# Diagnostic Analysis
{prompt['Diagnostic Analysis']}

# Previous (buggy) code
{prompt['Previous (buggy) code']}

# Instructions
{compile_prompt_to_md(prompt['Instructions'], 2)}
"""
        prompt_complete = [
            {"role": "system", "content": prompt['Introduction']},
            {"role": "user", "content": user_prompt}
        ]

    # Query LLM for implementation
    plan, code = self.plan_and_code_query(prompt_complete)
    return plan, code
```

### Step 3: Modify _debug() Method

**File**: `agent/mcts_agent.py`

Update the existing `_debug()` method:
```python
def _debug(self, parent_node: MCTSNode) -> MCTSNode:
    """Debug a buggy node, optionally using plan-first approach"""
    logger.info(f"Starting Debugging Node {parent_node.id}.")

    # Check if plan-first is enabled
    if self.cfg.plan_constraints.enabled and self.cfg.plan_constraints.debug_plan_first:
        # Two-phase debugging
        logger.info("Using two-phase debugging (diagnose then implement)")

        # Phase 1: Diagnose
        diagnosis = self._debug_diagnose(parent_node)

        # Phase 2: Implement
        plan, code = self._debug_implement(parent_node, diagnosis)

        # Store diagnosis in node metadata
        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=parent_node,
            stage="debug",
            local_best_node=parent_node.local_best_node
        )
        new_node.diagnosis = diagnosis  # Store for logging/analysis

    else:
        # Original single-phase debugging
        logger.info("Using single-phase debugging (plan + code together)")

        # ... existing _debug() implementation ...
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug..."
        )
        # ... rest of existing prompt building ...

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=parent_node,
            stage="debug",
            local_best_node=parent_node.local_best_node
        )

    logger.info(f"Debugging node {parent_node.id} to create new node {new_node.id}")
    parent_node.add_expected_child_count()
    return new_node
```

### Step 4: Enhance Node to Store Diagnosis

**File**: `search/mcts_node.py`

```python
@dataclass(eq=False)
class MCTSNode(Node):
    # ... existing fields ...

    # NEW: Store diagnostic planning when plan-first is used
    diagnosis: str | None = field(default=None, kw_only=True)
```

### Step 5: Update Logging

**File**: `utils/config_mcts.py` (save_run function)

When saving journal, include diagnosis field:
```python
def save_run(cfg: Config, journal: Journal):
    # ... existing saving logic ...

    # When exporting nodes, include diagnosis if present
    for node in journal.nodes:
        node_data = node.to_dict()
        # diagnosis will be included automatically if present in MCTSNode
```

---

## A/B Test Design

### Experimental Arms

1. **Control**: `plan_constraints.enabled=false`
   - Current single-phase debugging
   - Brief plan + code in one call

2. **Treatment**: `plan_constraints.enabled=true`
   - Two-phase: diagnose → implement
   - Separate diagnostic analysis

### Configuration Templates

**Control** (`configs/plan_constraint_control.yaml`):
```yaml
plan_constraints:
  enabled: false
```

**Treatment** (`configs/plan_constraint_treatment.yaml`):
```yaml
plan_constraints:
  enabled: true
  debug_plan_first: true
  plan_max_sentences: 5
  require_root_cause: true
```

### Metrics

**Primary**:
- **Bug resolution rate**: `num_successful_debugs / num_buggy_nodes`
- **Debugging depth**: Average number of debug attempts before success
- **First-fix success rate**: Proportion of bugs fixed on first debug attempt

**Secondary**:
- **Code quality**: Manual inspection of fixes (sample)
- **Reasoning depth**: Analysis of diagnostic quality
- **Token efficiency**: Total tokens used (diagnosis + implementation vs single-phase)

**Analysis**:
- Paired comparisons (same dataset + seed)
- Bootstrap 95% CIs
- McNemar's test for proportions (bug resolution)

---

## Runner Script

**File**: `experiments/run_plan_constraint_ab.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR=/home/ka3094/ML-Master_submit
DATASET_DIR=/path/to/mle-bench
OUT_ROOT=/home/ka3094/ML-Master_submit/experiments/plan_constraint_ab

STEPS=500
SEEDS="0,1,2"
DATASETS="plant-pathology-2020-fgvc7,house-prices,..."

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"

for enabled in "false" "true"; do
    arm=$([ "$enabled" == "true" ] && echo "with_plan_constraint" || echo "no_plan_constraint")

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
                plan_constraints.enabled=$enabled \
                plan_constraints.debug_plan_first=true \
                || true

            echo "✓ Completed: $arm / $ds / seed$seed"
        done
    done
done
```

---

## Expected Outcomes

**If plan-first debugging works**:
- **Higher resolution rate**: More bugs fixed successfully
- **Faster resolution**: Fewer debugging iterations needed
- **Better fixes**: Deeper analysis leads to more robust fixes
- **Trade-off**: More LLM calls (2× per debug), possibly higher token usage

**Potential risks**:
- **Over-analysis**: May spend tokens on diagnosis without improving fix
- **Coordination failure**: Diagnosis may not align with implementation
- **Parallel search**: Two-phase may slow down MCTS exploration

---

## Validation Checklist

- [ ] Two-phase flow works end-to-end
- [ ] Diagnosis stored in node and logged
- [ ] Fallback to single-phase when config disabled
- [ ] No code blocks in Phase 1 output
- [ ] Phase 2 successfully uses diagnosis
- [ ] Thread-safe for parallel MCTS search
- [ ] Run pilot on 2 datasets with both arms

---

## Timeline

1. **Day 1**: Implement Steps 1-2 (config, diagnostic planning)
2. **Day 2**: Implement Steps 3-4 (modify _debug, node enhancement)
3. **Day 3**: Testing, pilot run, debugging
4. **Day 4**: Full A/B test launch

**Total**: 3-4 days to implementation + testing.

---

**Next**: After Experiment 2, proceed to Experiment 3 (Bug Consultant), which is the most complex.
