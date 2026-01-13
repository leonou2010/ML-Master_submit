# Experiment 3: Bug Consultant (RAG + RL + Summarization) for ML-Master

**Priority**: Phase 3 (Implement Last)
**Risk Level**: High
**Expected Impact**: Very High
**Estimated Time**: 5-7 days

## Overview

Implement an intelligent debugging system that learns from past bugs to improve future debugging. This is the **most complex but potentially highest-impact** experiment, bringing AIDE's Bug Consultant v2 architecture to ML-Master's MCTS search.

## Motivation

### Current ML-Master Debugging

```python
def _debug(parent_node):
    prompt = {
        "Buggy Code": parent_node.code,
        "Error": parent_node.term_out
    }
    plan, code = query(prompt)
    return MCTSNode(plan=plan, code=code)
```

**Problems**:
1. **No memory**: Each debugging attempt is independent
2. **Repeated failures**: May try the same failed approach multiple times
3. **No learning**: Doesn't accumulate debugging knowledge
4. **No retrieval**: Can't leverage similar past bugs

**Example failure pattern**:
- Bug 1: `TypeError: LGBMRegressor.fit() got unexpected keyword argument 'early_stopping_rounds'`
- Fix attempt 1: Remove `early_stopping_rounds` → Still crashes
- Fix attempt 2: Try `early_stopping_rounds` differently → Still crashes
- Fix attempt 3: Search online (but LLM can't do this in real-time)
- **Without memory, may repeat attempts 1-2 multiple times**

### AIDE's Bug Consultant v2 Solution

Three-stage architecture:
1. **Bug Start**: Extract error signature, category, context tags (for RAG)
2. **Trial Recording**: Learn what works/fails at each attempt
3. **Bug Completion**: Synthesize final root cause and lessons

**Benefits**:
- **RAG retrieval**: "Similar bug in step 23 was fixed by using callbacks instead"
- **RL-style learning**: "Don't try early_stopping_rounds again, it failed 3 times"
- **World model**: Accumulated knowledge about the environment

---

## Architecture Design for ML-Master

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    BugConsultant                         │
├─────────────────────────────────────────────────────────┤
│  - bug_records: list[BugRecord]                          │
│  - world_model: str                                      │
│  - error_embeddings: dict                                │
│                                                          │
│  Methods:                                                │
│  + start_bug_record(node) -> BugRecord                   │
│  + record_trial(bug_record, outcome)                     │
│  + complete_bug_record(bug_record)                       │
│  + retrieve_similar_bugs(error_sig) -> list[BugRecord]  │
│  + get_advice(current_bug) -> str                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
          ┌─────────────────────────────┐
          │       BugRecord              │
          ├─────────────────────────────┤
          │  - id, node_id               │
          │  - error_signature           │
          │  - error_category            │
          │  - context_tags              │
          │  - trials: list[DebugTrial] │
          │  - failed_strategies         │
          │  - learned_constraints       │
          │  - successful_strategy       │
          └─────────────────────────────┘
```

### Data Flow

```
Buggy Node Detected
    │
    ▼
start_bug_record()  ← LLM Summarization: error signature, category, tags
    │
    ▼
retrieve_similar_bugs()  ← RAG: semantic search on error signatures
    │
    ▼
_debug() with advice  ← Includes: similar bugs + failed strategies
    │
    ▼
Execute fix attempt
    │
    ├─→ Success → record_trial(outcome="success")
    │             → complete_bug_record()
    │             → Extract successful strategy
    │
    └─→ Failure → record_trial(outcome="failed")
                  → Extract why_failed, learned_constraint
                  → Loop back to _debug() with updated advice
```

---

## Implementation Plan

### Step 1: Create Bug Record Data Structures

**File**: `utils/bug_consultant.py` (new file)

```python
"""Bug Consultant for ML-Master: RAG + RL + Summarization"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import uuid
import json
import logging
from pathlib import Path

logger = logging.getLogger("ml-master")


@dataclass
class DebugTrial:
    """Single debugging attempt"""
    trial_num: int
    debug_plan: str
    debug_code: str
    outcome: Literal["success", "failed"]
    error_output: str | None = None
    metric: float | None = None

    # Summarized by LLM
    why_worked: str | None = None  # If success
    why_failed: str | None = None  # If failed
    failed_strategy_summary: str | None = None  # If failed
    learned_constraint: str | None = None  # If failed
    successful_strategy_summary: str | None = None  # If success
    key_insight: str | None = None  # If success


@dataclass
class BugRecord:
    """Episode tracking for a single bug"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    node_step: int = 0

    # Original bug info
    original_code: str = ""
    original_error: str = ""
    original_plan: str = ""

    # LLM-extracted (for RAG)
    error_signature: str = ""  # Max 200 chars, specific error message
    error_category: str = ""  # API_MISUSE, TYPE_ERROR, VALUE_ERROR, etc.
    initial_hypothesis: str = ""  # Max 300 chars, preliminary root cause
    context_tags: list[str] = field(default_factory=list)  # For matching

    # Debugging trials
    trials: list[DebugTrial] = field(default_factory=list)
    num_trials: int = 0

    # Incremental learning (accumulated across trials)
    failed_strategies: list[str] = field(default_factory=list)
    learned_constraints: list[str] = field(default_factory=list)

    # Final outcome
    resolved: bool = False
    successful_strategy: str | None = None
    final_root_cause: str | None = None  # LLM-synthesized at completion

    # Metadata
    created_at: str = ""
    completed_at: str | None = None


@dataclass
class WorldModel:
    """Accumulated debugging knowledge"""
    observations: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    common_patterns: list[str] = field(default_factory=list)
    api_constraints: list[str] = field(default_factory=list)
```

### Step 2: Implement LLM Summarization Functions

**File**: `utils/bug_consultant.py` (continued)

```python
from backend import query, FunctionSpec


# Function specs for LLM structured output
bug_start_spec = FunctionSpec(
    name="submit_bug_start_summary",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "Specific error message (max 200 chars), e.g. 'TypeError: LGBMRegressor.fit() got unexpected keyword argument early_stopping_rounds'"
            },
            "error_category": {
                "type": "string",
                "enum": ["API_MISUSE", "TYPE_ERROR", "VALUE_ERROR", "ATTRIBUTE_ERROR", "IMPORT_ERROR", "LOGIC_ERROR", "DATA_ERROR", "OTHER"],
                "description": "High-level error category"
            },
            "initial_hypothesis": {
                "type": "string",
                "description": "Preliminary root cause analysis (max 300 chars)"
            },
            "context_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant tags for matching (e.g. ['lightgbm', 'sklearn_api', 'early_stopping'])"
            }
        },
        "required": ["error_signature", "error_category", "initial_hypothesis", "context_tags"]
    },
    description="Summarize bug start information for retrieval"
)


trial_failure_spec = FunctionSpec(
    name="submit_trial_failure",
    json_schema={
        "type": "object",
        "properties": {
            "why_failed": {
                "type": "string",
                "description": "Why this specific approach failed (max 250 chars)"
            },
            "failed_strategy_summary": {
                "type": "string",
                "description": "One-line summary of what was tried (max 150 chars)"
            },
            "learned_constraint": {
                "type": "string",
                "description": "Reusable rule learned from this failure (max 200 chars)"
            }
        },
        "required": ["why_failed", "failed_strategy_summary", "learned_constraint"]
    },
    description="Summarize failed debugging trial"
)


trial_success_spec = FunctionSpec(
    name="submit_trial_success",
    json_schema={
        "type": "object",
        "properties": {
            "why_worked": {
                "type": "string",
                "description": "Why this approach succeeded (max 250 chars)"
            },
            "successful_strategy_summary": {
                "type": "string",
                "description": "One-line summary of the fix (max 150 chars)"
            },
            "key_insight": {
                "type": "string",
                "description": "Core insight that led to success (max 200 chars)"
            }
        },
        "required": ["why_worked", "successful_strategy_summary", "key_insight"]
    },
    description="Summarize successful debugging trial"
)


def summarize_bug_start(
    code: str,
    error_output: str,
    plan: str,
    cfg
) -> dict:
    """Stage 1: Summarize bug start for RAG retrieval"""
    prompt = f"""Analyze this bug and extract key information for semantic search.

Code:
{code}

Error Output:
{error_output}

Previous Plan:
{plan}

Extract:
1. error_signature: The specific error message (200 chars max)
2. error_category: High-level category (API_MISUSE, TYPE_ERROR, etc.)
3. initial_hypothesis: Why you think this error occurred (300 chars max)
4. context_tags: List of relevant keywords (libraries, APIs, concepts)
"""

    result = query(
        prompt=prompt,
        model=cfg.agent.feedback.model,  # Use GPT-4o for summarization
        temperature=0.3,
        function_spec=bug_start_spec,
        cfg=cfg
    )

    return result


def summarize_trial_failure(
    bug_record: BugRecord,
    trial: DebugTrial,
    cfg
) -> dict:
    """Stage 2a: Summarize failed debugging attempt"""
    prev_trials_summary = "\n".join([
        f"Trial {t.trial_num}: {t.failed_strategy_summary}"
        for t in bug_record.trials if t.outcome == "failed"
    ])

    prompt = f"""Analyze this failed debugging attempt and extract learnings.

Original Bug:
{bug_record.error_signature}

Current Approach:
Plan: {trial.debug_plan}
Code: {trial.debug_code}

Error Output:
{trial.error_output}

Previous Failed Attempts:
{prev_trials_summary if prev_trials_summary else "None"}

Extract:
1. why_failed: Specific reason this approach failed
2. failed_strategy_summary: One-line summary of what was tried
3. learned_constraint: Reusable rule for future debugging
"""

    result = query(
        prompt=prompt,
        model=cfg.agent.feedback.model,
        temperature=0.3,
        function_spec=trial_failure_spec,
        cfg=cfg
    )

    return result


def summarize_trial_success(
    bug_record: BugRecord,
    trial: DebugTrial,
    cfg
) -> dict:
    """Stage 2b: Summarize successful debugging attempt"""
    prompt = f"""Analyze this successful bug fix and extract key insights.

Original Bug:
{bug_record.error_signature}

Successful Fix:
Plan: {trial.debug_plan}
Code: {trial.debug_code}

Extract:
1. why_worked: Why this approach succeeded
2. successful_strategy_summary: One-line summary of the fix
3. key_insight: Core insight for solving similar bugs
"""

    result = query(
        prompt=prompt,
        model=cfg.agent.feedback.model,
        temperature=0.3,
        function_spec=trial_success_spec,
        cfg=cfg
    )

    return result
```

### Step 3: Implement BugConsultant Class

**File**: `utils/bug_consultant.py` (continued)

```python
import numpy as np
from typing import Any


class BugConsultant:
    """
    Intelligent debugging assistant with RAG + RL + Summarization.

    Features:
    - RAG: Retrieves similar past bugs via semantic search
    - RL: Learns failed strategies to avoid repetition
    - Summarization: Multi-stage LLM compression of debugging knowledge
    """

    def __init__(self, cfg, log_dir: Path):
        self.cfg = cfg
        self.log_dir = log_dir
        self.consultant_dir = log_dir / "bug_consultant"
        self.consultant_dir.mkdir(exist_ok=True)

        self.bug_records: list[BugRecord] = []
        self.world_model = WorldModel()

        # For RAG: store embeddings of error signatures
        self.error_signatures: list[str] = []
        self.error_embeddings: np.ndarray | None = None

        # Thread safety for parallel MCTS
        import threading
        self.lock = threading.Lock()

    def start_bug_record(
        self,
        node_id: str,
        node_step: int,
        code: str,
        error_output: str,
        plan: str
    ) -> BugRecord:
        """
        Stage 1: Start tracking a new bug.
        Extracts error signature, category, and context tags via LLM.
        """
        logger.info(f"Starting bug record for node {node_id}")

        # LLM summarization
        summary = summarize_bug_start(code, error_output, plan, self.cfg)

        bug_record = BugRecord(
            node_id=node_id,
            node_step=node_step,
            original_code=code,
            original_error=error_output,
            original_plan=plan,
            error_signature=summary["error_signature"],
            error_category=summary["error_category"],
            initial_hypothesis=summary["initial_hypothesis"],
            context_tags=summary["context_tags"],
            created_at=str(uuid.uuid4())
        )

        with self.lock:
            self.bug_records.append(bug_record)

        # Save bug record
        self._save_bug_record(bug_record)

        logger.info(f"Bug record created: {bug_record.id}")
        logger.info(f"  Error signature: {bug_record.error_signature}")
        logger.info(f"  Category: {bug_record.error_category}")
        logger.info(f"  Context tags: {bug_record.context_tags}")

        return bug_record

    def record_trial(
        self,
        bug_record: BugRecord,
        trial_num: int,
        debug_plan: str,
        debug_code: str,
        outcome: Literal["success", "failed"],
        error_output: str | None = None,
        metric: float | None = None
    ):
        """
        Stage 2: Record a debugging trial (success or failure).
        Extracts learnings via LLM summarization.
        """
        logger.info(f"Recording trial {trial_num} for bug {bug_record.id}: {outcome}")

        trial = DebugTrial(
            trial_num=trial_num,
            debug_plan=debug_plan,
            debug_code=debug_code,
            outcome=outcome,
            error_output=error_output,
            metric=metric
        )

        if outcome == "failed":
            # Summarize failure
            failure_summary = summarize_trial_failure(bug_record, trial, self.cfg)
            trial.why_failed = failure_summary["why_failed"]
            trial.failed_strategy_summary = failure_summary["failed_strategy_summary"]
            trial.learned_constraint = failure_summary["learned_constraint"]

            # Accumulate learnings
            with self.lock:
                bug_record.failed_strategies.append(trial.failed_strategy_summary)
                bug_record.learned_constraints.append(trial.learned_constraint)

            logger.info(f"  Why failed: {trial.why_failed}")
            logger.info(f"  Learned: {trial.learned_constraint}")

        elif outcome == "success":
            # Summarize success
            success_summary = summarize_trial_success(bug_record, trial, self.cfg)
            trial.why_worked = success_summary["why_worked"]
            trial.successful_strategy_summary = success_summary["successful_strategy_summary"]
            trial.key_insight = success_summary["key_insight"]

            with self.lock:
                bug_record.successful_strategy = trial.successful_strategy_summary
                bug_record.resolved = True

            logger.info(f"  Why worked: {trial.why_worked}")
            logger.info(f"  Key insight: {trial.key_insight}")

        with self.lock:
            bug_record.trials.append(trial)
            bug_record.num_trials += 1

        self._save_bug_record(bug_record)

    def retrieve_similar_bugs(
        self,
        error_signature: str,
        context_tags: list[str],
        top_k: int = 3
    ) -> list[BugRecord]:
        """
        RAG retrieval: Find similar past bugs via semantic search.

        Uses:
        1. Embedding similarity on error signatures
        2. Context tag overlap
        3. Error category matching
        """
        if not self.bug_records:
            return []

        # Compute embedding for current error (if embeddings enabled)
        # For now, use simple text matching + tag overlap

        similarities = []
        for record in self.bug_records:
            if record.id == error_signature:  # Skip self
                continue

            # Text similarity (simple word overlap for now)
            sig_words = set(error_signature.lower().split())
            record_words = set(record.error_signature.lower().split())
            text_sim = len(sig_words & record_words) / max(len(sig_words | record_words), 1)

            # Tag overlap
            tag_sim = len(set(context_tags) & set(record.context_tags)) / max(len(set(context_tags) | set(record.context_tags)), 1)

            # Category match
            category_boost = 0.5 if record.error_category == error_signature else 0

            # Combined score
            score = 0.4 * text_sim + 0.4 * tag_sim + 0.2 * category_boost

            similarities.append((score, record))

        # Sort and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [record for score, record in similarities[:top_k] if score > 0.1]

    def get_advice(self, bug_record: BugRecord) -> str:
        """
        Generate debugging advice based on:
        1. Similar past bugs (RAG)
        2. Failed strategies for this bug (RL)
        3. Learned constraints (RL)
        """
        advice = []

        # 1. Similar bugs
        similar_bugs = self.retrieve_similar_bugs(
            bug_record.error_signature,
            bug_record.context_tags,
            top_k=self.cfg.agent.bug_consultant.max_similar_bugs
        )

        if similar_bugs:
            advice.append("## Similar Past Bugs\n")
            for i, sim_bug in enumerate(similar_bugs, 1):
                advice.append(f"{i}. Error: {sim_bug.error_signature}")
                if sim_bug.successful_strategy:
                    advice.append(f"   Solution: {sim_bug.successful_strategy}")
                elif sim_bug.failed_strategies:
                    advice.append(f"   Failed attempts: {', '.join(sim_bug.failed_strategies[:2])}")
                advice.append("")

        # 2. Failed strategies for current bug
        if bug_record.failed_strategies:
            advice.append("## Approaches That Failed (Don't Repeat)\n")
            for strategy in bug_record.failed_strategies[-5:]:  # Last 5
                advice.append(f"- {strategy}")
            advice.append("")

        # 3. Learned constraints
        if bug_record.learned_constraints:
            advice.append("## Learned Constraints\n")
            for constraint in bug_record.learned_constraints[-5:]:
                advice.append(f"- {constraint}")
            advice.append("")

        return "\n".join(advice) if advice else ""

    def complete_bug_record(self, bug_record: BugRecord):
        """Stage 3: Finalize bug record after resolution or timeout"""
        logger.info(f"Completing bug record {bug_record.id}")

        with self.lock:
            bug_record.completed_at = str(uuid.uuid4())

        self._save_bug_record(bug_record)

        # Update world model
        self._update_world_model(bug_record)

    def _save_bug_record(self, bug_record: BugRecord):
        """Save bug record to disk"""
        bug_file = self.consultant_dir / f"bug_{bug_record.id}.json"
        with open(bug_file, 'w') as f:
            json.dump(bug_record.__dict__, f, indent=2, default=str)

    def _update_world_model(self, bug_record: BugRecord):
        """Update world model with insights from completed bug"""
        # Add successful strategy to world model
        if bug_record.successful_strategy:
            self.world_model.insights.append(
                f"Bug '{bug_record.error_category}': {bug_record.successful_strategy}"
            )

        # Add learned constraints
        for constraint in bug_record.learned_constraints:
            if constraint not in self.world_model.api_constraints:
                self.world_model.api_constraints.append(constraint)

        # Save world model
        world_model_file = self.consultant_dir / "world_model.json"
        with open(world_model_file, 'w') as f:
            json.dump(self.world_model.__dict__, f, indent=2, default=str)
```

### Step 4: Integrate into Agent

**File**: `agent/mcts_agent.py`

```python
from utils.bug_consultant import BugConsultant

class MCTSAgent:
    def __init__(self, task_desc: str, cfg: Config, journal: Journal):
        # ... existing init ...

        # NEW: Initialize bug consultant if enabled
        if cfg.agent.search.use_bug_consultant:
            from utils.bug_consultant import BugConsultant
            self.bug_consultant = BugConsultant(cfg, Path(cfg.log_dir))
            self.active_bug_records = {}  # Map node_id -> BugRecord
        else:
            self.bug_consultant = None
            self.active_bug_records = {}

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        """Debug with bug consultant assistance"""
        logger.info(f"Starting Debugging Node {parent_node.id}")

        # NEW: Start bug record if consultant enabled
        bug_record = None
        if self.bug_consultant:
            bug_record = self.bug_consultant.start_bug_record(
                node_id=parent_node.id,
                node_step=parent_node.step,
                code=parent_node.code,
                error_output=parent_node.term_out,
                plan=parent_node.plan
            )
            self.active_bug_records[parent_node.id] = bug_record

            # Get advice (similar bugs + failed strategies)
            advice = self.bug_consultant.get_advice(bug_record)
        else:
            advice = ""

        # Build debugging prompt
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug..."
        )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
        }

        # NEW: Add bug consultant advice if available
        if advice:
            prompt["Debugging Advice"] = advice

        prompt |= {
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {}
        }

        # ... rest of prompt building ...

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=parent_node,
            stage="debug",
            local_best_node=parent_node.local_best_node
        )

        # NEW: Link bug record to new node
        if bug_record:
            new_node.bug_record_id = bug_record.id

        logger.info(f"Debugging node {parent_node.id} to create new node {new_node.id}")
        return new_node

    # NEW: Method to record trial outcome
    def record_debug_outcome(self, debug_node: MCTSNode):
        """Record debugging trial outcome after execution"""
        if not self.bug_consultant or not hasattr(debug_node, 'bug_record_id'):
            return

        bug_record_id = debug_node.bug_record_id
        bug_record = next((br for br in self.bug_consultant.bug_records if br.id == bug_record_id), None)

        if not bug_record:
            return

        trial_num = bug_record.num_trials + 1
        outcome = "success" if not debug_node.is_buggy else "failed"

        self.bug_consultant.record_trial(
            bug_record=bug_record,
            trial_num=trial_num,
            debug_plan=debug_node.plan,
            debug_code=debug_node.code,
            outcome=outcome,
            error_output=debug_node.term_out if debug_node.is_buggy else None,
            metric=debug_node.metric.value if debug_node.metric else None
        )

        # If resolved, complete bug record
        if outcome == "success":
            self.bug_consultant.complete_bug_record(bug_record)
            del self.active_bug_records[debug_node.parent.id]
```

### Step 5: Configuration

**File**: `utils/config_mcts.yaml`

```yaml
agent:
  search:
    # ... existing ...

    # NEW: Bug consultant configuration
    use_bug_consultant: false  # Control: false, Treatment: true

    bug_consultant:
      max_similar_bugs: 3  # Number of similar bugs to retrieve
      max_failed_strategies: 5  # Max failed strategies to show
      embedding_model: text-embedding-3-small  # For future embedding-based RAG
      enable_world_model: true  # Maintain global debugging knowledge
```

**File**: `utils/config_mcts.py`

```python
@dataclass
class BugConsultantConfig:
    max_similar_bugs: int = 3
    max_failed_strategies: int = 5
    embedding_model: str = "text-embedding-3-small"
    enable_world_model: bool = True


@dataclass
class SearchConfig:
    # ... existing ...
    use_bug_consultant: bool = False
    bug_consultant: BugConsultantConfig = field(default_factory=BugConsultantConfig)
```

---

## A/B Test Design

### Arms

1. **Control**: `use_bug_consultant=false`
2. **Treatment**: `use_bug_consultant=true`

### Metrics

**Primary**:
- **Bug resolution rate**: `resolved_bugs / total_bugs`
- **Time to resolution**: Average debugging iterations before success

**Secondary**:
- **Repeated error patterns**: Count of identical errors across trials
- **Advice relevance**: Manual inspection of retrieved similar bugs
- **World model growth**: Number of insights accumulated

---

## Expected Outcomes

**If bug consultant works**:
- **Higher resolution rate**: More bugs fixed successfully
- **Faster resolution**: Fewer wasted iterations on failed strategies
- **Better debugging**: Leverages accumulated knowledge
- **Trade-off**: More LLM calls for summarization

---

## Validation Checklist

- [ ] Bug records created and saved correctly
- [ ] LLM summarization works for all 3 stages
- [ ] RAG retrieval finds relevant similar bugs
- [ ] Failed strategies prevent repetition
- [ ] Thread-safe for parallel MCTS
- [ ] Pilot run on 2 datasets

---

## Timeline

- **Days 1-2**: Data structures + summarization functions
- **Days 3-4**: BugConsultant class + RAG retrieval
- **Days 5-6**: Agent integration + testing
- **Day 7**: Pilot run + debugging

**Total**: 5-7 days to implementation + testing.

---

**This is the final and most complex experiment. Success here would give ML-Master a significant debugging advantage over all current AutoML systems.**
