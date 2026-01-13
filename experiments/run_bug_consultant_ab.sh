#!/bin/bash
# A/B testing: Bug Consultant v2 (RAG + RL + summarization)
# Compares agent.search.use_bug_consultant=false (control) vs true (treatment)
#
# Usage: bash experiments/run_bug_consultant_ab.sh <data_dir> <desc_file> [seeds]
# Example: bash experiments/run_bug_consultant_ab.sh /path/to/task /path/to/task.md "0,1,2"

set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <data_dir> <desc_file> [seeds]"
  echo "Example: $0 /path/to/task /path/to/task.md '0,1,2'"
  exit 1
fi

DATA_DIR="$1"
DESC_FILE="$2"
SEEDS="${3:-0,1,2}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

ARMS=("control" "treatment")

RESULTS_DIR="./experiments/results/bug_consultant_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Running A/B tests for bug consultant"
echo "Data dir: $DATA_DIR"
echo "Task desc: $DESC_FILE"
echo "Seeds: ${SEED_ARRAY[@]}"
echo "Arms: ${ARMS[@]}"
echo "Results dir: $RESULTS_DIR"
echo ""

for arm in "${ARMS[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    echo "=========================================="
    echo "Running: arm=$arm, seed=$seed"
    echo "=========================================="

    EXP_NAME="bug_${arm}_seed${seed}"
    USE_CONSULTANT="false"
    if [ "$arm" = "treatment" ]; then
      USE_CONSULTANT="true"
    fi

    python3 main_mcts.py \
      data_dir="$DATA_DIR" \
      desc_file="$DESC_FILE" \
      exp_name="$EXP_NAME" \
      log_dir="$RESULTS_DIR" \
      agent.steps=50 \
      agent.search.use_bug_consultant="$USE_CONSULTANT" \
      agent.search.bug_context_mode="consultant" \
      || echo "WARNING: Experiment $EXP_NAME failed"

    echo ""
  done
done

SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "arm,seed,selected_node_id,selected_metric,selected_cv_mean,selected_cv_std" > "$SUMMARY_FILE"

for arm in "${ARMS[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    EXP_NAME="bug_${arm}_seed${seed}"
    SELECTION_FILE="$RESULTS_DIR/$EXP_NAME/final_selection.json"

    if [ -f "$SELECTION_FILE" ] && command -v jq &> /dev/null; then
      NODE_ID=$(jq -r '.selected_node_id // "N/A"' "$SELECTION_FILE")
      METRIC=$(jq -r '.selected_metric // "N/A"' "$SELECTION_FILE")
      CV_MEAN=$(jq -r '.selected_cv_mean // "N/A"' "$SELECTION_FILE")
      CV_STD=$(jq -r '.selected_cv_std // "N/A"' "$SELECTION_FILE")
      echo "$arm,$seed,$NODE_ID,$METRIC,$CV_MEAN,$CV_STD" >> "$SUMMARY_FILE"
    else
      echo "$arm,$seed,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
    fi
  done
done

echo "=========================================="
echo "Done. Summary: $SUMMARY_FILE"
echo "=========================================="

