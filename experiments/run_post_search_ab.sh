#!/bin/bash
# A/B Testing script for post-search selection strategies
# Tests different selection strategies on the same task with different seeds

# Usage: bash run_post_search_ab.sh <data_dir> <desc_file>
# Example: bash run_post_search_ab.sh /path/to/data /path/to/task.md

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir> <desc_file> [seeds]"
    echo "Example: $0 /path/to/data /path/to/task.md '0,1,2'"
    exit 1
fi

DATA_DIR="$1"
DESC_FILE="$2"
SEEDS="${3:-0,1,2}"  # Default to seeds 0,1,2

# Convert comma-separated seeds to array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Post-search selection strategies to test
STRATEGIES=(
    "best_valid"
    "maximin"
    "elite_maximin"
    "mean_minus_k_std"
    "maximin_no_filter"
)

# Create results directory
RESULTS_DIR="./experiments/results/post_search_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Running A/B tests for post-search selection strategies"
echo "Data dir: $DATA_DIR"
echo "Task desc: $DESC_FILE"
echo "Seeds: ${SEED_ARRAY[@]}"
echo "Strategies: ${STRATEGIES[@]}"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Run experiments for each strategy and seed combination
for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        echo "=========================================="
        echo "Running: strategy=$strategy, seed=$seed"
        echo "=========================================="

        EXP_NAME="${strategy}_seed${seed}"
        LOG_DIR="$RESULTS_DIR"

        # Run ML-Master with the specific strategy
        python3 main_mcts.py \
            data_dir="$DATA_DIR" \
            desc_file="$DESC_FILE" \
            exp_name="$EXP_NAME" \
            log_dir="$LOG_DIR" \
            agent.steps=50 \
            agent.k_fold_validation=5 \
            post_search.selection="$strategy" \
            || echo "WARNING: Experiment $EXP_NAME failed"

        echo ""
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

# Generate summary CSV
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "strategy,seed,selected_node_id,selected_metric,selected_cv_mean,selected_cv_std" > "$SUMMARY_FILE"

for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        EXP_NAME="${strategy}_seed${seed}"
        SELECTION_FILE="$RESULTS_DIR/$EXP_NAME/final_selection.json"

        if [ -f "$SELECTION_FILE" ]; then
            # Extract metrics from final_selection.json using jq (if available)
            if command -v jq &> /dev/null; then
                NODE_ID=$(jq -r '.selected_node_id // "N/A"' "$SELECTION_FILE")
                METRIC=$(jq -r '.selected_metric // "N/A"' "$SELECTION_FILE")
                CV_MEAN=$(jq -r '.selected_cv_mean // "N/A"' "$SELECTION_FILE")
                CV_STD=$(jq -r '.selected_cv_std // "N/A"' "$SELECTION_FILE")
                echo "$strategy,$seed,$NODE_ID,$METRIC,$CV_MEAN,$CV_STD" >> "$SUMMARY_FILE"
            else
                echo "$strategy,$seed,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
            fi
        else
            echo "$strategy,$seed,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
        fi
    done
done

echo ""
echo "Summary CSV saved to: $SUMMARY_FILE"
echo ""
echo "To analyze results, you can use:"
echo "  cat $SUMMARY_FILE"
echo "  # Or open in a spreadsheet application"
