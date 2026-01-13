"""
Experiment runner for post-search selection A/B testing.
Compares different post-search selection strategies on the same task.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_experiment(
    data_dir: str,
    desc_file: str,
    strategy: str,
    seed: int,
    log_dir: Path,
    exp_name: str,
    steps: int = 50,
    k_fold: int = 5,
    extra_args: list[str] = None,
) -> dict:
    """
    Run a single ML-Master experiment with specified post-search strategy.

    Returns:
        dict: Experiment results including selected node info
    """
    cmd = [
        sys.executable,
        "main_mcts.py",
        f"data_dir={data_dir}",
        f"desc_file={desc_file}",
        f"exp_name={exp_name}",
        f"log_dir={log_dir}",
        f"agent.steps={steps}",
        f"agent.k_fold_validation={k_fold}",
        f"post_search.selection={strategy}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Experiment failed with exit code {e.returncode}")
        success = False

    # Parse final_selection.json if it exists
    selection_file = log_dir / exp_name / "final_selection.json"
    results = {
        "strategy": strategy,
        "seed": seed,
        "exp_name": exp_name,
        "success": success,
        "selected_node_id": None,
        "selected_metric": None,
        "selected_cv_mean": None,
        "selected_cv_std": None,
        "selected_cv_folds": None,
    }

    if selection_file.exists():
        try:
            with open(selection_file) as f:
                selection_data = json.load(f)
                results.update({
                    "selected_node_id": selection_data.get("selected_node_id"),
                    "selected_metric": selection_data.get("selected_metric"),
                    "selected_cv_mean": selection_data.get("selected_cv_mean"),
                    "selected_cv_std": selection_data.get("selected_cv_std"),
                    "selected_cv_folds": selection_data.get("selected_cv_folds"),
                })
        except Exception as e:
            print(f"WARNING: Failed to parse {selection_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run post-search selection A/B experiments for ML-Master"
    )
    parser.add_argument("--data-dir", required=True, help="Path to task data directory")
    parser.add_argument("--desc-file", required=True, help="Path to task description file")
    parser.add_argument(
        "--strategies",
        default="best_valid,maximin,elite_maximin,mean_minus_k_std",
        help="Comma-separated list of strategies to test (default: best_valid,maximin,elite_maximin,mean_minus_k_std)",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated list of random seeds (default: 0,1,2)",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of MCTS steps (default: 50)"
    )
    parser.add_argument(
        "--k-fold", type=int, default=5, help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/results",
        help="Directory to save results (default: ./experiments/results)",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        help="Additional arguments to pass to main_mcts.py (e.g., agent.code.model=gpt-4o)",
    )

    args = parser.parse_args()

    # Parse strategies and seeds
    strategies = [s.strip() for s in args.strategies.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"post_search_ab_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Post-Search Selection A/B Testing")
    print("=" * 80)
    print(f"Data dir: {args.data_dir}")
    print(f"Task desc: {args.desc_file}")
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {args.steps}")
    print(f"K-fold: {args.k_fold}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)
    print()

    # Run all experiments
    all_results = []

    for strategy in strategies:
        for seed in seeds:
            print("=" * 80)
            print(f"Running: strategy={strategy}, seed={seed}")
            print("=" * 80)

            exp_name = f"{strategy}_seed{seed}"
            results = run_experiment(
                data_dir=args.data_dir,
                desc_file=args.desc_file,
                strategy=strategy,
                seed=seed,
                log_dir=output_dir,
                exp_name=exp_name,
                steps=args.steps,
                k_fold=args.k_fold,
                extra_args=args.extra_args,
            )
            all_results.append(results)
            print()

    # Save summary
    df = pd.DataFrame(all_results)
    summary_file = output_dir / "summary.csv"
    df.to_csv(summary_file, index=False)

    print("=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Summary CSV: {summary_file}")
    print()
    print("Summary statistics:")
    print(df[["strategy", "seed", "selected_metric", "selected_cv_mean", "selected_cv_std"]])
    print()

    # Compute mean and std for each strategy
    print("Strategy comparison (mean ± std across seeds):")
    strategy_stats = df.groupby("strategy")[["selected_metric", "selected_cv_mean"]].agg(["mean", "std"])
    print(strategy_stats)


if __name__ == "__main__":
    main()
