#!/usr/bin/env python3
"""
Recompute per-step grading artifacts for an existing ML-Master run directory.

Useful when the run completed with a bug in selection/grading logic and you want
to regenerate `per_step_grading/grading_history.*` from `journal.json`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from search.journal import Journal
from utils import serialize
from utils.mlebench_grading import setup_per_step_grading


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the run directory containing journal.json and config_mcts.yaml.",
    )
    p.add_argument(
        "--competition-id",
        default=None,
        help="Override competition id (default: config `competition_id` or env `COMPETITION_ID`).",
    )
    p.add_argument(
        "--mlebench-data-dir",
        default=None,
        help="Override MLE-bench data dir (default: config `per_step_grading.mlebench_data_dir`).",
    )
    p.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only recompute up to this step (default: max step in journal).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir: Path = args.run_dir.resolve()

    cfg_path = run_dir / "config_mcts.yaml"
    journal_path = run_dir / "journal.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    cfg = OmegaConf.load(cfg_path)
    # Ensure output goes under the provided run dir.
    cfg.log_dir = str(run_dir)

    # Force-enable grading for recomputation.
    if not getattr(cfg, "per_step_grading", None):
        raise ValueError("Config is missing `per_step_grading` section")
    cfg.per_step_grading.enabled = True
    if args.mlebench_data_dir is not None:
        cfg.per_step_grading.mlebench_data_dir = args.mlebench_data_dir

    competition_id = (
        args.competition_id
        or os.environ.get("COMPETITION_ID")
        or getattr(cfg, "competition_id", None)
    )
    if not competition_id:
        raise ValueError("No competition id provided (set --competition-id or COMPETITION_ID or cfg.competition_id)")

    journal = serialize.load_json(journal_path, Journal)
    max_step = args.max_step
    if max_step is None:
        steps = [getattr(n, "step", None) for n in getattr(journal, "nodes", [])]
        steps = [s for s in steps if isinstance(s, int)]
        max_step = max(steps) if steps else 0

    workspace_dir = Path(getattr(cfg, "workspace_dir", run_dir / "workspace")).resolve()

    callback = setup_per_step_grading(cfg, competition_id)
    if callback is None:
        raise RuntimeError("setup_per_step_grading returned None (check cfg.per_step_grading.* and competition id)")

    # ML-Master passes step numbers excluding the virtual root (step starts at 1).
    for step in range(1, max_step + 1):
        callback.on_step_complete(journal, step, workspace_dir, cfg)

    try:
        callback.save_results()
    except Exception:
        # Incremental outputs may already have been written; don't fail recomputation due to final export.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

