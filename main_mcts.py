import atexit
import logging
import shutil
import sys
from pathlib import Path
import os
import random

import backend
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from agent.mcts_agent import MCTSAgent as Agent
from interpreter.interpreter_parallel import Interpreter
from search.journal import Journal
from search.node import Node
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from utils.config_mcts import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils import serialize

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        metric_value = node.metric.value if node.metric is not None else None
        if node.is_buggy or metric_value is None:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {metric_value:.3f} (best)"
            else:
                s = f"[{style}green]● {metric_value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        metric_value = node.metric.value if node.metric is not None else None
        if node.is_buggy or metric_value is None:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str:
                s = f"{indent}● {metric_value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {metric_value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str


def run():
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # dont want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("ml-master")
    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "ml-master.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "ml-master.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')

    # Optional deterministic seed (useful for multi-seed sweeps).
    seed_env = os.environ.get("MLMASTER_SEED") or os.environ.get("SEED")
    if seed_env is not None:
        try:
            seed = int(seed_env)
            random.seed(seed)
        except ValueError:
            logger.warning(f"Invalid seed env var (expected int): {seed_env!r}")

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    if cfg.agent.steerable_reasoning == True:
        logger.warning("Steerable reasoning is enabled, please make sure your open sourced model api support `client.compeletion.create()`, otherwise the process may fail")
        if "gpt" in cfg.agent.code.model or "gemini" in cfg.agent.code.model or "claude" in cfg.agent.code.model:
            logger.warning("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
            raise ValueError("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
    
    if cfg.agent.check_format == True:
        logger.warning("Check format is enabled, please make sure you have launched the server, or this step will be skipped")


    atexit.register(cleanup)

    journal_path = cfg.log_dir / "journal.json"
    if getattr(cfg, "resume", False) and journal_path.exists():
        logger.info(f"[Resume] Loading journal from: {journal_path}")
        journal = serialize.load_json(journal_path, Journal)
    else:
        journal = Journal()

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    # Setup per-step grading (for generalization gap experiments)
    from utils.mlebench_grading import setup_per_step_grading

    competition_id = os.environ.get("COMPETITION_ID") or getattr(cfg, "competition_id", None)
    grading_callback = setup_per_step_grading(cfg, competition_id)

    max_workers = cfg.agent.search.parallel_search_num
    total_steps = cfg.agent.steps
    completed = max(0, len(journal) - 1)  # exclude virtual root

    if completed >= total_steps:
        logger.info(
            f"[Resume] Search already complete (completed={completed}, total_steps={total_steps}). "
            "Proceeding to post-search selection/export."
        )
        max_workers = 0

    if max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            remaining = total_steps - completed
            futures = {executor.submit(step_task) for _ in range(min(max_workers, remaining))}
            consecutive_failures = 0
            lock = threading.Lock()
            while completed < total_steps:
                if not futures:
                    raise RuntimeError(
                        "No active worker futures but search is not complete. "
                        "This indicates a scheduling bug in the parallel search loop."
                    )

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    futures.remove(fut)
                    try:
                        cur_node = fut.result()
                        logger.info(f"current node count is {completed}, current node.id is {cur_node.id}")
                        consecutive_failures = 0
                    except Exception as e:
                        logger.exception(f"Exception during step_task execution: {e}")
                        cur_node = None
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            raise RuntimeError(
                                "ML-Master failed 3 times in a row during step execution. "
                                "Check your API keys / model settings and inspect the traceback above."
                            ) from e

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node

                        # Per-step grading callback
                        if grading_callback and cur_node is not None:
                            try:
                                grading_callback.on_step_complete(
                                    journal,
                                    completed,
                                    cfg.workspace_dir,
                                    cfg
                                )
                            except Exception as e:
                                logger.warning(f"[Per-step grading] Failed at step {completed}: {e}")

                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                    if completed + len(futures) < total_steps:
                        futures.add(executor.submit(step_task, cur_node))

    # Save per-step grading results
    if grading_callback:
        try:
            grading_callback.save_results()
            logger.info("[Per-step grading] Results saved successfully")
        except Exception as e:
            logger.warning(f"[Per-step grading] Failed to save results: {e}")

    # Post-search selection (copied from AIDE)
    logger.info("=" * 80)
    if getattr(cfg.post_search, "enabled", True):
        logger.info("MCTS search complete. Running post-search selection...")
    else:
        logger.info("MCTS search complete. Post-search selection is disabled.")

    from utils.post_search import select_final_node_with_info
    import json

    final_node = None
    selection_info = {}
    per_method: dict[str, dict] = {}
    if getattr(cfg.post_search, "enabled", True):
        final_node, selection_info = select_final_node_with_info(
            journal,
            selection=cfg.post_search.selection,
            top_k=cfg.post_search.top_k,
            k_std=cfg.post_search.k_std,
            z_threshold=cfg.post_search.z_threshold,
            guard_std=cfg.post_search.guard_std,
            elite_top_k=cfg.post_search.elite_top_k,
            elite_ratio=cfg.post_search.elite_ratio,
            elite_k_std=cfg.post_search.elite_k_std,
            only_good=True,
        )

        # Also compute AIDE-like auxiliary selections in the same run so users can
        # inspect/grade different post-search strategies without rerunning search.
        for method in ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"]:
            node, info = select_final_node_with_info(
                journal,
                selection=method,
                top_k=cfg.post_search.top_k,
                k_std=cfg.post_search.k_std,
                z_threshold=cfg.post_search.z_threshold,
                guard_std=cfg.post_search.guard_std,
                elite_top_k=cfg.post_search.elite_top_k,
                elite_ratio=cfg.post_search.elite_ratio,
                elite_k_std=cfg.post_search.elite_k_std,
                only_good=True,
            )
            per_method[method] = {
                "selected_node_id": getattr(node, "id", None),
                "selected_node_step": getattr(node, "step", None),
                "selected_metric": node.metric.value if node and node.metric else None,
                "selected_cv_mean": getattr(node, "cv_mean", None),
                "selected_cv_std": getattr(node, "cv_std", None),
                "selected_cv_folds": getattr(node, "cv_folds", None),
                "info": info,
            }

    if final_node is not None:
        logger.info(f"Selected final node: {final_node.id} (step {final_node.step})")
        logger.info(f"  Selection strategy: {cfg.post_search.selection}")
        logger.info(f"  Final metric: {final_node.metric.value if final_node.metric else None}")
        if final_node.cv_mean is not None:
            logger.info(f"  CV mean: {final_node.cv_mean:.4f}")
        if final_node.cv_std is not None:
            logger.info(f"  CV std: {final_node.cv_std:.4f}")

        # Save final selection info (AIDE-like structure).
        selection_file = Path(cfg.log_dir) / "final_selection.json"
        selection_data = {
            "post_search": {
                **selection_info,
                "configured_selection": cfg.post_search.selection,
                "selected_node_id": final_node.id,
                "selected_node_step": final_node.step,
                "selected_metric": final_node.metric.value if final_node.metric else None,
                "selected_cv_mean": final_node.cv_mean,
                "selected_cv_std": final_node.cv_std,
                "selected_cv_folds": final_node.cv_folds,
            },
            "methods": per_method,
        }

        with open(selection_file, "w") as f:
            json.dump(selection_data, f, indent=2)
        logger.info(f"Final selection info saved to: {selection_file}")

        # Apply post-search selection to the canonical outputs so downstream tooling can
        # consistently read `best_solution/` and `best_submission/`.
        try:
            selected_submission = cfg.workspace_dir / "submission" / f"submission_{final_node.id}.csv"
            if selected_submission.exists():
                best_submission_dir = cfg.workspace_dir / "best_submission"
                best_submission_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy(selected_submission, best_submission_dir / "submission.csv")
            else:
                logger.warning(
                    f"Selected node submission not found: {selected_submission}. "
                    "Enable `agent.save_all_submission=true` to keep all candidate submissions."
                )

            best_solution_dir = cfg.workspace_dir / "best_solution"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            with open(best_solution_dir / "solution.py", "w") as f:
                f.write(final_node.code)
            with open(best_solution_dir / "node_id.txt", "w") as f:
                f.write(str(final_node.id))

            # Export AIDE-like submission files into the log dir for easy grading.
            def _copy_submission(node_id: str | None, dst_name: str) -> None:
                if not node_id:
                    return
                src = cfg.workspace_dir / "submission" / f"submission_{node_id}.csv"
                if not src.exists():
                    return
                shutil.copy(src, Path(cfg.log_dir) / dst_name)

            # Configured selection ("post_search")
            _copy_submission(final_node.id, "submission_post_search.csv")
            # Baselines/robust selectors
            _copy_submission(per_method.get("best_valid", {}).get("selected_node_id"), "submission_raw.csv")
            _copy_submission(
                per_method.get("mean_minus_k_std", {}).get("selected_node_id"),
                "submission_mean_minus_k_std.csv",
            )
            _copy_submission(per_method.get("maximin", {}).get("selected_node_id"), "submission_max_min.csv")
            _copy_submission(
                per_method.get("elite_maximin", {}).get("selected_node_id"),
                "submission_elite_maximin.csv",
            )
        except Exception as e:
            logger.warning(f"Failed to write post-search selected outputs: {e}", exc_info=True)
    else:
        if getattr(cfg.post_search, "enabled", True):
            # Still write selection diagnostics if available.
            try:
                selection_file = Path(cfg.log_dir) / "final_selection.json"
                with open(selection_file, "w") as f:
                    json.dump(
                        {
                            "post_search": {
                                **selection_info,
                                "configured_selection": cfg.post_search.selection,
                                "selected_node_id": None,
                            },
                            "methods": per_method,
                        },
                        f,
                        indent=2,
                    )
                logger.info(f"Final selection info saved to: {selection_file}")

                # Best-effort export of any method submissions that were found.
                def _copy_submission(node_id: str | None, dst_name: str) -> None:
                    if not node_id:
                        return
                    src = cfg.workspace_dir / "submission" / f"submission_{node_id}.csv"
                    if not src.exists():
                        return
                    shutil.copy(src, Path(cfg.log_dir) / dst_name)

                _copy_submission(per_method.get("best_valid", {}).get("selected_node_id"), "submission_raw.csv")
                _copy_submission(
                    per_method.get("mean_minus_k_std", {}).get("selected_node_id"),
                    "submission_mean_minus_k_std.csv",
                )
                _copy_submission(per_method.get("maximin", {}).get("selected_node_id"), "submission_max_min.csv")
                _copy_submission(
                    per_method.get("elite_maximin", {}).get("selected_node_id"),
                    "submission_elite_maximin.csv",
                )
            except Exception:
                pass
            logger.warning("Post-search selection returned None")
        else:
            logger.info("Skipping post-search selection outputs (disabled).")

    logger.info("=" * 80)

    # Export AIDE-like HTML tree visualization for easier debugging/inspection.
    try:
        from utils.tree_export import generate as generate_tree_html

        out_path = Path(cfg.log_dir) / "tree_plot.html"
        generate_tree_html(cfg, journal, out_path)
        logger.info(f"Tree visualization saved to: {out_path}")
    except Exception as e:
        logger.warning(f"Failed to generate tree visualization HTML: {e}", exc_info=True)

    interpreter.cleanup_session(-1)


if __name__ == "__main__":
    run()
