"""configuration and setup utils"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Hashable, cast, Literal

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging
import shutil

from search.journal import Journal, filter_journal

from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("ml-master")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str

@dataclass
class LinearDecay:
    alpha: float

@dataclass
class ExponentialDecay:
    gamma: float

@dataclass
class PiecewiseDecay:
    alpha: float
    phase_ratios: list

@dataclass
class DynamicPiecewiseDecay:
    alpha: float
    phase_ratios: list

@dataclass
class DecayConfig:
    decay_type: str
    exploration_constant: float
    lower_bound: float
    linear_decay: LinearDecay
    exponential_decay: ExponentialDecay
    piecewise_decay: PiecewiseDecay
    dynamic_piecewise_decay: DynamicPiecewiseDecay
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    invalid_metric_upper_bound: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    max_improve_failure: int
    parallel_search_num: int
    # Bug consultant (RAG + RL + Summarization debugging)
    use_bug_consultant: bool = False
    max_bug_records: int = 500
    advice_budget_chars: int = 200000
    max_active_bugs: int = 200
    max_trials_per_bug: int = 20
    delete_pruned_bug_files: bool = False
    bug_context_mode: str = "consultant"  # consultant | buggy_code | both
    bug_context_count: int = 0

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool
    check_format: bool
    save_all_submission: bool
    steerable_reasoning: bool
    code: StageConfig
    feedback: StageConfig
    search: SearchConfig
    decay: DecayConfig

@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class PostSearchConfig:
    """
    Configuration for selecting final solution after MCTS search.
    Copied from AIDE to test robust selection strategies.
    """
    enabled: bool = True
    selection: str = "best_valid"  # best_valid | maximin | elite_maximin | mean_minus_k_std
    top_k: int = 20
    k_std: float = 2.0
    z_threshold: float = 2.0
    guard_std: float = 2.0
    elite_top_k: int = 3
    elite_ratio: float = 0.05
    elite_k_std: float = 2.0


@dataclass
class PlanConstraintsConfig:
    """
    Configuration for plan/sketch constraints.
    Copied from AIDE to test if constraining plan length improves solution quality.
    Note: Constraints are enforced via prompting only, not via truncation.
    """
    enabled: bool = False
    max_sentences: int = 5


@dataclass
class PerStepGradingConfig:
    """
    Configuration for per-step grading (generalization gap experiments).
    Grades all selection methods at each MCTS step using MLE-bench ground truth.
    """
    enabled: bool = False
    mlebench_data_dir: str = "/home/ka3094/mle-bench/data/competitions"
    methods: list[str] = field(
        default_factory=lambda: [
            "best_valid",
            "mean_minus_k_std",
            "maximin",
            "elite_maximin",
        ]
    )
    grade_every_n_steps: int = 1
    # Write `per_step_grading/grading_history.*` incrementally during the run.
    save_every_n_steps: int = 1


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path | None
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig
    post_search: PostSearchConfig
    plan_constraints: PlanConstraintsConfig
    per_step_grading: PerStepGradingConfig
    start_cpu_id: str
    cpu_number: str
    competition_id: str | None = None
    # Whether to delete any existing workspace dir for this exp_name before preparing input/working/submission.
    reset_workspace: bool = True
    # If true, load and continue from an existing journal under `log_dir/<exp_name>/journal.json`.
    resume: bool = False


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config_mcts.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg

def load_cfg(path: Path = Path(__file__).parent / "config_mcts.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    # `dataset_dir` is only required for components that need access to MLE-Bench private data
    # (e.g. `grading_server.py`). Keep it optional for standalone runs.
    if getattr(cfg, "dataset_dir", None) is not None:
        cfg.dataset_dir = Path(cfg.dataset_dir).resolve()

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # Resume mode: keep the existing workspace directory if present.
    if getattr(cfg, "resume", False):
        cfg.reset_workspace = False

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    if getattr(cfg, "reset_workspace", True) and cfg.workspace_dir.exists():
        shutil.rmtree(cfg.workspace_dir)

    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Save journal for crash-resume support.
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # Also save a JSONL view that is convenient for grepping / external tooling.
    try:
        with open(cfg.log_dir / "journal.jsonl", "w") as f:
            for n in journal.nodes:
                metric = getattr(n, "metric", None)
                record = {
                    "id": getattr(n, "id", None),
                    "step": getattr(n, "step", None),
                    "stage": getattr(n, "stage", None),
                    "parent": getattr(getattr(n, "parent", None), "id", None),
                    "children": sorted([c.id for c in getattr(n, "children", set())]),
                    "is_buggy": getattr(n, "is_buggy", None),
                    "is_valid": getattr(n, "is_valid", None),
                    "metric": metric.to_dict() if metric is not None else None,
                    "cv_mean": getattr(n, "cv_mean", None),
                    "cv_std": getattr(n, "cv_std", None),
                    "cv_folds": getattr(n, "cv_folds", None),
                }
                f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write journal.jsonl: {e}", exc_info=True)
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config_mcts.yaml")

    # create/update the tree + code visualization (AIDE-like) for easier inspection during runs
    try:
        from utils.tree_export import generate as generate_tree_html

        tree_path = cfg.log_dir / "tree_plot.html"
        existed_before = tree_path.exists()
        generate_tree_html(cfg, journal, tree_path)
        if not existed_before and tree_path.exists():
            logger.info(f"Tree visualization saved to: {tree_path}")
    except Exception as e:
        logger.warning(f"Failed to generate tree visualization HTML: {e}", exc_info=True)

    # Export per-node artifacts (AIDE-like) for easier inspection/grading.
    try:
        solutions_dir = cfg.log_dir / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: avoid O(steps^2) rewrites by only exporting NEW nodes.
        state_path = solutions_dir / "export_state.json"
        last_exported_idx = -1
        try:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                last_exported_idx = int(state.get("last_exported_idx", -1))
        except Exception:
            last_exported_idx = -1

        if last_exported_idx >= len(journal.nodes):
            # Journal was reset/overwritten; start fresh.
            last_exported_idx = -1

        start_idx = max(0, last_exported_idx + 1)
        append_rows: list[dict] = []

        for idx in range(start_idx, len(journal.nodes)):
            n = journal.nodes[idx]

            # Save generated code once.
            node_path = solutions_dir / f"node_{idx}.py"
            if not node_path.exists():
                node_path.write_text(
                    getattr(n, "code", "") or "", encoding="utf-8", errors="replace"
                )

            # Copy per-node submission once (if it exists).
            node_id = getattr(n, "id", None)
            if node_id:
                submission_src = cfg.workspace_dir / "submission" / f"submission_{node_id}.csv"
                submission_dst = solutions_dir / f"submission_node_{idx}.csv"
                if submission_src.is_file() and not submission_dst.exists():
                    shutil.copy2(submission_src, submission_dst)

            metric = getattr(n, "metric", None)
            append_rows.append(
                {
                    "idx": idx,
                    "id": node_id,
                    "step": getattr(n, "step", None),
                    "stage": getattr(n, "stage", None),
                    "is_buggy": getattr(n, "is_buggy", None),
                    "metric": getattr(metric, "value", None),
                    "maximize": getattr(metric, "maximize", None),
                    "cv_mean": getattr(n, "cv_mean", None),
                    "cv_std": getattr(n, "cv_std", None),
                }
            )

        if append_rows:
            jsonl_path = solutions_dir / "metrics.jsonl"
            jsonl_mode = "a" if start_idx > 0 and jsonl_path.exists() else "w"
            with open(jsonl_path, jsonl_mode, encoding="utf-8") as f:
                for row in append_rows:
                    f.write(json.dumps(row, default=str) + "\n")

            # Optional convenience export (AIDE produces both JSONL and CSV).
            try:
                import csv as _csv

                fieldnames = [
                    "idx",
                    "id",
                    "step",
                    "stage",
                    "is_buggy",
                    "metric",
                    "maximize",
                    "cv_mean",
                    "cv_std",
                ]
                csv_path = solutions_dir / "metrics.csv"
                csv_mode = "a" if start_idx > 0 and csv_path.exists() else "w"
                write_header = csv_mode == "w"
                with open(csv_path, csv_mode, newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    for row in append_rows:
                        writer.writerow({k: row.get(k) for k in fieldnames})
            except Exception:
                pass

        # Persist state for incremental exports (best-effort; not required for correctness).
        try:
            state_path.write_text(
                json.dumps({"last_exported_idx": len(journal.nodes) - 1}),
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed to export solutions/: {e}", exc_info=True)
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
    # # concatenate logs
    # with open(cfg.log_dir / "full_log.txt", "w") as f:
    #     f.write(
    #         concat_logs(
    #             cfg.log_dir / "ml-master.log",
    #             cfg.workspace_dir / "best_solution" / "node_id.txt",
    #             cfg.log_dir / "filtered_journal.json",
    #         )
    #     )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the ml-master run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."
