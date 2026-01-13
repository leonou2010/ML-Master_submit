from __future__ import annotations

"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, Lock
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

from types import SimpleNamespace

def _parse_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    if raw.lower() in ("auto", "max"):
        return -1
    try:
        return int(raw)
    except Exception:
        return default


def _compute_thread_cap(cpu_budget: int) -> int:
    """
    Compute the per-process thread cap for BLAS / OpenMP / ML libraries.

    Defaults to 1 (safe, avoids oversubscription). Override via:
      - MLMASTER_THREAD_CAP=<int>   (or "auto"/"max" to use cpu_budget)
    """
    cpu_budget = max(1, int(cpu_budget))
    requested = _parse_int_env("MLMASTER_THREAD_CAP", default=1)
    if requested in (0, -1):
        return cpu_budget
    return max(1, min(cpu_budget, requested))


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string

logger = logging.getLogger("ml-master")

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None



def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "ml-master/" not in l and "importlib" not in l]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg:str):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg = None
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        
        # self.process: Process = None  # type: ignore
        self.max_parallel_run = cfg.agent.search.parallel_search_num if cfg.agent.search.parallel_search_num else max_parallel_run
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.process: list[Process] = [None] * self.max_parallel_run
        self.current_parallel_run = 0 #
        self.status_map = [0] * self.max_parallel_run
        self.code_inq = [None] * self.max_parallel_run
        self.result_outq = [None] * self.max_parallel_run
        self.event_outq = [None] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id)
        self.cpu_number = int(cfg.cpu_number)
        if self.cpu_number < self.max_parallel_run:
            raise ValueError("The maximum level of parallelism exceeds the number of allocated CPU cores; ensure that each process has at least one CPU core.")
        self._interpreter_mode = os.environ.get("MLMASTER_INTERPRETER_MODE", "auto").strip().lower()
        self._use_multiprocessing = self._interpreter_mode in ("auto", "multiprocessing", "mp", "")
        if self._interpreter_mode in ("subprocess", "popen", "no-mp", "nomp"):
            self._use_multiprocessing = False

        if self._use_multiprocessing:
            try:
                # In some sandboxed environments, multiprocessing primitives fail due to /dev/shm restrictions.
                # Detect and fall back to a subprocess-based interpreter.
                self.lock = Lock()
                _q = Queue()
                try:
                    _q.close()
                except Exception:
                    pass
            except PermissionError as e:
                if self._interpreter_mode in ("multiprocessing", "mp"):
                    raise
                logger.warning(
                    "Multiprocessing primitives are unavailable (%s). Falling back to subprocess interpreter. "
                    "Set MLMASTER_INTERPRETER_MODE=multiprocessing to force multiprocessing (may fail).",
                    e,
                )
                self._use_multiprocessing = False
                self.lock = threading.Lock()
        else:
            self.lock = threading.Lock()

    def check_current_status(self):
        '''
        check current parallel run number
        '''
        if self.current_parallel_run < self.max_parallel_run:
            # logger.info(f"current_parallel_run:{self.current_parallel_run},max parallel run:{self.max_parallel_run}")
            # logger.info(f"check_current_status return True")

            return True
        else:
            # logger.info(f"current_parallel_run:{self.current_parallel_run},max parallel run:{self.max_parallel_run}")
            # logger.info(f"check_current_status return false")

            return False

    # def child_proc_setup(self, result_outq: Queue) -> None:
    #     # disable all warnings (before importing anything)
    #     import shutup

    #     shutup.mute_warnings()
    #     os.chdir(str(self.working_dir))

    #     # this seems to only  benecessary because we're exec'ing code from a string,
    #     # a .py file should be able to import modules from the cwd anyway
    #     sys.path.append(str(self.working_dir))

    #     # capture stdout and stderr
    #     # trunk-ignore(mypy/assignment)
    #     sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def child_proc_setup(self, result_outq: Queue) -> None:
        try:
            verbose = os.environ.get("MLMASTER_CHILD_SETUP_DEBUG", "").strip() not in ("", "0", "false", "False")
            if verbose:
                print("[child_proc_setup] import shutup...")
            import shutup

            if verbose:
                print("[child_proc_setup] mute_warnings...")
            shutup.mute_warnings()

            if verbose:
                print(f"[child_proc_setup] chdir to {self.working_dir}...")
            os.chdir(str(self.working_dir))

            if verbose:
                print("[child_proc_setup] append to sys.path...")
            sys.path.append(str(self.working_dir))

            if verbose:
                print("[child_proc_setup] redirect stdout/stderr...")
            sys.stdout = sys.stderr = RedirectQueue(result_outq)

        except Exception as e:
            import traceback
            result_outq.put("[child_proc_setup error] " + traceback.format_exc())
            raise


    def replace_submission_name(self, code, _id):
        submission_file_name = f"submission_{_id}.csv"
        modified_code = code
        if "submission/submission.csv" in code:
            modified_code = code.replace("submission/submission.csv", f"submission/{submission_file_name}")
        if "/submission.csv" in modified_code:
            modified_code = modified_code.replace("/submission.csv", f"/{submission_file_name}")

        if "to_csv('submission.csv" in modified_code:
            modified_code = modified_code.replace("to_csv('submission.csv", f"to_csv('submission/{submission_file_name}")
        if 'to_csv("submission.csv' in modified_code:
            modified_code = modified_code.replace('to_csv("submission.csv', f'to_csv("submission/{submission_file_name}')

        if '"submission.csv"' in modified_code:
            modified_code = modified_code.replace('"submission.csv"', f'"{submission_file_name}"')
        if "'submission.csv'" in modified_code:
            modified_code = modified_code.replace("'submission.csv'", f"'{submission_file_name}'")
        
        return modified_code
    
    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue, process_id: int
    ) -> None:
        cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
        start_cpu_id_session = self.start_cpu_id + process_id * cpu_number_per_session
        cpu_set = set(range(start_cpu_id_session, start_cpu_id_session + cpu_number_per_session))
        thread_cap = _compute_thread_cap(len(cpu_set))

        # Best-effort safety controls for nested parallelism:
        # - keep joblib/loky from forking more processes
        # - cap BLAS threads to the per-session CPU budget
        # - force some libraries to stay single-threaded
        try:
            os.sched_setaffinity(0, cpu_set)
        except Exception:
            pass

        def _cap_env(name: str, value: int) -> None:
            current = os.environ.get(name)
            if not current:
                os.environ[name] = str(value)
                return
            try:
                if int(current) > value:
                    os.environ[name] = str(value)
            except Exception:
                os.environ[name] = str(value)

        def _force_env(name: str, value: str) -> None:
            os.environ[name] = value

        for env_name in [
            "MLMASTER_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "CATBOOST_THREAD_COUNT",
            "XGBOOST_NUM_THREADS",
            "LIGHTGBM_NUM_THREADS",
        ]:
            _cap_env(env_name, thread_cap)

        # Force single-thread / single-process for known nested-parallel libs.
        _force_env("NUMEXPR_NUM_THREADS", "1")
        _force_env("NUMEXPR_MAX_THREADS", "1")
        _force_env("SKLEARN_N_JOBS", "1")
        _force_env("TOKENIZERS_PARALLELISM", "false")
        _force_env("RAYON_NUM_THREADS", "1")
        _force_env("LOKY_MAX_CPU_COUNT", "1")

        try:
            from threadpoolctl import threadpool_limits as _mlm_threadpool_limits

            _mlm_threadpool_limits(thread_cap)
        except Exception:
            pass

        self.child_proc_setup(result_outq)
        
        while True:
            
            code, id = code_inq.get()
            os.chdir(str(self.working_dir))
            # Run each snippet in a fresh module-like namespace to avoid state leakage
            # (and memory bloat) across executions, while still reusing the same
            # long-lived REPL process for speed.
            global_scope: dict = {"__name__": "__main__"}
            pre_code = (
                "import os\n"
                f"try:\n    os.sched_setaffinity(0, {cpu_set})\nexcept Exception:\n    pass\n"
                f"_MLMASTER_THREAD_CAP = int({thread_cap})\n"
                "_MLMASTER_THREAD_CAP_STR = str(_MLMASTER_THREAD_CAP)\n"
                "def _mlm_cap_env(_name: str) -> None:\n"
                "    _v = os.environ.get(_name)\n"
                "    if not _v:\n"
                "        os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
                "        return\n"
                "    try:\n"
                "        if int(_v) > _MLMASTER_THREAD_CAP:\n"
                "            os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
                "    except Exception:\n"
                "        os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
                "for _k in [\n"
                "    'MLMASTER_NUM_THREADS',\n"
                "    'OMP_NUM_THREADS',\n"
                "    'MKL_NUM_THREADS',\n"
                "    'OPENBLAS_NUM_THREADS',\n"
                "    'BLIS_NUM_THREADS',\n"
                "    'VECLIB_MAXIMUM_THREADS',\n"
                "    'CATBOOST_THREAD_COUNT',\n"
                "    'XGBOOST_NUM_THREADS',\n"
                "    'LIGHTGBM_NUM_THREADS',\n"
                "]:\n"
                "    _mlm_cap_env(_k)\n"
                "del _k\n"
                "def _mlm_force_env(_name: str, _value: str) -> None:\n"
                "    os.environ[_name] = _value\n"
                "_mlm_force_env('NUMEXPR_NUM_THREADS', '1')\n"
                "_mlm_force_env('NUMEXPR_MAX_THREADS', '1')\n"
                "_mlm_force_env('SKLEARN_N_JOBS', '1')\n"
                "_mlm_force_env('TOKENIZERS_PARALLELISM', 'false')\n"
                "_mlm_force_env('RAYON_NUM_THREADS', '1')\n"
                "_mlm_force_env('LOKY_MAX_CPU_COUNT', '1')\n"
                "try:\n"
                "    from threadpoolctl import threadpool_limits as _mlm_threadpool_limits\n"
                "    _mlm_threadpool_limits(_MLMASTER_THREAD_CAP)\n"
                "except Exception:\n"
                "    pass\n"
            )
            
            code  = self.replace_submission_name(code=code, _id=id)
            code = pre_code + code
            with open(self.agent_file_name[process_id], "w") as f:
                f.write(code)
            event_outq.put(("state:ready",))
            try:               
                exec(compile(code, self.agent_file_name[process_id], "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name[process_id],
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name[process_id])

            try:
                import gc

                gc.collect()
            except Exception:
                pass

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self,process_id) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        logger.debug("create_process(process_id=%s)", process_id)
        self.code_inq[process_id], self.result_outq[process_id], self.event_outq[process_id] = Queue(), Queue(), Queue()
        self.process[process_id] = Process(
            target=self._run_session,
            args=(self.code_inq[process_id], self.result_outq[process_id], self.event_outq[process_id],process_id),
        )
        self.process[process_id].start()

    def _infer_exc_type_from_output(self, output: str) -> str | None:
        if not output:
            return None
        for line in reversed(output.splitlines()):
            s = line.strip()
            if not s:
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?::|$)", s)
            if not m:
                continue
            name = m.group(1)
            if name == "KeyboardInterrupt":
                return "TimeoutError"
            if name.endswith("Error") or name.endswith("Exception"):
                return name
        return None

    def _run_subprocess(self, code: str, id, process_id: int) -> ExecutionResult:
        start_time = time.time()

        cpu_number_per_session = int(self.cpu_number / self.max_parallel_run)
        start_cpu_id_session = self.start_cpu_id + process_id * cpu_number_per_session
        cpu_set = set(range(start_cpu_id_session, start_cpu_id_session + cpu_number_per_session))

        thread_cap = _compute_thread_cap(len(cpu_set))
        pre_code = (
            "import os\n"
            f"try:\n    os.sched_setaffinity(0, {cpu_set})\n"
            "except Exception:\n    pass\n"
            f"_MLMASTER_THREAD_CAP = int({thread_cap})\n"
            "_MLMASTER_THREAD_CAP_STR = str(_MLMASTER_THREAD_CAP)\n"
            "def _mlm_cap_env(_name: str) -> None:\n"
            "    _v = os.environ.get(_name)\n"
            "    if not _v:\n"
            "        os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
            "        return\n"
            "    try:\n"
            "        if int(_v) > _MLMASTER_THREAD_CAP:\n"
            "            os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
            "    except Exception:\n"
            "        os.environ[_name] = _MLMASTER_THREAD_CAP_STR\n"
            "for _k in [\n"
            "    'MLMASTER_NUM_THREADS',\n"
            "    'OMP_NUM_THREADS',\n"
            "    'MKL_NUM_THREADS',\n"
            "    'OPENBLAS_NUM_THREADS',\n"
            "    'BLIS_NUM_THREADS',\n"
            "    'VECLIB_MAXIMUM_THREADS',\n"
            "    'CATBOOST_THREAD_COUNT',\n"
            "    'XGBOOST_NUM_THREADS',\n"
            "    'LIGHTGBM_NUM_THREADS',\n"
            "]:\n"
            "    _mlm_cap_env(_k)\n"
            "del _k\n"
            "def _mlm_force_env(_name: str, _value: str) -> None:\n"
            "    os.environ[_name] = _value\n"
            "_mlm_force_env('NUMEXPR_NUM_THREADS', '1')\n"
            "_mlm_force_env('NUMEXPR_MAX_THREADS', '1')\n"
            "_mlm_force_env('SKLEARN_N_JOBS', '1')\n"
            "_mlm_force_env('TOKENIZERS_PARALLELISM', 'false')\n"
            "_mlm_force_env('RAYON_NUM_THREADS', '1')\n"
            "_mlm_force_env('LOKY_MAX_CPU_COUNT', '1')\n"
            "try:\n"
            "    from threadpoolctl import threadpool_limits as _mlm_threadpool_limits\n"
            "    _mlm_threadpool_limits(_MLMASTER_THREAD_CAP)\n"
            "except Exception:\n"
            "    pass\n"
        )

        code = self.replace_submission_name(code=code, _id=id)
        code = pre_code + code

        runfile_name = self.agent_file_name[process_id]
        runfile_path = self.working_dir / runfile_name
        runfile_path.write_text(code)

        proc = subprocess.Popen(
            [sys.executable, runfile_name],
            cwd=str(self.working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        timed_out = False
        try:
            if self.timeout is None:
                out, _ = proc.communicate()
            else:
                out, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            out, _ = proc.communicate()

        exec_time = time.time() - start_time

        term_out: list[str] = []
        if out:
            term_out.extend(out.splitlines(True))

        exc_type: str | None = None
        exc_info: dict | None = None
        exc_stack: list[tuple] | None = None

        if timed_out:
            exc_type = "TimeoutError"
            if self.timeout is not None:
                exec_time = float(self.timeout)
            term_out.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            if proc.returncode not in (0, None):
                exc_type = self._infer_exc_type_from_output(out) or "RuntimeError"
            term_out.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        try:
            runfile_path.unlink()
        except Exception:
            pass

        return ExecutionResult(term_out, exec_time, exc_type, exc_info, exc_stack)

    def cleanup_session(self, process_id: int, *, close_queues: bool = True):
        if not self._use_multiprocessing:
            return

        def _close_queue(q) -> None:
            if q is None:
                return
            # `multiprocessing.Queue.join_thread()` can block indefinitely if the
            # feeder thread is wedged (e.g., child process crash / forced kill).
            # We prefer a best-effort cleanup that never hangs the main loop.
            try:
                cancel = getattr(q, "cancel_join_thread", None)
                if callable(cancel):
                    cancel()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

        if process_id == -1 : #clean all process
            for pid in range(self.max_parallel_run):
                if self.process[pid] is None:
                    if close_queues:
                        _close_queue(self.code_inq[pid])
                        _close_queue(self.result_outq[pid])
                        _close_queue(self.event_outq[pid])
                        self.code_inq[pid] = None
                        self.result_outq[pid] = None
                        self.event_outq[pid] = None
                    continue
                # give the child process a chance to terminate gracefully
                self.process[pid].terminate()
                self.process[pid].join(timeout=2)
                # kill the child process if it's still alive
                if self.process[pid].exitcode is None:
                    logger.warning("Child process failed to terminate gracefully, killing it..")
                    self.process[pid].kill()
                    self.process[pid].join()
                # don't wait for gc, clean up immediately
                self.process[pid].close()
                self.process[pid] = None  # type: ignore
                if close_queues:
                    _close_queue(self.code_inq[pid])
                    _close_queue(self.result_outq[pid])
                    _close_queue(self.event_outq[pid])
                    self.code_inq[pid] = None
                    self.result_outq[pid] = None
                    self.event_outq[pid] = None
        else: #clean given process
            if self.process[process_id] is None:
                if close_queues:
                    _close_queue(self.code_inq[process_id])
                    _close_queue(self.result_outq[process_id])
                    _close_queue(self.event_outq[process_id])
                    self.code_inq[process_id] = None
                    self.result_outq[process_id] = None
                    self.event_outq[process_id] = None
                return
            # give the child process a chance to terminate gracefully
            self.process[process_id].terminate()
            self.process[process_id].join(timeout=2)
            # kill the child process if it's still alive
            if self.process[process_id].exitcode is None:
                logger.warning("Child process failed to terminate gracefully, killing it..")
                self.process[process_id].kill()
                self.process[process_id].join()
            # don't wait for gc, clean up immediately
            self.process[process_id].close()
            self.process[process_id] = None  # type: ignore
            if close_queues:
                _close_queue(self.code_inq[process_id])
                _close_queue(self.result_outq[process_id])
                _close_queue(self.event_outq[process_id])
                self.code_inq[process_id] = None
                self.result_outq[process_id] = None
                self.event_outq[process_id] = None

    def run(self, code: str, id, reset_session=True):
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """
        logger.info(f"REPL is executing code (reset_session={reset_session})")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None
        reuse_process = os.environ.get("MLMASTER_REPL_REUSE_PROCESS", "1") != "0"
        try:
            # assign process_id
            with self.lock:
                for idx in range(self.max_parallel_run):
                    if self.status_map[idx] == 0:
                        self.status_map[idx] = 1  # signals occupied
                        process_id = idx
                        self.current_parallel_run += 1
                        logger.info(f"has assigned process_id，process_id is {process_id}")
                        break
                if process_id is None:
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")

            if not self._use_multiprocessing:
                return self._run_subprocess(code=code, id=id, process_id=process_id)
		        
            # Prefer reusing the child REPL process for speed. We already execute each
            # snippet in a fresh namespace inside `_run_session`, so we don't need to
            # tear down the process between runs unless explicitly requested.
            if self.process[process_id] is None or (
                hasattr(self.process[process_id], "is_alive") and not self.process[process_id].is_alive()
            ):
                self.create_process(process_id=process_id)
            elif reset_session and not reuse_process:
                try:
                    self.cleanup_session(process_id=process_id)
                except Exception:
                    logger.warning("Failed to cleanup_session during reset; continuing.")
                self.create_process(process_id=process_id)

            assert self.process[process_id].is_alive()

            self.code_inq[process_id].put((code, id))
            # wait for child to actually start execution (we don't want interrupt child setup)
            try:
                state = self.event_outq[process_id].get(timeout=10)
            except queue.Empty:
                msg = "REPL child process failed to start execution"
                logger.critical(msg)
                queue_dump = ""
                while not self.result_outq[process_id].empty():
                    queue_dump = self.result_outq[process_id].get()
                    logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
                return ExecutionResult(  # Considered as running timeout
                    term_out=[msg, queue_dump],
                    exec_time=0,
                    exc_type="RuntimeError",
                    exc_info={},
                    exc_stack=[],
                )

            assert state[0] == "state:ready", state
            start_time = time.time()

            # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
            # if the child process dies without this flag being set, it's an unexpected termination
            child_in_overtime = False
            while True:
                try:
                    # check if the child is done
                    state = self.event_outq[process_id].get(timeout=1)  # wait for state:finished
                    assert state[0] == "state:finished", state
                    exec_time = time.time() - start_time
                    break
                except queue.Empty:
                    # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                    if not child_in_overtime and not self.process[process_id].is_alive():
                        msg = "REPL child process died unexpectedly"
                        logger.critical(msg)
                        queue_dump = ""
                        while not self.result_outq[process_id].empty():
                            queue_dump = self.result_outq[process_id].get()
                            logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
                        return ExecutionResult(
                            term_out=[msg, queue_dump],
                            exec_time=0,
                            exc_type="RuntimeError",
                            exc_info={},
                            exc_stack=[],
                        )

                    # child is alive and still executing -> check if we should sigint..
                    if self.timeout is None:
                        continue
                    running_time = time.time() - start_time
                    if running_time > self.timeout:

                        # [TODO] handle this in a better way
                        assert reset_session, "Timeout ocurred in interactive session"

                        # send interrupt to child
                        os.kill(self.process[process_id].pid, signal.SIGINT)  # type: ignore
                        child_in_overtime = True
                        # terminate if we're overtime by more than a minute
                        if running_time > self.timeout + 60:
                            logger.warning("Child failed to terminate, killing it..")
                            # Keep queues open so we can still drain any buffered output.
                            self.cleanup_session(process_id=process_id, close_queues=False)

                            state = (None, "TimeoutError", {}, [])
                            exec_time = self.timeout
                            break

            output: list[str] = []
            # Read stdout/stderr from child up to the EOF marker.
            # IMPORTANT: never block forever if EOF is missing (child crash / forced kill).
            saw_eof = False
            drain_start = time.time()
            drain_timeout_s = int(
                os.environ.get("MLMASTER_REPL_DRAIN_TIMEOUT_S", "15") or "15"
            )
            while True:
                try:
                    res = self.result_outq[process_id].get(timeout=1)
                    output.append(res)
                    if res == "<|EOF|>":
                        saw_eof = True
                        break
                except queue.Empty:
                    if not self.process[process_id].is_alive():
                        break
                    if time.time() - drain_start > drain_timeout_s:
                        break

            if saw_eof:
                output.pop()  # remove the EOF marker
            else:
                logger.warning(
                    "REPL output missing EOF marker (process_id=%s). Drained %d chunks before continuing.",
                    process_id,
                    len(output),
                )

            e_cls_name, exc_info, exc_stack = state[1:]

            if e_cls_name == "TimeoutError":
                output.append(
                    f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                )
            logger.debug(
                "execution done (process_id=%s, exec_time=%.3fs, exc_type=%s)",
                process_id,
                exec_time,
                e_cls_name,
            )
            return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
        finally:
            if process_id is not None:
                try:
                    if self._use_multiprocessing and not reuse_process:
                        self.cleanup_session(process_id=process_id)
                except Exception:
                    pass
                try:
                    with self.lock:
                        self.current_parallel_run -= 1
                        self.status_map[process_id] = 0
                except Exception:
                    pass

if __name__ == "__main__":
    cfg = SimpleNamespace()
    cfg.start_cpu_id = 0
    cfg.cpu_number = 10
    cfg.agent = SimpleNamespace()
    cfg.agent.search = SimpleNamespace()
    cfg.agent.search.parallel_search_num = 2

    # create tmo directory
    working_dir = f"./test"

    # create Interpreter
    interpreter = Interpreter(
        working_dir=working_dir,
        cfg=cfg
    )

    # test code
    test_code = """import os
import json
import tarfile
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data import Mixup, create_transform
from timm.models import efficientnet_b4
from torch.optim import AdamW
from timm.utils import ModelEma
from PIL import Image

# Configuration
BATCH_SIZE = 64
IMG_SIZE = 380
NUM_CLASSES = 1010
NUM_EPOCHS = 5  # Reduced from 15 to fit time constraints
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Improved extraction handling both gzipped and plain tar
def extract_tar_file(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        print(f"Extracting {tar_path} to {extract_path}...")
        try:
            # First try gzip compression
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
        except tarfile.ReadError:
            print("Failed as gzip, trying plain tar...")
            # Fallback to plain tar extraction
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_path)


# Data preparation
class INaturalistDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None, split="train"):
        self.root_dir = root_dir
        self.transform = transform
        with open(json_path) as f:
            data = json.load(f)

        if split in ["train", "val"]:
            categories = data["categories"]
            self.cat_id_to_index = {
                cat["id"]: idx for idx, cat in enumerate(categories)
            }
            self.image_data = {img["id"]: img for img in data["images"]}
            self.annotations = {
                ann["image_id"]: ann["category_id"] for ann in data["annotations"]
            }
            self.ids = list(self.annotations.keys())
        else:  # test
            self.image_data = {img["id"]: img for img in data["images"]}
            self.ids = list(self.image_data.keys())
        self.split = split

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_data[img_id]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.split in ["train", "val"]:
            label = self.annotations[img_id]
            label_index = self.cat_id_to_index[label]
            return image, label_index
        return image, img_id


# Augmentation strategy
def get_transform(train):
    if train:
        return create_transform(
            input_size=IMG_SIZE,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            re_prob=0.25,
            re_mode="pixel",
            interpolation="bicubic",
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# Create model
def create_model():
    model = efficientnet_b4(pretrained=True, num_classes=NUM_CLASSES)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.classifier.in_features, NUM_CLASSES)
    )
    return model.to(DEVICE)


# Training function
def train_model():
    # Extract datasets with improved extraction
    if not os.path.exists("input/train_val2019"):
        extract_tar_file("input/train_val2019.tar.gz", "input/train_val2019")
    if not os.path.exists("input/test2019"):
        extract_tar_file("input/test2019.tar.gz", "input/test2019")

    # Prepare data
    train_data = INaturalistDataset(
        root_dir="input/train_val2019",
        json_path="input/train2019.json",
        transform=get_transform(True),
        split="train",
    )
    val_data = INaturalistDataset(
        root_dir="input/train_val2019",
        json_path="input/val2019.json",
        transform=get_transform(False),
        split="val",
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Compute class weights for loss
    with open("input/train2019.json") as f:
        train_json = json.load(f)
    categories = train_json["categories"]
    cat_id_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    train_annotations = train_json["annotations"]
    train_labels = [cat_id_to_index[ann["category_id"]] for ann in train_annotations]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize model and optimizer
    model = create_model()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ema_model = ModelEma(model, decay=0.999)

    # Mixup augmentation
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode="batch"
    )

    # Training loop
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ema_model.update(model)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = ema_model.module(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(ema_model.module.state_dict(), "working/best_model.pth")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return ema_model.module, best_acc


# Inference and submission
def create_submission(model):
    test_data = INaturalistDataset(
        root_dir="input/test2019",
        json_path="input/test2019.json",
        transform=get_transform(False),
        split="test",
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Create index to category ID mapping
    with open("input/train2019.json") as f:
        train_json = json.load(f)
    categories = train_json["categories"]
    index_to_cat_id = [cat["id"] for cat in categories]

    model.eval()
    predictions = []
    ids = []
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            topk_indices = torch.topk(outputs, k=5, dim=1)[1].cpu().numpy()
            for img_id, indices in zip(img_ids, topk_indices):
                pred_cat_ids = [index_to_cat_id[idx] for idx in indices]
                ids.append(img_id)
                predictions.append(" ".join(map(str, pred_cat_ids)))

    # Create submission
    submission = pd.DataFrame({"id": ids, "predicted": predictions})
    os.makedirs("submission", exist_ok=True)
    submission.to_csv("submission/submission.csv", index=False)


# Main execution
if __name__ == "__main__":
    # Train and evaluate
    model, val_acc = train_model()
    print(f"Validation Accuracy: {val_acc}")

    # Create submission
    create_submission(model)
    print("Submission file created at submission/submission.csv")
    """
    result = interpreter.replace_submission_name(test_code, _id="12345")
    print(result)
    
