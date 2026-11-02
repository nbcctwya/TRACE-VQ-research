"""Watch the SP500 supervisor independently; recover a lost supervisor twice.

Scientific errors stop the queue for review. Only a vanished supervisor is
restarted, after its child exits, retaining completed phases and checkpoints.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts/research/sp500_candidates_multiseed_20261031"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def recover_completed_phase(root, state):
    name = state.get("current_phase")
    entry = state.get("phases", {}).get(name)
    if not entry or entry.get("status") == "complete":
        return
    markers = {"stage1_fsq": root / "stage1_fsq/training_result.json",
               "cache_fsq": root / "cache/fsq/completion.json",
               "stage1_preflight": root / "stage1_gpu_check.json",
               "stage2_preflight": root / "stage2_gpu_checks.json",
               "integrity_before": root / "integrity_before.json",
               "integrity_after": root / "integrity_after.json"}
    pieces = name.split(".")
    if len(pieces) == 3 and pieces[0] in ("fsq", "ranknet", "fsq_ranknet"):
        folder = root / pieces[0] / pieces[1]
        markers[name] = folder / {"train": "training_result.json", "evaluate": "evaluation/completion.json",
                                  "backtest": "backtest/backtest_config.json"}[pieces[2]]
    complete = False
    if name == "tests":
        log = root / "all_tests.log"
        complete = log.exists() and "\nOK\n" in log.read_text()
    elif name in markers:
        record = read(markers[name])
        if record:
            complete = record.get("status") in ("complete", "passed") or record.get("source_unchanged") is True
            if name.endswith(".backtest"):
                complete = (markers[name].parent / "portfolio_return.csv").is_file()
            if complete and "checkpoint" in record:
                checkpoint = Path(record["checkpoint"])
                complete = checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["checkpoint_sha256"]
    if complete:
        entry.update(status="complete", returncode=0, finished=now(), recovered_after_supervisor_exit=True)
        state["updated"] = now()
        write(root / "progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("Research output required")
    lock = (root / "watchdog.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / "watchdog_status.json"
    previous = read(path, {})
    restarts = previous.get("restarts", [])
    started = now()
    while True:
        state = read(root / "progress.json", {})
        monitor = read(root / "monitor_status.json", {})
        parent_alive = alive(state.get("pid"))
        entry = state.get("phases", {}).get(state.get("current_phase"), {})
        child_alive = alive(entry.get("pid"))
        status = dict(updated=now(), started=started, watchdog_pid=os.getpid(), status="monitoring",
                      supervisor_pid=state.get("pid"), supervisor_alive=parent_alive,
                      child_pid=entry.get("pid"), child_alive=child_alive,
                      pipeline_status=state.get("status"), current_phase=state.get("current_phase"),
                      completed_candidate_runs=len(state.get("runs", {})), restarts=restarts,
                      heartbeat_age_seconds=monitor.get("heartbeat_age_seconds"),
                      stalled_warning=monitor.get("stalled_warning", False))
        if state.get("status") in ("complete", "failed"):
            status["status"] = state["status"]
            status["error"] = state.get("error")
            write(path, status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return
        if not parent_alive and state and not child_alive:
            if len(restarts) >= 2:
                status.update(status="needs_review", error="Supervisor recovery limit reached")
                write(path, status)
                return
            recover_completed_phase(root, state)
            with (root / "launch.log").open("a") as log:
                process = subprocess.Popen([str(ROOT / "research/run.sh"), "research/run_sp500_candidates.py",
                                            "--root", str(root)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            restarts.append(dict(time=now(), old_pid=state.get("pid"), new_pid=process.pid))
            status.update(status="restarting_supervisor", supervisor_pid=process.pid, restarts=restarts)
            print(json.dumps(status, ensure_ascii=False), flush=True)
        write(path, status)
        time.sleep(30)


if __name__ == "__main__":
    main()
