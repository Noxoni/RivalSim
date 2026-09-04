"""One requested accepted-boundary pause/evaluation/resume; no trainer edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("G:/dev/RivalSim-runs/ssl-foundation-v5-long-trace-v1")
RESULTS = ROOT / "results/rival2/ssl_foundation_v5_long_trace_v1"
RUNNER = ROOT / "benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py"
TARGET = 20
OWNER = "user-requested-long-trace-evaluation-20-20260904"
STOP = RUN / "STOP_REQUESTED"
CONTROL = RUN / "evaluation_u0020_control.json"
CANCEL = RUN / "CANCEL_EVALUATION_U0020"
REQUEST = RESULTS / "evaluation_u0020_request.json"
AUTHORITY_SHA = "BE82C618296A6124858BFCCE99EB66E779074B524108476690D2E227F6B8CB4C"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def records():
    text = (RESULTS / "training_curve.jsonl").read_text()
    # The writer appends one line. An in-progress last line is not a boundary.
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith("\n")]


def in_update(stack_dump):
    return any(
        frame["name"] in {"collect_rollout", "recurrent_ppo_update"}
        and "rivalsim" in frame["filename"].lower()
        for thread in stack_dump
        if thread.get("thread_name") == "MainThread"
        for frame in thread["frames"]
    )


def should_request_pause(before, stack_dump, after):
    # Observed inside update20, with19 still the latest completed boundary.
    # Never infer in-flight update number from an arbitrary sleep duration.
    return before == after == TARGET - 1 and in_update(stack_dump)


def comparison(baseline, current):
    if (
        baseline["accepted_updates"] != 10
        or current["accepted_updates"] != TARGET
        or any(baseline[k] != current[k] for k in ("worlds", "ticks", "deterministic_policy"))
    ):
        raise ValueError("evaluation protocol or boundary mismatch")
    result = {}
    for opponent, measured in current["opponents"].items():
        before = baseline["opponents"][opponent]
        keys = (
            "goals_for",
            "goals_against",
            "touches_per_minute",
            "no_touch_resets",
            "goalward_touch_fraction",
            "mean_speed_uu_per_second",
        )
        result[opponent] = {
            key: {
                "update10": before[key],
                "update20": measured[key],
                "delta": measured[key] - before[key],
            }
            for key in keys
        }
        old_net, new_net = (
            before["goals_for"] - before["goals_against"],
            (measured["goals_for"] - measured["goals_against"]),
        )
        result[opponent]["goal_differential"] = {
            "update10": old_net,
            "update20": new_net,
            "delta": new_net - old_net,
        }
    return result


def run(args):
    if CONTROL.exists() or STOP.exists() or CANCEL.exists():
        raise RuntimeError("control/stop/cancel already exists; inspect rather than duplicate")
    if sha(RESULTS / "authority.json") != AUTHORITY_SHA:
        raise RuntimeError("campaign authority changed")
    request = json.loads(REQUEST.read_text())
    if (
        request["target_update"] != TARGET
        or request["owner"] != OWNER
        or request.get("resume_authorized") is not True
    ):
        raise RuntimeError("request identity mismatch")
    worker = psutil.Process(args.worker_pid)
    if RUNNER.name not in " ".join(worker.cmdline()):
        raise RuntimeError("worker identity mismatch")
    deadline = json.loads((RUN / "campaign_state.json").read_text())["deadline_unix"]
    status = {
        "owner": OWNER,
        "controller_pid": psutil.Process().pid,
        "worker_pid": worker.pid,
        "worker_created": worker.create_time(),
        "target_update": TARGET,
        "status": "waiting_for_update20",
        "request_sha256": sha(REQUEST),
        "original_deadline_unix": deadline,
    }
    write(CONTROL, status)
    while time.time() < deadline:
        if CANCEL.exists() or STOP.exists() or not worker.is_running():
            raise RuntimeError("cancelled, stopped, or worker exited before requested boundary")
        latest = records()[-1]["accepted_update"]
        if latest >= TARGET:
            raise RuntimeError("target passed without pause; never label another update20")
        if latest == TARGET - 1:
            probe = subprocess.run(
                [args.py_spy, "dump", "--pid", str(worker.pid), "--json", "--nonblocking"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            try:
                dump = json.loads(probe.stdout) if probe.returncode == 0 else []
            except json.JSONDecodeError:
                dump = []
            after = records()[-1]["accepted_update"]
            if should_request_pause(latest, dump, after):
                with STOP.open("x") as stream:
                    json.dump({"owner": OWNER, "target_update": TARGET}, stream)
                status.update(status="pause_requested_inside_update20", latest_completed=after)
                write(CONTROL, status)
                break
        time.sleep(1)
    else:
        raise RuntimeError("original deadline elapsed")
    while worker.is_running():
        if CANCEL.exists() or time.time() >= deadline:
            raise RuntimeError("cancelled or deadline elapsed; leave stopped, do not resume")
        time.sleep(1)
    summary = json.loads((RESULTS / "training_summary.json").read_text())
    if (
        summary["accepted_updates"] != TARGET
        or summary["hard_failure"] is not None
        or summary["stop_reason"] != "user_requested_stop_at_accepted_boundary"
    ):
        raise RuntimeError("exact update20 pause/evaluation did not complete safely")
    evaluation = summary["last_evaluation"]
    manifest = json.loads((RESULTS / "snapshot_manifest.json").read_text())
    baseline = next(e for e in manifest["evaluations"] if e["accepted_updates"] == 10)
    final = Path(summary["final_checkpoint"]["path"])
    snapshot = ROOT / "checkpoints/rival2/ssl_foundation_v5_long_trace_v1/evaluation_u0020.pt"
    if snapshot.exists():
        raise RuntimeError("preserve existing update20 snapshot")
    shutil.copy2(final, snapshot)
    if sha(snapshot) != summary["final_checkpoint"]["sha256"]:
        raise RuntimeError("saved checkpoint hash mismatch")
    extra = {**summary["final_checkpoint"], "path": str(snapshot)}
    manifest["snapshots"].append(extra)
    manifest["snapshots"].sort(key=lambda item: item["accepted_updates"])
    write(RESULTS / "snapshot_manifest.json", manifest)
    train_rows = [r for r in records() if 11 <= r["accepted_update"] <= TARGET]
    report = {
        "owner": OWNER,
        "checkpoint": extra,
        "evaluation": evaluation,
        "comparison": comparison(baseline, evaluation),
        "baseline_update": 10,
        "accepted_updates_since_baseline": len(train_rows),
        "new_trainable_samples": sum(r["rollout"]["trainable_agent_samples"] for r in train_rows),
        "new_physics_ticks": len(train_rows) * 32768 * 360,
        "training_wall_seconds": sum(r["wall_seconds"] for r in train_rows),
        "physics_exposure_ratio_per_update_vs_old": 360 / 128,
        "interpretation": "matched protocol, sequential comparison; not a randomized causal test",
        "reward_optimizer_and_learning_contract_changed": False,
        "resume_semantics": (
            "exact saved weights/Adam/RNG; fresh simulator episodes and hidden state"
        ),
    }
    write(RESULTS / "evaluation_u0020_comparison.json", report)
    status.update(status="evaluation_complete", checkpoint=extra)
    write(CONTROL, status)
    if (
        CANCEL.exists()
        or time.time() >= deadline
        or json.loads(STOP.read_text()).get("owner") != OWNER
    ):
        raise RuntimeError("resume cancelled, deadline elapsed, or stop ownership changed")
    # Wait for the venv launcher to exit too; never run two training workers.
    for _ in range(20):
        running = [
            p
            for p in psutil.process_iter(["cmdline"])
            if RUNNER.name in " ".join(p.info["cmdline"] or [])
        ]
        if not running:
            break
        time.sleep(0.5)
    if running:
        raise RuntimeError("another training process exists; no resume")
    if (
        CANCEL.exists()
        or time.time() >= deadline
        or json.loads(STOP.read_text()).get("owner") != OWNER
        or sha(RESULTS / "authority.json") != AUTHORITY_SHA
    ):
        raise RuntimeError("resume cancelled or authority changed immediately before launch")
    STOP.unlink()  # Only this controller's precisely identified marker.
    with (
        (RUN / "campaign.post_eval20.stdout.log").open("wb") as stdout,
        (RUN / "campaign.post_eval20.stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER), "--resume", str(snapshot)],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    status.update(status="resumed_after_evaluation20", resumed_pid=process.pid)
    write(CONTROL, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--py-spy", required=True)
    try:
        run(parser.parse_args())
    except Exception as error:
        status = json.loads(CONTROL.read_text()) if CONTROL.exists() else {"owner": OWNER}
        status.update(status="FAILED_REQUIRES_REVIEW", error=repr(error))
        write(CONTROL, status)
        raise
