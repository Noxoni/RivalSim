"""Prepare, validate, and run the user-stopped fresh random 30Hz experiment.

No automatic update/time ceiling. STOP file requests a checkpointed boundary
stop. Only --resume can load weights, and only this exact fresh lineage is valid.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rivalsim.fresh_ground_30hz import authority, content_hash, scenario_hash, VERSION
from rivalsim.fresh_ground_30hz_training import make_trainer, evaluate
from rivalsim.rival2_recurrent_ppo import recurrent_minibatch_step, _sequence_major

RESULTS = ROOT / "results/rival2/fresh_ground_30hz_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/fresh_ground_30hz_v1"
SOURCES = [
    "rivalsim/fresh_ground_30hz.py", "rivalsim/fresh_ground_30hz_training.py",
    "benchmarks/run_rival2_fresh_ground_30hz_v1.py", "rivalsim/rival2_env.py",
    "rivalsim/kernels/rival2.py", "rivalsim/ssl_foundation_v1.py", "rivalsim/rival2_contracts.py",
    "rivalsim/rival2_recurrent_training.py", "rivalsim/rival2_recurrent_ppo.py",
    "rivalsim/rival2_independent_critic.py", "rivalsim/rival2_unified_policy.py",
    "rivalsim/rival2_policy.py", "rivalsim/rival2_ppo.py", "rivalsim/recurrent_execution.py",
    "third_party/nexto/adapter.py",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest().upper()


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save(trainer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = trainer.checkpoint_payload()
    payload["package"] = json.loads((RESULTS / "package.json").read_text()) if (RESULTS / "package.json").exists() else None
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"path": str(path), "sha256": sha(path), "accepted_updates": trainer.accepted_updates_total}


def check_package(require_preflight=False):
    p = json.loads((RESULTS / "package.json").read_text())
    if json.loads((RESULTS / "authority.json").read_text()) != authority():
        raise RuntimeError("authority mismatch")
    for name, digest in p["sources"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"source changed after freeze: {name}")
    if sha(CHECKPOINTS / "initial.pt") != p["initial_checkpoint"]["sha256"]:
        raise RuntimeError("initial checkpoint changed")
    if require_preflight:
        report = json.loads((RESULTS / "preflight.json").read_text())
        if report["verdict"] != "PASS" or report["package_sha256"] != sha(RESULTS / "package.json"):
            raise RuntimeError("full-scale preflight missing or stale")
    return p


def prepare(args):
    if (RESULTS / "package.json").exists():
        raise RuntimeError("package already exists; do not overwrite a frozen training authority")
    trainer, bank = make_trainer(args.collision_root)
    write_json(RESULTS / "authority.json", authority())
    initial = save(trainer, CHECKPOINTS / "initial.pt")
    write_json(RESULTS / "package.json", {
        "version": VERSION, "prepared_utc": utc(), "authority_sha256": content_hash(authority()),
        "sources": {name: sha(ROOT / name) for name in SOURCES},
        "initial_checkpoint": initial, "initial_model_sha256": tensor_hash(trainer.model.state_dict()),
        "initial_optimizer_state_count": len(trainer.optimizer.state), "scenario_sha256": scenario_hash(bank),
        "scenario_summary": bank.summary(), "inherited_policy_checkpoint": None,
        "git_before_implementation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    print(json.dumps(initial), flush=True)


def preflight(args):
    package = check_package()
    trainer, bank = make_trainer(args.collision_root)
    torch.cuda.reset_peak_memory_stats()
    before = tensor_hash(trainer.model.state_dict())
    start = time.monotonic()
    rollout = trainer.collect_rollout()
    print("Full-scale 32768 x 90 rollout captured; starting no-step backward", flush=True)
    rollout.compute_gae(trainer.ppo_config)
    advantage = _sequence_major(rollout.advantages)
    normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    count = rollout.sequence_layout(trainer.ppo_config.minibatch_size).sequences_per_minibatch
    metrics = recurrent_minibatch_step(trainer.model, trainer.optimizer, trainer.ppo_config, None,
        observation=_sequence_major(rollout.observations),
        initial_hidden=rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim),
        reset_before=_sequence_major(rollout.reset_before), action=_sequence_major(rollout.actions),
        pre_tanh=_sequence_major(rollout.pre_tanh), old_log_probability=_sequence_major(rollout.old_log_probability),
        normalized_advantage=normalized, returns=_sequence_major(rollout.returns),
        train_mask=_sequence_major(rollout.train_mask), sequence_index=torch.arange(count, device=trainer.device),
        sequence_microbatch_size=728, take_step=False, optimize_execution=True)
    checks = {
        "fresh_model_reproduced": before == package["initial_model_sha256"],
        "scenario_reproduced": scenario_hash(bank) == package["scenario_sha256"],
        "model_byte_identical_after_backward": before == tensor_hash(trainer.model.state_dict()),
        "no_optimizer_state_or_steps": len(trainer.optimizer.state) == trainer.accepted_updates_total == 0,
        "finite_rollout": all(bool(torch.isfinite(getattr(rollout, name)).all()) for name in
                              ("observations", "actions", "values", "rewards", "advantages", "returns")),
        "finite_gradients": all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters() if p.grad is not None),
        "finite_loss": bool(torch.isfinite(metrics["total_loss"])),
        "cadence": trainer.env.policy_hz == 30 and trainer.env.physics_ticks_per_decision == 4,
        "two_gib_vram_headroom": torch.cuda.max_memory_allocated() < torch.cuda.get_device_properties(0).total_memory - 2*2**30,
        "no_named_mechanics_reward_path": trainer.env.world.gameplay_v3 is None and trainer.env.world.gameplay_120 is None,
    }
    report = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "package_sha256": sha(RESULTS / "package.json"), "worlds": 32768, "horizon": 90,
        "rollout_logical_gib": rollout.logical_bytes/2**30,
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "elapsed_seconds": time.monotonic()-start, "optimizer_step_taken": False,
        "rollout_metrics": trainer.last_rollout_metrics}
    write_json(RESULTS / "preflight.json", report)
    print(json.dumps(report), flush=True)
    if report["verdict"] != "PASS":
        raise RuntimeError("preflight failed")


def run(args):
    package = check_package(require_preflight=True)
    # Refuse unpublished or dirty authority/source files, without touching concurrent work.
    for name in [*SOURCES, "results/rival2/fresh_ground_30hz_v1/authority.json",
                 "results/rival2/fresh_ground_30hz_v1/package.json", "results/rival2/fresh_ground_30hz_v1/preflight.json"]:
        committed = subprocess.check_output(["git", "show", f"origin/main:{name}"], cwd=ROOT)
        if committed.replace(b"\r\n", b"\n") != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise RuntimeError(f"prospective package not persisted on origin/main: {name}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "STOP").exists():
        raise RuntimeError("STOP request present; do not restart without user authorization")
    if (run_dir / "campaign_state.json").exists() and not args.resume:
        raise RuntimeError("existing campaign requires explicit same-lineage --resume")
    # Exclusive OS-held lease: stale files cannot spawn a second GPU learner.
    import msvcrt
    lease = (run_dir / "campaign.lock").open("a+b")
    lease.seek(0)
    if lease.read(1) == b"":
        lease.write(b"0"); lease.flush()
    lease.seek(0)
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    trainer, bank = make_trainer(args.collision_root)
    if scenario_hash(bank) != package["scenario_sha256"]:
        raise RuntimeError("scenario identity mismatch")
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif tensor_hash(trainer.model.state_dict()) != package["initial_model_sha256"]:
        raise RuntimeError("fresh initialization mismatch")
    started = utc()
    latest = None
    def state(status, **extra):
        write_json(run_dir / "campaign_state.json", dict(
            version=VERSION, status=status, pid=os.getpid(), process_start_utc=started, updated_utc=utc(),
            accepted_updates=trainer.accepted_updates_total, total_trainable_samples=trainer.total_agent_samples,
            latest_checkpoint=latest, maximum_updates=None, deadline=None,
            nexto_probability=trainer.nexto_probability, authority_sha256=content_hash(authority()), **extra))
    def checkpoint():
        # Two rolling slots avoid overwriting the last valid checkpoint before publication.
        path = run_dir / f"rolling_{trainer.accepted_updates_total % 2}.pt"
        record = save(trainer, path)
        write_json(run_dir / "latest.json", record)
        return record
    def eval_boundary():
        offset = trainer.accepted_updates_total
        state("evaluating", evaluation_update=offset)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        report = evaluate(trainer.model, args.collision_root)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        trainer.accept_evaluation(report)
        report = dict(update=offset, checkpoint=latest, completed_utc=utc(), **report)
        write_json(RESULTS / "evaluations" / f"u{offset:06d}.json", report)
        write_json(run_dir / "latest_evaluation.json", report)
        print("EVALUATION " + json.dumps(report), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    try:
        latest = checkpoint()
        if trainer.accepted_updates_total == 0:
            eval_boundary()
            latest = checkpoint()
        while not (run_dir / "STOP").exists():
            state("rollout")
            start = time.monotonic()
            rollout = trainer.collect_rollout()
            rollout_seconds = time.monotonic()-start
            rollout.compute_gae(trainer.ppo_config)
            state("optimizing", rollout_seconds=rollout_seconds)
            update_start = time.monotonic()
            metrics = trainer.update(rollout)
            row = {"update": trainer.accepted_updates_total, "utc": utc(),
                "rollout_seconds": rollout_seconds, "ppo_seconds": time.monotonic()-update_start,
                "training": trainer.last_rollout_metrics,
                "ppo": {k: (float(v) if torch.isfinite(v) else str(float(v))) for k, v in metrics.items()},
                "total_trainable_samples": trainer.total_agent_samples}
            del rollout
            latest = checkpoint()
            append_json(run_dir / "training_curve.jsonl", row)
            append_json(RESULTS / "training_curve.jsonl", row)
            if trainer.accepted_updates_total in (10, 20) or trainer.accepted_updates_total % 50 == 0:
                snapshot = save(trainer, CHECKPOINTS / f"u{trainer.accepted_updates_total:06d}.pt")
                append_json(RESULTS / "snapshots.jsonl", snapshot)
                eval_boundary()
                latest = checkpoint()
            state("running")
            print("UPDATE " + json.dumps(row), flush=True)
        state("stopped_by_user")
    except BaseException as error:
        record = {"utc": utc(), "error": str(error), "type": type(error).__name__,
                  "traceback": traceback.format_exc(), "last_accepted_checkpoint": latest,
                  "diagnostics": getattr(error, "diagnostics", None)}
        write_json(run_dir / "failure.json", record)
        state("fault_stopped", error=str(error))
        raise
    finally:
        lease.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "preflight", "run"))
    parser.add_argument("--collision-root", default="G:/dev/RLBot-Rival/bot/collision_meshes")
    parser.add_argument("--run-dir", default="G:/dev/RivalSim-runs/fresh-ground-30hz-v1")
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.set_num_threads(8)
    {"prepare": prepare, "preflight": preflight, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
