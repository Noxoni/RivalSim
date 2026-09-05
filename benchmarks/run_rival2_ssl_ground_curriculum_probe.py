"""One reset-only PPO probe, preserving the comparison parent and old experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    SOURCES as BASE_SOURCES,
)
from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    append_json,
    check_package,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from benchmarks.run_rival2_ssl_exploration_comparison import PARENT, PARENT_SHA
from rivalsim.fresh_ground_30hz import (
    SEED,
    FreshGroundEnv,
    content_hash,
    policy_config,
    scenario_hash,
)
from rivalsim.fresh_ground_30hz import authority as base_authority
from rivalsim.fresh_ground_30hz_training import FreshGroundTrainer, evaluate
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic
from rivalsim.rival2_policy import hybrid_log_probability
from rivalsim.rival2_recurrent_ppo import _sequence_major, recurrent_minibatch_step
from rivalsim.ssl_ground_curriculum_probe import VERSION, probe_scenarios, specification

RESULTS = ROOT / "results/rival2/ssl_ground_curriculum_probe_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/ssl_ground_curriculum_probe_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/ssl-ground-curriculum-probe-v1")
SOURCES = [
    *BASE_SOURCES,
    "rivalsim/ssl_ground_curriculum_probe.py",
    "benchmarks/run_rival2_ssl_ground_curriculum_probe.py",
    "tests/test_ssl_ground_curriculum_probe.py",
]
COLLISION = "G:/dev/RLBot-Rival/bot/collision_meshes"


def authority():
    return dict(
        version=VERSION,
        parent_path=str(PARENT.relative_to(ROOT)),
        parent_sha256=PARENT_SHA,
        parent_update=597,
        additional_updates=30,
        worlds=32768,
        base_authority_sha256=content_hash(base_authority()),
        curriculum=specification(),
        only_change="Training reset-state bank. Inherit parent model, Adam, RNG; no sigma multiplier.",
        unchanged=[
            "reward",
            "physics",
            "observations",
            "actions",
            "actor_and_critic_architecture",
            "PPO",
            "exploration",
            "current_selfplay",
            "finite_safety",
            "KL_telemetry_only",
        ],
        checkpoints="Rolling each update; immutable at +10,+20,+30",
        evaluation="Original deterministic acquisition, finishing, Nexto protocols at +0,+10,+20,+30. "
        "Development cases, not untouched tests, not full matches or SSL proof.",
        interpretation="Promising only if original acquisition exceeds matched control by >=4/64 "
        "at BOTH +20 and +30, and exceeds parent by >=4/64 at +30, without "
        "finishing goals falling >3 below control. This bounds an experiment, "
        "not a permanent capability/deployment gate. Report every boundary.",
        no_automatic_extension=True,
        goal_continues="SSL development continues after pilot analysis; never silently resume original stalled campaign.",
        resume="Same probe only, verified checkpoint SHA, retained Adam/RNG/counters; fresh scenario episodes and zero hidden.",
    )


def verify(published=False):
    check_package(require_preflight=True)
    assert sha(PARENT) == PARENT_SHA
    package = json.loads((RESULTS / "package.json").read_text())
    assert json.loads((RESULTS / "authority.json").read_text()) == authority()
    for name, digest in package["sources"].items():
        assert sha(ROOT / name) == digest, name
    if published:
        p = json.loads((RESULTS / "preflight.json").read_text())
        assert p["verdict"] == "PASS" and p["package_sha256"] == sha(RESULTS / "package.json")
        for name in [
            *SOURCES,
            *(
                f"results/rival2/ssl_ground_curriculum_probe_v1/{n}.json"
                for n in ("authority", "package", "preflight")
            ),
        ]:
            blob = subprocess.check_output(["git", "show", f"origin/main:{name}"], cwd=ROOT)
            assert blob.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(
                b"\r\n", b"\n"
            ), name
    return package


def make():
    bank = probe_scenarios(32768)
    env = FreshGroundEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    trainer = FreshGroundTrainer(env, model=IndependentCriticActorCritic(policy_config()))
    trainer.load_checkpoint(PARENT)
    assert trainer.nexto_probability == 0 and trainer.nexto is None
    return trainer, bank


def prepare():
    if (RESULTS / "package.json").exists():
        raise RuntimeError("frozen package exists")
    check_package(require_preflight=True)
    assert sha(PARENT) == PARENT_SHA
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bank = probe_scenarios(32768)
    write_json(RESULTS / "authority.json", authority())
    write_json(
        RESULTS / "package.json",
        dict(
            utc=utc(),
            sources={n: sha(ROOT / n) for n in SOURCES},
            parent_model_sha256=tensor_hash(parent["model"]),
            scenario_sha256=scenario_hash(bank),
            authority_sha256=content_hash(authority()),
            scenario_summary=bank.summary(),
        ),
    )


def preflight():
    package = verify()
    trainer, bank = make()
    before = tensor_hash(trainer.model.state_dict())
    steps = [float(s["step"]) for s in trainer.optimizer.state.values()]
    original = trainer.env.observation
    frozen = original.clone()
    rollout = trainer.collect_rollout()
    rollout.compute_gae(trainer.ppo_config)
    obs = _sequence_major(rollout.observations)
    reset = _sequence_major(rollout.reset_before)
    hidden = rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim)
    with torch.no_grad():
        actor, values, _ = trainer.model(obs[:728], hidden[:, :728], reset_before=reset[:728])
        logp = hybrid_log_probability(
            actor,
            _sequence_major(rollout.actions)[:728],
            pre_tanh=_sequence_major(rollout.pre_tanh)[:728],
            config=trainer.policy_config,
        )
    log_error = float((logp - _sequence_major(rollout.old_log_probability)[:728]).abs().max())
    value_error = float((values - _sequence_major(rollout.values)[:728]).abs().max())
    advantages = _sequence_major(rollout.advantages)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    metrics = recurrent_minibatch_step(
        trainer.model,
        trainer.optimizer,
        trainer.ppo_config,
        None,
        observation=obs,
        initial_hidden=hidden,
        reset_before=reset,
        action=_sequence_major(rollout.actions),
        pre_tanh=_sequence_major(rollout.pre_tanh),
        old_log_probability=_sequence_major(rollout.old_log_probability),
        normalized_advantage=advantages,
        returns=_sequence_major(rollout.returns),
        train_mask=_sequence_major(rollout.train_mask),
        sequence_index=torch.arange(728, device=trainer.device),
        sequence_microbatch_size=728,
        take_step=False,
        optimize_execution=True,
    )
    checks = dict(
        exact_parent=before == package["parent_model_sha256"],
        scenario_hash=scenario_hash(bank) == package["scenario_sha256"],
        model_unchanged=before == tensor_hash(trainer.model.state_dict()),
        optimizer_steps_unchanged=steps
        == [float(s["step"]) for s in trainer.optimizer.state.values()],
        observation_not_mutated=torch.equal(original, frozen),
        first_stored_observation_correct=torch.equal(rollout.observations[0], frozen),
        logprob_exact=log_error == 0,
        value_consistent=value_error < 1e-3,
        finite_loss=bool(torch.isfinite(metrics["total_loss"])),
        finite_gradients=all(
            bool(torch.isfinite(p.grad).all())
            for p in trainer.model.parameters()
            if p.grad is not None
        ),
        finite_rollout=all(
            bool(torch.isfinite(getattr(rollout, k)).all())
            for k in ("observations", "actions", "rewards", "values", "returns", "advantages")
        ),
        no_mechanics_reward=trainer.env.world.gameplay_v3 is None
        and trainer.env.world.gameplay_120 is None,
        parent_unchanged=sha(PARENT) == PARENT_SHA,
    )
    report = dict(
        utc=utc(),
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        package_sha256=sha(RESULTS / "package.json"),
        optimizer_steps_taken=0,
        logprob_error=log_error,
        value_error=value_error,
        rollout=trainer.last_rollout_metrics,
    )
    write_json(RESULTS / "preflight.json", report)
    print(json.dumps(dict(verdict=report["verdict"], checks=checks)), flush=True)
    assert report["verdict"] == "PASS"


def save(trainer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = trainer.checkpoint_payload()
    p.update(
        format=VERSION + "_CHECKPOINT",
        lineage=VERSION,
        probe_authority_sha256=content_hash(authority()),
        parent_sha256=PARENT_SHA,
        additional_updates=trainer.accepted_updates_total - 597,
        curriculum_sha256=json.loads((RESULTS / "package.json").read_text())["scenario_sha256"],
        source=dict(
            kind="fresh_ground_30hz_parent_continuation",
            parent_sha256=PARENT_SHA,
            parent_update=597,
        ),
    )
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as f:
        torch.save(p, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    return dict(
        path=str(path),
        sha256=sha(path),
        additional_updates=p["additional_updates"],
        accepted_updates=trainer.accepted_updates_total,
    )


def restore(trainer, path, expected_sha):
    assert expected_sha and sha(path) == expected_sha.upper(), (
        "resume requires verified checkpoint hash"
    )
    p = torch.load(path, map_location=trainer.device, weights_only=False)
    assert p["format"] == VERSION + "_CHECKPOINT" and p["parent_sha256"] == PARENT_SHA
    assert p["probe_authority_sha256"] == content_hash(authority())
    assert p["policy_config_sha256"] == trainer.policy_config.content_hash
    assert p["ppo_config_sha256"] == trainer.ppo_config.content_hash
    trainer.model.load_state_dict(p["model"], strict=True)
    trainer.optimizer.load_state_dict(p["optimizer"])
    for k in (
        "accepted_updates_total",
        "phase_accepted_updates",
        "policy_version",
        "total_agent_samples",
        "physical_physics_ticks_experienced",
    ):
        setattr(trainer, k, p[k])
    for name, key in (
        ("policy_generator", "policy_generator_state"),
        ("shuffle_generator", "shuffle_generator_state"),
        ("opponent_generator", "opponent_rng"),
    ):
        getattr(trainer, name).set_state(p[key].cpu())
    torch.set_rng_state(p["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(p["torch_cuda_rng_state"].cpu(), trainer.device)
    trainer.hidden.zero_()
    trainer.reset_before.fill_(True)
    trainer.episode_has_touch.zero_()


def run(args):
    verify(published=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if (EXTERNAL / "STOP").exists():
        raise RuntimeError("STOP present")
    if (EXTERNAL / "latest.json").exists() and not args.resume:
        raise RuntimeError("existing campaign needs same-lineage resume")
    import msvcrt

    lease = (EXTERNAL / "campaign.lock").open("a+b")
    lease.seek(0)
    if not lease.read(1):
        lease.write(b"0")
        lease.flush()
    lease.seek(0)
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    trainer, _ = make()
    if args.resume:
        restore(trainer, args.resume, args.resume_sha256)
    latest = None

    def state(status, **kwargs):
        write_json(
            EXTERNAL / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                status=status,
                additional_updates=trainer.accepted_updates_total - 597,
                latest_checkpoint=latest,
                **kwargs,
            ),
        )

    def checkpoint():
        record = save(trainer, EXTERNAL / f"rolling_{trainer.accepted_updates_total % 2}.pt")
        write_json(EXTERNAL / "latest.json", record)
        return record

    def evaluation():
        state("evaluating")
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        report = evaluate(trainer.model, COLLISION)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        report.update(
            utc=utc(), checkpoint=latest, additional_updates=trainer.accepted_updates_total - 597
        )
        write_json(RESULTS / f"evaluation_{trainer.accepted_updates_total - 597:03d}.json", report)
        print("EVALUATION " + json.dumps(report), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        latest = checkpoint()
        if trainer.accepted_updates_total == 597:
            evaluation()
        while trainer.accepted_updates_total < 627 and not (EXTERNAL / "STOP").exists():
            state("rollout")
            start = time.monotonic()
            rollout = trainer.collect_rollout()
            rollout_seconds = time.monotonic() - start
            state("optimizing")
            start = time.monotonic()
            metrics = trainer.update(rollout)
            del rollout
            latest = checkpoint()
            row = dict(
                utc=utc(),
                additional_updates=trainer.accepted_updates_total - 597,
                rollout_seconds=rollout_seconds,
                ppo_seconds=time.monotonic() - start,
                training=trainer.last_rollout_metrics,
                ppo={
                    k: float(v) if torch.isfinite(v) else str(float(v)) for k, v in metrics.items()
                },
            )
            append_json(EXTERNAL / "training_curve.jsonl", row)
            append_json(RESULTS / "training_curve.jsonl", row)
            if row["additional_updates"] % 10 == 0:
                save(trainer, CHECKPOINTS / f"plus_{row['additional_updates']:03d}.pt")
                evaluation()
            print("UPDATE " + json.dumps(row), flush=True)
        state("completed" if trainer.accepted_updates_total == 627 else "stopped")
    except BaseException as error:
        write_json(
            EXTERNAL / "failure.json",
            dict(
                utc=utc(),
                error=str(error),
                traceback=traceback.format_exc(),
                diagnostics=getattr(error, "diagnostics", None),
                last_accepted_checkpoint=latest,
            ),
        )
        state("fault_stopped")
        raise
    finally:
        lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "preflight", "run"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == "run":
        run(args)
    else:
        {"prepare": prepare, "preflight": preflight}[args.mode]()
