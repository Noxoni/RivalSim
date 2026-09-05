"""Frozen, bounded two-arm PPO experiment under the user's SSL-development goal."""

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
    SOURCES,
    append_json,
    check_package,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from rivalsim.fresh_ground_30hz import authority as base_authority
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash
from rivalsim.fresh_ground_30hz_training import evaluate
from rivalsim.fresh_ground_exploration_comparison import ARMS, VERSION, make_comparison_trainer
from rivalsim.rival2_policy import hybrid_log_probability
from rivalsim.rival2_recurrent_ppo import _sequence_major, recurrent_minibatch_step

RESULTS = ROOT / "results/rival2/ssl_development_exploration_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/ssl_development_exploration_v1"
PARENT = CHECKPOINTS / "parent_u000597.pt"
PARENT_SHA = "B0B35CDAF3B3551EC667776EB99C3822F863AAA1F17A0BA2F013B5F216BD87A5"
EXTERNAL = Path("G:/dev/RivalSim-runs/ssl-development-exploration-v1")
NEW_SOURCES = [
    "rivalsim/fresh_ground_exploration_comparison.py",
    "benchmarks/run_rival2_ssl_exploration_comparison.py",
    "tests/test_fresh_ground_exploration_comparison.py",
]


def authority():
    return dict(
        version=VERSION,
        parent=dict(
            path=PARENT.relative_to(ROOT).as_posix(), sha256=PARENT_SHA, accepted_updates=597
        ),
        base_authority_sha256=content_hash(base_authority()),
        arms={
            k: dict(
                learned_effective_analog_sigma_multiplier=v,
                button_changes=False,
                optimizer_steps_inherited=True,
            )
            for k, v in ARMS.items()
        },
        accepted_updates_per_arm=30,
        evaluation_offsets=[0, 10, 20, 30],
        unchanged=[
            "reward",
            "scenario_bank",
            "optimizer_start_state",
            "PPO_settings",
            "actor_means_at_start",
            "button_logits_at_start",
            "model_tensors_at_start",
            "recurrent_semantics",
            "critic",
            "initial_RNG_states",
        ],
        current_selfplay_probability=1.0,
        nexto_training_probability=0.0,
        worlds=32768,
        horizon=90,
        policy_hz=30,
        physics_hz=120,
        evaluation="Same frozen v1 deterministic 64-case protocols; diagnostic/selection, not untouched tests or match win rates",
        selection="Compare final +30 and +20 boundaries, not cherry-picked best. Half-sigma is a promising pilot only if acquisition touch coverage exceeds control by >=4/64 at both boundaries and exceeds parent at +30 by >=4/64, with finishing touch coverage no worse than control by >4/64 and finishing goals no worse by >3 at +30. Otherwise inconclusive/negative. No SSL promotion.",
        optimizer="Same parent Adam, counters and RNG in both arms; no reset of moments",
        resume="Same-arm only; preserve all optimizer/RNG/counters, reset simulator episodes/hidden as in v1",
        safety="KL telemetry only; unchanged finite/corruption rollback; no reward changes",
        parent_campaign="Agent-requested pause, not abandoned user goal. Original STOP remains to prevent accidental concurrent resume.",
    )


def verify(published=False):
    check_package(require_preflight=True)
    assert sha(PARENT) == PARENT_SHA, "comparison parent changed"
    package = json.loads((RESULTS / "package.json").read_text())
    assert json.loads((RESULTS / "authority.json").read_text()) == authority()
    for name, digest in package["sources"].items():
        assert sha(ROOT / name) == digest, f"comparison source changed: {name}"
    if published:
        report = json.loads((RESULTS / "preflight.json").read_text())
        assert report["verdict"] == "PASS" and report["package_sha256"] == sha(
            RESULTS / "package.json"
        )
        for name in [
            *package["sources"],
            *[
                f"results/rival2/ssl_development_exploration_v1/{x}.json"
                for x in ("authority", "package", "preflight")
            ],
        ]:
            remote = subprocess.check_output(["git", "show", f"origin/main:{name}"], cwd=ROOT)
            assert remote.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(
                b"\r\n", b"\n"
            ), name
    return package


def prepare(_args):
    if (RESULTS / "package.json").exists():
        raise RuntimeError("prospective authority already exists; never overwrite")
    check_package(require_preflight=True)
    assert sha(PARENT) == PARENT_SHA
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    assert parent["accepted_updates_total"] == 597 and parent["opponents"]["nexto_probability"] == 0
    write_json(RESULTS / "authority.json", authority())
    write_json(
        RESULTS / "package.json",
        dict(
            utc=utc(),
            authority_sha256=content_hash(authority()),
            parent_model_sha256=tensor_hash(parent["model"]),
            sources={name: sha(ROOT / name) for name in [*SOURCES, *NEW_SOURCES]},
            parent_sha256=PARENT_SHA,
            parent_optimizer_parameter_states=len(parent["optimizer"]["state"]),
        ),
    )
    print("Prepared prospective same-parent comparison", flush=True)


def make(arm, collision_root):
    return make_comparison_trainer(PARENT, arm, collision_root)


def preflight(args):
    package = verify()
    reports = {}
    for arm in ARMS:
        trainer, bank = make(arm, args.collision_root)
        before_model = tensor_hash(trainer.model.state_dict())
        before_steps = [float(s["step"]) for s in trainer.optimizer.state.values()]
        torch.cuda.reset_peak_memory_stats()
        frozen_observation = trainer.env.observation
        observation_copy = frozen_observation.clone()
        start = time.monotonic()
        rollout = trainer.collect_rollout()
        assert torch.equal(frozen_observation, observation_copy), "observation alias mutation"
        assert torch.equal(rollout.observations[0], observation_copy), (
            "state/action timing mismatch"
        )
        rollout.compute_gae(trainer.ppo_config)
        obs, reset = _sequence_major(rollout.observations), _sequence_major(rollout.reset_before)
        initial = rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim)
        count = 728
        with torch.no_grad():
            actor, values, _ = trainer.model(
                obs[:count], initial[:, :count], reset_before=reset[:count]
            )
            logp = hybrid_log_probability(
                actor,
                _sequence_major(rollout.actions)[:count],
                pre_tanh=_sequence_major(rollout.pre_tanh)[:count],
                config=trainer.policy_config,
            )
            log_error = float(
                (logp - _sequence_major(rollout.old_log_probability)[:count]).abs().max()
            )
            value_error = float((values - _sequence_major(rollout.values)[:count]).abs().max())
        adv = _sequence_major(rollout.advantages)
        normalized = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        metrics = recurrent_minibatch_step(
            trainer.model,
            trainer.optimizer,
            trainer.ppo_config,
            None,
            observation=obs,
            initial_hidden=initial,
            reset_before=reset,
            action=_sequence_major(rollout.actions),
            pre_tanh=_sequence_major(rollout.pre_tanh),
            old_log_probability=_sequence_major(rollout.old_log_probability),
            normalized_advantage=normalized,
            returns=_sequence_major(rollout.returns),
            train_mask=_sequence_major(rollout.train_mask),
            sequence_index=torch.arange(count, device=trainer.device),
            sequence_microbatch_size=count,
            take_step=False,
            optimize_execution=True,
        )
        checks = dict(
            model_equals_parent=before_model == package["parent_model_sha256"],
            model_unchanged=before_model == tensor_hash(trainer.model.state_dict()),
            optimizer_steps_unchanged=before_steps
            == [float(s["step"]) for s in trainer.optimizer.state.values()],
            step_sequence_logprob_agrees=log_error < 1e-3,
            value_agrees=value_error < 1e-3,
            finite_loss=bool(torch.isfinite(metrics["total_loss"])),
            finite_gradients=all(
                bool(torch.isfinite(p.grad).all())
                for p in trainer.model.parameters()
                if p.grad is not None
            ),
            finite_rollout=all(
                bool(torch.isfinite(getattr(rollout, k)).all())
                for k in ("observations", "actions", "rewards", "values", "advantages", "returns")
            ),
            nexto_absent=trainer.nexto is None,
            parent_unchanged=sha(PARENT) == PARENT_SHA,
        )
        reports[arm] = dict(
            checks=checks,
            scenario_sha256=scenario_hash(bank),
            log_probability_max_error=log_error,
            value_max_error=value_error,
            peak_torch_gib=torch.cuda.max_memory_allocated() / 2**30,
            seconds=time.monotonic() - start,
            optimizer_steps_taken=0,
            training=trainer.last_rollout_metrics,
        )
        print("PREFLIGHT " + arm + " " + json.dumps(checks), flush=True)
        del rollout, trainer, obs, reset, initial, actor, values, logp, adv, normalized, metrics
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(
        verdict="PASS" if all(all(x["checks"].values()) for x in reports.values()) else "FAIL",
        package_sha256=sha(RESULTS / "package.json"),
        arms=reports,
        utc=utc(),
    )
    write_json(RESULTS / "preflight.json", report)
    if report["verdict"] != "PASS":
        raise RuntimeError("exploration comparison preflight failed")


def save(trainer, arm, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = trainer.checkpoint_payload()
    payload.update(
        format=VERSION + "_CHECKPOINT",
        lineage=VERSION,
        comparison_authority_sha256=content_hash(authority()),
        parent_sha256=PARENT_SHA,
        arm=arm,
        analog_sigma_scale=ARMS[arm],
        additional_updates=trainer.accepted_updates_total - 597,
        reward_authority_sha256=content_hash(base_authority()),
        source=dict(
            kind="fresh_ground_30hz_v1_continuation",
            parent_sha256=PARENT_SHA,
            parent_update=597,
            no_V5_or_BC_weights=True,
        ),
    )
    temp = path.with_suffix(".pt.tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    return dict(
        path=str(path),
        sha256=sha(path),
        additional_updates=payload["additional_updates"],
        accepted_updates=trainer.accepted_updates_total,
        arm=arm,
    )


def restore(trainer, arm, path):
    p = torch.load(path, map_location=trainer.device, weights_only=False)
    assert p["format"] == VERSION + "_CHECKPOINT" and p["arm"] == arm
    assert (
        p["comparison_authority_sha256"] == content_hash(authority())
        and p["parent_sha256"] == PARENT_SHA
    )
    assert p["policy_config_sha256"] == trainer.policy_config.content_hash
    assert p["ppo_config_sha256"] == trainer.ppo_config.content_hash
    trainer.model.load_state_dict(p["model"], strict=True)
    trainer.optimizer.load_state_dict(p["optimizer"])
    for key in (
        "accepted_updates_total",
        "phase_accepted_updates",
        "policy_version",
        "total_agent_samples",
        "physical_physics_ticks_experienced",
    ):
        setattr(trainer, key, p[key])
    trainer.policy_generator.set_state(p["policy_generator_state"].cpu())
    trainer.shuffle_generator.set_state(p["shuffle_generator_state"].cpu())
    trainer.opponent_generator.set_state(p["opponent_rng"].cpu())
    torch.set_rng_state(p["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(p["torch_cuda_rng_state"].cpu(), trainer.device)
    trainer.hidden.zero_()
    trainer.reset_before.fill_(True)
    trainer.episode_has_touch.zero_()


def run(args):
    verify(published=True)
    arm = args.arm
    directory = EXTERNAL / arm
    directory.mkdir(parents=True, exist_ok=True)
    if (EXTERNAL / "STOP").exists():
        raise RuntimeError("comparison STOP present")
    import msvcrt

    lease = (EXTERNAL / "campaign.lock").open("a+b")
    lease.seek(0)
    if not lease.read(1):
        lease.write(b"0")
        lease.flush()
    lease.seek(0)
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    trainer, _ = make(arm, args.collision_root)
    if args.resume:
        restore(trainer, arm, args.resume)
    elif (directory / "latest.json").exists():
        raise RuntimeError("existing arm requires explicit resume")
    latest = None

    def state(status, **more):
        write_json(
            directory / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                arm=arm,
                status=status,
                accepted_updates=trainer.accepted_updates_total,
                additional_updates=trainer.accepted_updates_total - 597,
                latest_checkpoint=latest,
                **more,
            ),
        )

    def checkpoint():
        record = save(trainer, arm, directory / f"rolling_{trainer.accepted_updates_total % 2}.pt")
        write_json(directory / "latest.json", record)
        return record

    def evaluation():
        offset = trainer.accepted_updates_total - 597
        state("evaluating")
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        report = evaluate(trainer.model, args.collision_root)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        report.update(arm=arm, additional_updates=offset, checkpoint=latest, utc=utc())
        write_json(RESULTS / arm / f"evaluation_{offset:03d}.json", report)
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
            rollout.compute_gae(trainer.ppo_config)
            rollout_seconds = time.monotonic() - start
            state("optimizing")
            start = time.monotonic()
            metrics = trainer.update(rollout)
            del rollout
            row = dict(
                utc=utc(),
                arm=arm,
                additional_updates=trainer.accepted_updates_total - 597,
                accepted_updates=trainer.accepted_updates_total,
                rollout_seconds=rollout_seconds,
                ppo_seconds=time.monotonic() - start,
                training=trainer.last_rollout_metrics,
                ppo={
                    k: float(v) if torch.isfinite(v) else str(float(v)) for k, v in metrics.items()
                },
            )
            latest = checkpoint()
            append_json(directory / "training_curve.jsonl", row)
            append_json(RESULTS / arm / "training_curve.jsonl", row)
            if row["additional_updates"] % 10 == 0:
                save(trainer, arm, CHECKPOINTS / arm / f"plus_{row['additional_updates']:03d}.pt")
                evaluation()
            print("UPDATE " + json.dumps(row), flush=True)
        state("completed" if trainer.accepted_updates_total == 627 else "stopped")
    except BaseException as error:
        write_json(
            directory / "failure.json",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "preflight", "run", "both"))
    parser.add_argument("--arm", choices=tuple(ARMS), default="control")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--collision-root", default="G:/dev/RLBot-Rival/bot/collision_meshes")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == "both":
        for arm in ARMS:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "run",
                    "--arm",
                    arm,
                    "--collision-root",
                    args.collision_root,
                ],
                check=True,
                cwd=ROOT,
            )
    else:
        {"prepare": prepare, "preflight": preflight, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
