"""Prospective same-weight/Adam continuation with staged inference-only Nexto."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    append_json,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from benchmarks.run_rival2_ssl_entity_joint_control import COLLISION
from benchmarks.run_rival2_ssl_entity_joint_control import verify as verify_pilot
from rivalsim.fresh_ground_30hz import (
    SEED,
    FreshGroundEnv,
    content_hash,
    ppo_config,
    scenario_hash,
    scenarios,
)
from rivalsim.fresh_ground_30hz_training import evaluate
from rivalsim.ssl_entity_evaluation import DeterministicEvaluationView
from rivalsim.ssl_entity_mixed_training import (
    MixedEntityRolloutCollector,
    mixed_joint_ppo_update,
    mixed_sequence_data,
)
from rivalsim.ssl_entity_policy import ENTITY_VERSION, EntityJointControlActorCritic
from rivalsim.ssl_entity_training import (
    finite_model_and_optimizer,
    fresh_entity_optimizer,
    joint_sequence_loss,
)

VERSION = "RIVAL2_SSL_ENTITY_CONTINUATION_V1"
RESULTS = ROOT / "results/rival2/ssl_entity_continuation_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/ssl_entity_continuation_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/ssl-entity-continuation-v1")
PILOT = ROOT / "results/rival2/ssl_entity_joint_control_v1"
PARENT = ROOT / "checkpoints/rival2/ssl_entity_joint_control_v1/plus_100.pt"
PARENT_SHA = "B5F7D19471257758966EB0B407797CFCFBD0F6CE8E86BCF9E13D41FBA4EA7ABA"
SOURCES = [
    "benchmarks/run_rival2_ssl_entity_continuation.py",
    "rivalsim/ssl_entity_mixed_training.py",
    "tests/test_ssl_entity_mixed_training.py",
]
EVIDENCE = [
    PILOT / name
    for name in (
        "evaluation_100.json",
        "full_match_protocol.json",
        "full_match_comparison.json",
        "full_match_cuda_interface_check.json",
        "accepted_100_integrity.json",
    )
]


def text_sha(path):
    # Explicit canonical LF identity for text; checkpoint hashes remain raw bytes.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def authority():
    return dict(
        version=VERSION,
        parent_path=PARENT.relative_to(ROOT).as_posix(),
        parent_sha256=PARENT_SHA,
        initial_accepted_updates=100,
        old_pilot_unchanged=True,
        initialization=(
            "Exact entity +100 model and Adam moments/counters; no new weights or optimizer reset"
        ),
        worlds=32768,
        physics_hz=120,
        policy_hz=30,
        hold_ticks=4,
        ppo=asdict(ppo_config()),
        architecture_and_action_contract=(
            "Unchanged entity recurrent joint90, external 182 observations/8 controls"
        ),
        reward=(
            "Unchanged FreshGround30Hz +/-10 terminal plus seven discounted "
            "state-potential differences"
        ),
        no_new_rewards_or_mechanic_detectors=True,
        no_bc_or_old_v5_lineage=True,
        initial_nexto_probability=0.0,
        nexto_probability_after_qualification=0.20,
        qualification=(
            "Existing schedule: two consecutive scheduled development evaluations "
            "with acquisition touch fraction >=.60, conditional median first touch "
            "<=5 seconds and >=1 finishing goal. +100 is the first qualifying "
            "boundary; +50 did not qualify."
        ),
        opponent_assignment=(
            "Sample family and learner side at physical resets only; "
            "once enabled, .20 Nexto remains enabled. Rest is current "
            "self-play. No historical/V5/Wisp."
        ),
        train_mask=(
            "Both current agents in self-play; only current agent versus Nexto. "
            "Nexto is inference-only."
        ),
        advantage=(
            "Family-local normalization over trainable samples; "
            "full 90-decision recurrent sequences and reset masks"
        ),
        nexto_policy_hz=15,
        nexto_control=(
            "Existing pinned native adapter and kickoff sequence; Rival has no scripted prefix"
        ),
        optimizer=(
            "Resume existing Adam and step counters; policy1e-4, critic3e-4, "
            "two epochs. Independent critic loss."
        ),
        kl="Telemetry only, no KL rejection or retention objective",
        finite_checks="Full model/gradient/Adam; whole-update corruption rollback",
        evaluation_every=50,
        snapshot_every=50,
        checkpoint_every=1,
        evaluation=(
            "Same fixed64 acquisition/finishing/Nexto development cases; "
            "no selection by a lucky boundary. Full match follow-ups are "
            "separate read-only comparisons."
        ),
        update_ceiling=None,
        deadline=None,
        stop=(
            "User stop or numerical/corruption/runtime fault. Monitor each new evaluation "
            "and investigate stagnation rather than silently changing settings."
        ),
        resume=(
            "Restore model/Adam/counters/policy+shuffle+opponent+CPU+CUDA RNG. "
            "Fresh scenario episodes and zero hidden. Saved opponent assignments/cache "
            "are provenance, not replayed onto different physical states."
        ),
        exposure=(
            "Actual trainable current-agent decisions; ignored Nexto samples excluded. "
            "Physical world ticks separately counted."
        ),
        publication=(
            "Authority, sources, focused tests, full-scale no-step preflight and "
            "parent identity published before first continuation optimizer step"
        ),
    )


def native_preflight():
    verify_pilot(published=True)
    assert sha(PARENT) == PARENT_SHA
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = EntityJointControlActorCritic().cuda()
    model.load_state_dict(payload["model"], strict=True)
    before = tensor_hash(model.state_dict())
    bank = scenarios(32768)
    env = FreshGroundEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = MixedEntityRolloutCollector(env, model)
    # Explicit preflight coverage of the later mixed regime; not campaign activation.
    collector.nexto_probability = 0.20
    collector.assign(torch.ones(32768, device="cuda:0", dtype=torch.bool))
    teacher_hash = tensor_hash(collector.nexto.actor.state_dict())
    optimizer = fresh_entity_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    assert all(float(s["step"]) == 18200 for s in optimizer.state.values())
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rollout = collector.collect()
    seconds = time.monotonic() - started
    data = mixed_sequence_data(rollout, ppo_config())
    index = data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()
    total, _ = joint_sequence_loss(model, data, index, ppo_config())
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
    gradient_finite = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    optimizer.zero_grad(set_to_none=True)
    _, values, _ = model(
        data["observations"][index],
        data["initial_hidden"][:, index],
        reset_before=data["reset_before"][index],
    )
    values[data["train_mask"][index]].sum().backward()
    isolation = all(
        p.grad is None or not bool(p.grad.count_nonzero())
        for name, p in model.named_parameters()
        if not name.startswith("critic.")
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        logits, value, _ = model(
            data["observations"][index],
            data["initial_hidden"][:, index],
            reset_before=data["reset_before"][index],
        )
        from rivalsim.ssl_joint_control_policy import categorical_statistics

        logp, _ = categorical_statistics(logits, data["action_indices"][index])
        mask = data["train_mask"][index]
        logp_error = float((logp[mask] - data["old_log_probability"][index][mask]).abs().max())
        value_error = float((value[mask] - data["values"][index][mask]).abs().max())
    checks = dict(
        worlds_32768_horizon90=rollout.num_envs == 32768 and rollout.horizon == 90,
        learner_action_table_exact=torch.equal(
            model.action_table[rollout.action_indices][rollout.train_mask],
            rollout.actions[rollout.train_mask],
        ),
        sample_counts_exact=collector.last_metrics["trainable_agent_samples"]
        == int(rollout.train_mask.sum()),
        nexto_samples_excluded=not bool(
            (rollout.train_mask & (rollout.opponent_family == 1)).sum(2).gt(1).any()
        ),
        nexto_present=collector.last_metrics["nexto_training_sample_count"] > 0,
        finite_state=finite_model_and_optimizer(model, optimizer),
        finite_gradient=gradient_finite,
        critic_isolated=isolation,
        logp_parity=logp_error <= 1e-5,
        value_parity=value_error <= 1e-5,
        unchanged_parent=sha(PARENT) == PARENT_SHA and tensor_hash(model.state_dict()) == before,
        unchanged_teacher=tensor_hash(collector.nexto.actor.state_dict()) == teacher_hash,
        no_optimizer_step=all(float(s["step"]) == 18200 for s in optimizer.state.values()),
        no_mechanics_hotpath=env.world.gameplay_v3 is None and env.world.gameplay_120 is None,
    )
    report = dict(
        utc=utc(),
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        parent_sha256=PARENT_SHA,
        authority_sha256=content_hash(authority()),
        sources={p: text_sha(ROOT / p) for p in SOURCES},
        scenario_sha256=scenario_hash(bank),
        optimizer_steps=0,
        rollout_seconds=seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        logp_max_error=logp_error,
        value_max_error=value_error,
        training_telemetry=collector.last_metrics,
    )
    write_json(RESULTS / "native_preflight.json", report)
    print(json.dumps(report), flush=True)
    assert all(checks.values()), checks


def prepare():
    if (RESULTS / "package.json").exists():
        raise RuntimeError("Continuation authority already frozen")
    verify_pilot(published=True)
    preflight = json.loads((RESULTS / "native_preflight.json").read_text())
    assert preflight["verdict"] == "PASS"
    assert preflight["authority_sha256"] == content_hash(authority())
    assert preflight["sources"] == {p: text_sha(ROOT / p) for p in SOURCES}
    suites = ET.parse(RESULTS / "focused_tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
    assert sha(PARENT) == PARENT_SHA
    assert (
        json.loads((PILOT / "full_match_cuda_interface_check.json").read_text())["verdict"]
        == "PASS"
    )
    result = json.loads((PILOT / "full_match_comparison.json").read_text())
    assert result["policies"][1]["checkpoint"]["sha256"] == PARENT_SHA
    write_json(RESULTS / "authority.json", authority())
    write_json(
        RESULTS / "package.json",
        dict(
            authority_sha256=content_hash(authority()),
            sources=preflight["sources"],
            source_text_identity="SHA256 of UTF8 file bytes with CRLF normalized to LF",
            evidence={p.relative_to(ROOT).as_posix(): text_sha(p) for p in EVIDENCE},
            native_preflight_sha256=text_sha(RESULTS / "native_preflight.json"),
            focused_tests_sha256=text_sha(RESULTS / "focused_tests.xml"),
            scenario_sha256=preflight["scenario_sha256"],
            parent_sha256=PARENT_SHA,
        ),
    )


def verify(published=True):
    verify_pilot(published=published)
    package = json.loads((RESULTS / "package.json").read_text())
    assert json.loads((RESULTS / "authority.json").read_text()) == authority()
    assert content_hash(authority()) == package["authority_sha256"]
    assert sha(PARENT) == PARENT_SHA == package["parent_sha256"]
    for name, digest in {**package["sources"], **package["evidence"]}.items():
        assert text_sha(ROOT / name) == digest, name
    for name in ("native_preflight", "focused_tests"):
        suffix = ".json" if name == "native_preflight" else ".xml"
        assert text_sha(RESULTS / (name + suffix)) == package[name + "_sha256"]
    if published:
        paths = [
            *SOURCES,
            *package["evidence"],
            PARENT.relative_to(ROOT).as_posix(),
            *[
                (RESULTS / p).relative_to(ROOT).as_posix()
                for p in (
                    "authority.json",
                    "package.json",
                    "native_preflight.json",
                    "focused_tests.xml",
                )
            ],
        ]
        for path in paths:
            remote = subprocess.check_output(["git", "show", "origin/main:" + path], cwd=ROOT)
            local = (ROOT / path).read_bytes()
            assert (
                remote == local
                if path.endswith(".pt")
                else remote.replace(b"\r\n", b"\n") == local.replace(b"\r\n", b"\n")
            ), path
    return package


@contextmanager
def gpu_lease():
    import msvcrt

    # Share the prior pilot/evaluation lease: never overlap with their GPU worker.
    path = Path("G:/dev/RivalSim-runs/ssl-entity-joint-control-v1/campaign.lock")
    with path.open("r+b") as lease:
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def run(args):
    package = verify()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if (EXTERNAL / "STOP").exists():
        raise RuntimeError("STOP present")
    if (EXTERNAL / "latest.json").exists() and not args.resume:
        raise RuntimeError("Existing continuation requires explicit same-lineage resume")
    if args.resume and not args.resume_sha256:
        raise ValueError("Resume requires the exact latest accepted checkpoint SHA256")
    with gpu_lease():
        run_locked(args, package)


def run_locked(args, package):
    path = Path(args.resume) if args.resume else PARENT
    assert sha(path) == (args.resume_sha256.upper() if args.resume else PARENT_SHA)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if args.resume:
        assert payload["format"] == VERSION + "_CHECKPOINT"
        assert payload["continuation_authority_sha256"] == content_hash(authority())
        assert payload["continuation_parent_sha256"] == PARENT_SHA
    else:
        assert payload["format"] == ENTITY_VERSION + "_CHECKPOINT"
        assert payload["accepted_updates"] == 100
    model = EntityJointControlActorCritic().cuda()
    assert payload["policy_config_sha256"] == model.config.content_hash
    assert payload["ppo_config_sha256"] == ppo_config().content_hash
    model.load_state_dict(payload["model"], strict=True)
    optimizer = fresh_entity_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    steps = {int(s["step"]) for s in optimizer.state.values()}
    assert len(steps) == 1, steps
    optimizer_steps = steps.pop()
    assert optimizer_steps == payload.get("cumulative_optimizer_steps", 18200)
    assert finite_model_and_optimizer(model, optimizer)
    bank = scenarios(32768)
    assert scenario_hash(bank) == package["scenario_sha256"]
    env = FreshGroundEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = MixedEntityRolloutCollector(env, model)
    if args.resume:
        opponent = payload["opponent_state"]
        collector.nexto_probability = opponent["nexto_probability"]
        collector.competence_streak = opponent["competence_streak"]
        collector.opponent_generator.set_state(opponent["generator_state"].cpu())
        collector.assign(torch.ones_like(collector.is_nexto))
    else:
        collector.accept_evaluation(json.loads((PILOT / "evaluation_100.json").read_text()))
        assert collector.competence_streak == 1 and collector.nexto_probability == 0
    collector.generator.set_state(payload["policy_generator_state"].cpu())
    shuffle = torch.Generator(device="cuda:0")
    shuffle.set_state(payload["shuffle_generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu())
    offset, samples, ticks = (
        payload[k] for k in ("accepted_updates", "new_agent_samples", "new_physics_ticks")
    )
    latest = None

    def state(status, **kwargs):
        write_json(
            EXTERNAL / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                status=status,
                accepted_updates=offset,
                continuation_accepted_updates=offset - 100,
                nexto_probability=collector.nexto_probability,
                cumulative_optimizer_steps=optimizer_steps,
                latest_checkpoint=latest,
                **kwargs,
            ),
        )

    def save(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        p = dict(payload)
        p.update(
            format=VERSION + "_CHECKPOINT",
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            continuation_parent_sha256=PARENT_SHA,
            continuation_authority_sha256=content_hash(authority()),
            continuation_package=package,
            accepted_updates=offset,
            continuation_accepted_updates=offset - 100,
            new_agent_samples=samples,
            new_physics_ticks=ticks,
            cumulative_optimizer_steps=optimizer_steps,
            policy_generator_state=collector.generator.get_state(),
            shuffle_generator_state=shuffle.get_state(),
            opponent_state=collector.opponent_checkpoint_state(),
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            resume_count=payload["resume_count"] + 1,
            scenario_reset_note=authority()["resume"],
            last_training_metrics=collector.last_metrics,
        )
        temporary = destination.with_suffix(".pt.tmp")
        with temporary.open("wb") as file:
            torch.save(p, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        return dict(path=str(destination), sha256=sha(destination), accepted_updates=offset)

    def checkpoint():
        record = save(EXTERNAL / f"rolling_{offset % 2}.pt")
        write_json(EXTERNAL / "latest.json", record)
        return record

    def evaluation():
        nonlocal latest
        state("evaluating")
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        report = evaluate(DeterministicEvaluationView(model), COLLISION)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        collector.accept_evaluation(report)
        latest = checkpoint()  # Persist the transition decision before any new rollout.
        immutable = save(CHECKPOINTS / f"plus_{offset:06d}.pt")
        report.update(
            utc=utc(),
            accepted_updates=offset,
            checkpoint=immutable,
            authority_sha256=content_hash(authority()),
            nexto_probability=collector.nexto_probability,
            competence_streak=collector.competence_streak,
        )
        write_json(RESULTS / f"evaluation_{offset:06d}.json", report)
        print("EVALUATION " + json.dumps(report), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        latest = checkpoint()
        # A worker interrupted during a boundary evaluation owes that evaluation
        # before any additional training. It is development data, not a test split.
        if (
            offset > 100
            and offset % 50 == 0
            and not (RESULTS / f"evaluation_{offset:06d}.json").exists()
        ):
            evaluation()
        while not (EXTERNAL / "STOP").exists():
            state("rollout")
            started = time.monotonic()
            rollout = collector.collect()
            rollout_seconds = time.monotonic() - started
            state("optimizing")
            started = time.monotonic()
            metrics = mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic() - started
            offset += 1
            optimizer_steps += metrics["optimizer_steps"]
            samples += collector.last_metrics["trainable_agent_samples"]
            ticks += collector.last_metrics["physical_physics_ticks"]
            del rollout
            latest = checkpoint()
            row = dict(
                utc=utc(),
                accepted_updates=offset,
                continuation_accepted_updates=offset - 100,
                cumulative_optimizer_steps=optimizer_steps,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
                training=collector.last_metrics,
                ppo=metrics,
            )
            append_json(EXTERNAL / "training_curve.jsonl", row)
            append_json(RESULTS / "training_curve.jsonl", row)
            print("ACCEPTED " + json.dumps(row), flush=True)
            if offset % 50 == 0:
                evaluation()
        state("stopped_at_accepted_boundary")
    except Exception as exc:
        failure = dict(
            utc=utc(),
            accepted_updates=offset,
            latest_checkpoint=latest,
            exception=repr(exc),
            traceback=traceback.format_exc(),
            automatic_retry=False,
        )
        write_json(EXTERNAL / "failure.json", failure)
        write_json(RESULTS / "failure.json", failure)
        state("failed", failure=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "verify", "run"))
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "preflight":
        with gpu_lease():
            native_preflight()
    elif args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    else:
        run(args)
