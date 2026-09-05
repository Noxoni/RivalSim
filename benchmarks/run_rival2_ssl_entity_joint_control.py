"""Prospectively frozen entity/joint-control PPO pilot, from fresh-lineage u597."""
# ruff: noqa: E402 -- Direct script invocation must establish repository imports.

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import SOURCES as BASE_SOURCES
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
    ppo_config,
    scenario_hash,
    scenarios,
)
from rivalsim.fresh_ground_30hz import (
    authority as base_authority,
)
from rivalsim.fresh_ground_30hz_training import evaluate
from rivalsim.rival2_contracts import ACTION_NAMES, OBSERVATION_SCHEMA_HASH
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentPPOCorruption
from rivalsim.ssl_entity_evaluation import DeterministicEvaluationView
from rivalsim.ssl_entity_policy import ENTITY_VERSION, EntityJointControlActorCritic, entity_schema
from rivalsim.ssl_entity_training import (
    EntityRolloutCollector,
    fresh_entity_optimizer,
    joint_ppo_update,
)

RESULTS = ROOT / "results/rival2/ssl_entity_joint_control_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/ssl_entity_joint_control_v1"
INITIAL = CHECKPOINTS / "initialized.pt"
EXTERNAL = Path("G:/dev/RivalSim-runs/ssl-entity-joint-control-v1")
COLLISION = "G:/dev/RLBot-Rival/bot/collision_meshes"
BOUNDARIES = (0, 10, 20, 50, 100)
SOURCES = [
    *BASE_SOURCES,
    "rivalsim/ssl_joint_control_policy.py",
    "rivalsim/ssl_entity_policy.py",
    "rivalsim/ssl_entity_training.py",
    "rivalsim/ssl_entity_evaluation.py",
    "benchmarks/run_rival2_ssl_entity_joint_control.py",
    "benchmarks/validate_rival2_ssl_entity_policy.py",
    "tests/test_ssl_joint_control_policy.py",
    "tests/test_ssl_entity_policy.py",
    "tests/test_ssl_entity_training.py",
    "rivalsim/kernels/boost_pad.py",
]


def authority():
    return dict(
        version=ENTITY_VERSION,
        parent_path=str(PARENT.relative_to(ROOT)),
        parent_sha256=PARENT_SHA,
        parent_ppo_update=597,
        initialize_old_v5_or_bc=False,
        accepted_update_budget=100,
        evaluations=list(BOUNDARIES),
        worlds=32768,
        policy_hz=30,
        physics_hz=120,
        hold_ticks=4,
        ppo=asdict(ppo_config()),
        reward_authority_sha256=content_hash(base_authority()),
        reward_changes=False,
        scenario_seed=SEED,
        scenario="Original fresh 30Hz bank, not easier reset-only probe.",
        opponents="Pure current self-play; both players train. Nexto only in evaluation.",
        policy=entity_schema(),
        action_contract=dict(
            version="RIVAL2_ACTION_JOINT90_30HZ_V1",
            external_channels=list(ACTION_NAMES),
            distribution="90-way categorical over pinned standard lookup table",
            physics_hz=120,
            policy_hz=30,
            hold_ticks=4,
            hybrid_actor_contract_reused=False,
            ground_or_air_action_mask=False,
            gaussian_sampling=False,
            bernoulli_sampling=False,
            initial_head_projection="Fixed sigma .65 preference projection of u597 "
            "mean/button logits, not exact policy migration.",
        ),
        optimizer="Fresh Adam for changed actor/attention parameterization and independent "
        "critic; no moment reuse or projection.",
        learning_rates=dict(actor=1e-4, critic=3e-4),
        initialization_seed=SEED + 701,
        observation_schema_sha256=OBSERVATION_SCHEMA_HASH,
        previous_action="Unchanged fresh-30Hz native previous decision history; "
        "not the older BC masking lane.",
        safety="KL telemetry only; full model/Adam/shuffle RNG rollback on numerical "
        "corruption; preserve last accepted checkpoint and stop.",
        resume="Verified same-candidate model/Adam/RNG/counters; native scenarios and hidden "
        "restart fresh, explicitly not an exact world-state continuation.",
        checkpoints="Rolling each accepted update, immutable at 0/10/20/50/100.",
        evaluation="Unchanged original 64-case acquisition/finishing/Nexto development "
        "evaluations through lossless deterministic interface. Argmax, no stochastic "
        "actions. No held-out test reuse.",
        interpretation="Combined architecture/control/optimizer experiment, not a causal "
        "one-factor ablation. Compare every boundary with initial candidate and immutable "
        "u597. Judge acquisition coverage, no-touch and scoring together. Do not promote "
        "based on training health or lucky intermediate outcome.",
        next_review="At100 preserve the whole curve and choose the next scoped "
        "experiment/continuation from evidence. No silent extension under this authority; "
        "SSL goal remains active.",
        new_human_demonstrations=False,
        named_rewards=False,
        no_online_ranked_automation=True,
    )


def initialized_model():
    assert sha(PARENT) == PARENT_SHA
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    torch.manual_seed(SEED + 701)
    model = EntityJointControlActorCritic()
    model.initialize_from_hybrid(parent["model"])
    return model, parent


def prepare():
    check_package(require_preflight=True)
    if (RESULTS / "package.json").exists() or INITIAL.exists():
        raise RuntimeError("candidate already frozen; do not overwrite")
    preflight = json.loads((RESULTS / "native_preflight.json").read_text())
    assert preflight["verdict"] == "PASS"
    suites = ET.parse(RESULTS / "focused_tests.xml").getroot().findall("testsuite")
    assert suites and all(
        int(s.get("failures", "0")) == 0 and int(s.get("errors", "0")) == 0 for s in suites
    )
    model, parent = initialized_model()
    assert tensor_hash(model.state_dict()) == preflight["initialization_sha256"]
    bank = scenarios(32768)
    assert scenario_hash(bank) == preflight["scenario_sha256"]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format=ENTITY_VERSION + "_CHECKPOINT",
        authority_sha256=content_hash(authority()),
        model=model.state_dict(),
        optimizer=fresh_entity_optimizer(model).state_dict(),
        policy_config=asdict(model.config),
        policy_config_sha256=model.config.content_hash,
        ppo_config_sha256=ppo_config().content_hash,
        parent_sha256=PARENT_SHA,
        accepted_updates=0,
        new_agent_samples=0,
        new_physics_ticks=0,
        parent_agent_samples=parent["total_agent_samples"],
        parent_ppo_updates=597,
        policy_generator_state=parent["policy_generator_state"].cpu(),
        shuffle_generator_state=parent["shuffle_generator_state"].cpu(),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_state=torch.cuda.get_rng_state(),
        resume_count=0,
        observation_schema_sha256=OBSERVATION_SCHEMA_HASH,
        action_contract=authority()["action_contract"],
        entity_schema=entity_schema(),
        fresh_optimizer=True,
        no_v5_bc_lineage=True,
    )
    torch.save(payload, INITIAL)
    write_json(RESULTS / "authority.json", authority())
    write_json(
        RESULTS / "package.json",
        dict(
            utc=utc(),
            sources={n: sha(ROOT / n) for n in SOURCES},
            authority_sha256=content_hash(authority()),
            initial_checkpoint_sha256=sha(INITIAL),
            initial_model_sha256=tensor_hash(model.state_dict()),
            native_preflight_sha256=sha(RESULTS / "native_preflight.json"),
            focused_tests_sha256=sha(RESULTS / "focused_tests.xml"),
            scenario_sha256=scenario_hash(bank),
            action_table_sha256=tensor_hash({"action_table": model.action_table}),
        ),
    )
    print(json.dumps(dict(status="frozen_requires_push", sha256=sha(INITIAL))), flush=True)


def verify(published=False):
    check_package(require_preflight=True)
    package = json.loads((RESULTS / "package.json").read_text())
    assert json.loads((RESULTS / "authority.json").read_text()) == authority()
    assert sha(PARENT) == PARENT_SHA and sha(INITIAL) == package["initial_checkpoint_sha256"]
    assert sha(RESULTS / "native_preflight.json") == package["native_preflight_sha256"]
    assert sha(RESULTS / "focused_tests.xml") == package["focused_tests_sha256"]
    for name, digest in package["sources"].items():
        assert sha(ROOT / name) == digest, name
    if published:
        for name in [
            *SOURCES,
            *[
                f"results/rival2/ssl_entity_joint_control_v1/{s}"
                for s in (
                    "authority.json",
                    "package.json",
                    "native_preflight.json",
                    "focused_tests.xml",
                )
            ],
            str(INITIAL.relative_to(ROOT)).replace("\\", "/"),
        ]:
            blob = subprocess.check_output(["git", "show", f"origin/main:{name}"], cwd=ROOT)
            local = (ROOT / name).read_bytes()
            assert (
                blob == local
                if name.endswith(".pt")
                else blob.replace(b"\r\n", b"\n") == local.replace(b"\r\n", b"\n")
            ), name
    return package


def run(args):
    package = verify(published=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if (EXTERNAL / "STOP").exists():
        raise RuntimeError("STOP present")
    if (EXTERNAL / "latest.json").exists() and not args.resume:
        raise RuntimeError("Existing candidate requires explicit same-lineage resume")
    import msvcrt

    lease = (EXTERNAL / "campaign.lock").open("a+b")
    lease.seek(0)
    if not lease.read(1):
        lease.write(b"0")
        lease.flush()
    lease.seek(0)
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    path = Path(args.resume) if args.resume else INITIAL
    if args.resume:
        assert args.resume_sha256 and sha(path) == args.resume_sha256.upper()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == ENTITY_VERSION + "_CHECKPOINT"
    assert (
        payload["authority_sha256"] == content_hash(authority())
        and payload["parent_sha256"] == PARENT_SHA
    )
    model, _ = initialized_model()
    assert payload["policy_config_sha256"] == model.config.content_hash
    assert payload["ppo_config_sha256"] == ppo_config().content_hash
    model.load_state_dict(payload["model"], strict=True)
    bank = scenarios(32768)
    assert scenario_hash(bank) == package["scenario_sha256"]
    env = FreshGroundEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = EntityRolloutCollector(env, model)
    optimizer = fresh_entity_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    collector.generator.set_state(payload["policy_generator_state"].cpu())
    shuffle = torch.Generator(device=env.device)
    shuffle.set_state(payload["shuffle_generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu())
    offset = payload["accepted_updates"]
    new_samples = payload["new_agent_samples"]
    new_ticks = payload["new_physics_ticks"]
    latest = None

    def state(status, **kwargs):
        write_json(
            EXTERNAL / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                status=status,
                accepted_updates=offset,
                latest_checkpoint=latest,
                **kwargs,
            ),
        )

    def save(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        p = dict(payload)
        p.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            accepted_updates=offset,
            new_agent_samples=new_samples,
            new_physics_ticks=new_ticks,
            policy_generator_state=collector.generator.get_state(),
            shuffle_generator_state=shuffle.get_state(),
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            resume_count=payload["resume_count"] + int(bool(args.resume)),
            package=package,
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
        state("evaluating")
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        view = DeterministicEvaluationView(model)
        report = evaluate(view, COLLISION)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        report.update(
            utc=utc(),
            accepted_updates=offset,
            checkpoint=latest,
            authority_sha256=content_hash(authority()),
        )
        write_json(RESULTS / f"evaluation_{offset:03d}.json", report)
        print("EVALUATION " + json.dumps(report), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        latest = checkpoint()
        if offset == 0 and not (RESULTS / "evaluation_000.json").exists():
            evaluation()
        while offset < 100 and not (EXTERNAL / "STOP").exists():
            state("rollout")
            started = time.monotonic()
            rollout = collector.collect()
            rollout_seconds = time.monotonic() - started
            state("optimizing")
            started = time.monotonic()
            metrics = joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic() - started
            offset += 1
            new_samples += collector.last_metrics["trainable_agent_samples"]
            new_ticks += collector.last_metrics["physical_physics_ticks"]
            del rollout
            latest = checkpoint()
            row = dict(
                utc=utc(),
                accepted_updates=offset,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
                training=collector.last_metrics,
                ppo={
                    k: (v if not isinstance(v, float) or math.isfinite(v) else str(v))
                    for k, v in metrics.items()
                },
            )
            append_json(EXTERNAL / "training_curve.jsonl", row)
            append_json(RESULTS / "training_curve.jsonl", row)
            print("ACCEPTED " + json.dumps(row), flush=True)
            if offset in BOUNDARIES:
                save(CHECKPOINTS / f"plus_{offset:03d}.pt")
                evaluation()
        state("completed" if offset == 100 else "stopped_at_accepted_boundary")
    except Exception as exc:
        failure = dict(
            utc=utc(),
            accepted_updates=offset,
            latest_checkpoint=latest,
            exception=repr(exc),
            traceback=traceback.format_exc(),
            classification="numerical_corruption"
            if isinstance(exc, Rival2RecurrentPPOCorruption)
            else "requires_operational_audit",
            automatic_retry=False,
        )
        write_json(EXTERNAL / "failure.json", failure)
        write_json(RESULTS / "failure.json", failure)
        state("failed", failure=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify(published=True)
    else:
        run(args)
