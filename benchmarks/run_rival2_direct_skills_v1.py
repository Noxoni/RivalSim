"""Single-policy direct-skill/natural-play PPO; frozen authority then monitored run."""

# Long literal authority descriptions are hashed and kept intact for review.
# ruff: noqa: E402, E501
from __future__ import annotations

import argparse
import gc
import json
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

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_rival2_ssl_entity_full_match import (
    OVERTIME_CAP_TICKS,
    REGULATION_TICKS,
    CandidateMatchRunner,
    summarize,
)
from benchmarks.evaluate_rival2_ssl_entity_full_match import (
    spec as match_spec,
)
from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    append_json,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease, text_sha
from benchmarks.run_rival2_ssl_entity_joint_control import COLLISION
from rivalsim.direct_skills_training import DirectSkillCollector
from rivalsim.direct_skills_v1 import (
    NAMES,
    SEED,
    VERSION,
    DirectSkillsEnv,
    reward_authority,
    scenarios,
)
from rivalsim.fresh_ground_30hz import content_hash, ppo_config, scenario_hash
from rivalsim.ssl_entity_mixed_training import mixed_joint_ppo_update, mixed_sequence_data
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from rivalsim.ssl_entity_training import (
    finite_model_and_optimizer,
    fresh_entity_optimizer,
    joint_sequence_loss,
)
from rivalsim.ssl_joint_control_policy import categorical_statistics
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

RESULTS = ROOT / "results/rival2/direct_skills_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/direct_skills_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-v1")
PARENT = CHECKPOINTS / "parent_entity_000293.pt"
PARENT_SHA = "01CD1D075C3319D19FF607C0498DFFF6FE6335BE70EEA9708B2679884601DCBE"
MATCH_RESET_VERSION = "RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2"
SOURCES = (
    "benchmarks/run_rival2_direct_skills_v1.py",
    "benchmarks/direct_skills_eval_stream.py",
    "tests/test_direct_skills_eval_stream.py",
    "rivalsim/direct_skills_v1.py",
    "rivalsim/direct_skills_training.py",
    "tests/test_direct_skills_v1.py",
    "tests/test_direct_skills_native.py",
    "rivalsim/fresh_ground_30hz.py",
    "rivalsim/ssl_entity_mixed_training.py",
    "rivalsim/ssl_entity_training.py",
    "rivalsim/ssl_entity_policy.py",
    "rivalsim/ssl_joint_control_policy.py",
    "rivalsim/rival2_recurrent_ppo.py",
    "rivalsim/rival2_independent_critic.py",
    "rivalsim/rival2_env.py",
    "rivalsim/kernels/rival2.py",
    "rivalsim/ssl_foundation_v1.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "rivalsim/full_match.py",
    "third_party/nexto/adapter.py",
)


def authority():
    return dict(
        version=VERSION,
        parent=PARENT.relative_to(ROOT).as_posix(),
        parent_sha256=PARENT_SHA,
        parent_entity_update=293,
        parent_optimizer_steps=52150,
        initialization="Same parent weights, Adam moments and counters; new declared reward/curriculum lineage; not fresh random or BC",
        model="Unchanged entity attention recurrent joint90; native182 observation, no task ID, one deployed policy",
        reward=reward_authority(),
        seed=SEED,
        worlds=32768,
        ppo=asdict(ppo_config()),
        critic_lr=0.0003,
        kl="Telemetry only; no KL rejection, rollback or retention objective",
        safety="Finite model/gradient/Adam checks and whole-update corruption rollback preserved",
        opponents="Fixed even world slots Nexto, odd current self-play. Exactly one third of learner decisions against Nexto; current learner is task focal side vs Nexto. Both current agents train in self-play",
        nexto="Pinned15Hz inference/stock kickoff; no Rival scripted action or prefix",
        advantages="Existing opponent-family-local normalization; no hidden task ID injected into actor or critic",
        rollout="90 decisions/3seconds,4physics ticks each, complete recurrent sequences; goal resets hidden; truncate bootstrap before reset",
        snapshots="Every accepted update rolling; permanent at0,10,25,50 and each50 afterward",
        evaluations="Fixed64cases per family against actual Nexto at0,10,25,50 and every50; fixed ten regulation Nexto matches at0 and every50",
        match_method={k: v for k, v in match_spec().items() if k not in ("version", "checkpoints")},
        evaluation_seed=SEED + 1000,
        policy_sampling="Categorical during training; deterministic argmax in evaluation",
        selection="No cherry-picked checkpoint; scheduled snapshots, compare parent0 on identical cases",
        maximum_updates=None,
        deadline=None,
        stop="User stop or operational/nonfinite/corruption fault; report failures and do not silently change frozen reward coefficients",
        review="If first50 updates do not improve task outcomes and natural play, diagnose before another large unchanged block",
        resume="Same lineage model/Adam/counters/RNG; fresh physical episodes and zero hidden, fixed opponent slots, task focal side reloaded from new episode",
        no_bc_or_mechanic_rewards=True,
    )


def verify(published=True):
    package = json.loads((RESULTS / "package.json").read_text())
    assert json.loads((RESULTS / "authority.json").read_text()) == authority()
    assert package["authority_sha256"] == content_hash(authority())
    assert sha(PARENT) == PARENT_SHA
    for path, digest in package["sources"].items():
        assert text_sha(ROOT / path) == digest, path
    for name in package["evidence"]:
        assert text_sha(RESULTS / name) == package["evidence"][name]
    if published:
        paths = [
            *package["sources"],
            PARENT.relative_to(ROOT).as_posix(),
            *[
                (RESULTS / n).relative_to(ROOT).as_posix()
                for n in ("authority.json", "package.json", *package["evidence"])
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


def load_model_optimizer(payload):
    model = EntityJointControlActorCritic().cuda()
    model.load_state_dict(payload["model"], strict=True)
    assert payload["policy_config_sha256"] == model.config.content_hash
    optimizer = fresh_entity_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    assert {int(s["step"]) for s in optimizer.state.values()} == {
        payload["cumulative_optimizer_steps"]
    }
    assert finite_model_and_optimizer(model, optimizer)
    return model, optimizer


def preflight():
    assert sha(PARENT) == PARENT_SHA
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    model, optimizer = load_model_optimizer(payload)
    before, adam_before = (
        tensor_hash(model.state_dict()),
        tensor_hash(
            {
                f"{i}.{k}": v
                for i, state in optimizer.state_dict()["state"].items()
                for k, v in state.items()
            }
        ),
    )
    bank = scenarios(32768)
    env = DirectSkillsEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = DirectSkillCollector(env, model, seed=SEED)
    nexto_before = tensor_hash(collector.nexto.actor.state_dict())
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rollout = collector.collect()
    elapsed = time.monotonic() - started
    data = mixed_sequence_data(rollout, ppo_config())
    index = data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()  # cuDNN recurrent backward needs training-mode reserve buffers.
    loss, _ = joint_sequence_loss(model, data, index, ppo_config())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.isolated_value(data["observations"][index]).sum().backward()
    isolated = all(
        p.grad is None or not bool(p.grad.count_nonzero())
        for name, p in model.named_parameters()
        if not name.startswith("critic.")
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        logits, values, _ = model(
            data["observations"][index],
            data["initial_hidden"][:, index],
            reset_before=data["reset_before"][index],
        )
        logp, _ = categorical_statistics(logits, data["action_indices"][index])
        mask = data["train_mask"][index]
        logp_error = float((logp[mask] - data["old_log_probability"][index][mask]).abs().max())
        value_error = float((values[mask] - data["values"][index][mask]).abs().max())
    stats = collector.last_metrics
    checks = dict(
        full_scale=rollout.num_envs == 32768 and rollout.horizon == 90,
        exact_nexto_sample_third=stats["nexto_training_sample_count"] * 3
        == stats["trainable_agent_samples"],
        sample_count=stats["trainable_agent_samples"] == 32768 * 90 * 3 // 2,
        critic_isolated=isolated,
        finite_gradient=bool(torch.isfinite(norm)),
        finite_model_adam=finite_model_and_optimizer(model, optimizer),
        logp_parity=logp_error <= 1e-5,
        value_parity=value_error <= 1e-5,
        exact_actions=torch.equal(
            model.action_table[rollout.action_indices][rollout.train_mask],
            rollout.actions[rollout.train_mask],
        ),
        parent_unchanged=sha(PARENT) == PARENT_SHA and tensor_hash(model.state_dict()) == before,
        adam_unchanged=adam_before
        == tensor_hash(
            {
                f"{i}.{k}": v
                for i, s in optimizer.state_dict()["state"].items()
                for k, v in s.items()
            }
        ),
        nexto_unchanged=nexto_before == tensor_hash(collector.nexto.actor.state_dict()),
        no_optimizer_step={int(s["step"]) for s in optimizer.state.values()} == {52150},
        no_mechanics_hotpath=env.world.gameplay_v3 is None and env.world.gameplay_120 is None,
        rewards_finite=bool(torch.isfinite(rollout.rewards).all()),
        all_roles_represented=all(
            any(
                v["samples"] > 0
                for k, v in stats["by_reward_role_and_opponent"].items()
                if k.startswith(role + "_")
            )
            for role in NAMES
        ),
    )
    result = dict(
        utc=utc(),
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        authority_sha256=content_hash(authority()),
        sources={p: text_sha(ROOT / p) for p in SOURCES},
        scenario_sha256=scenario_hash(bank),
        optimizer_steps=0,
        rollout_seconds=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        logp_error=logp_error,
        value_error=value_error,
        telemetry=stats,
    )
    write_json(RESULTS / "native_preflight.json", result)
    print(json.dumps(result), flush=True)
    assert all(checks.values()), checks


def prepare():
    assert not (RESULTS / "package.json").exists(), "Already frozen"
    pre = json.loads((RESULTS / "native_preflight.json").read_text())
    assert pre["verdict"] == "PASS" and pre["authority_sha256"] == content_hash(authority())
    assert pre["sources"] == {p: text_sha(ROOT / p) for p in SOURCES}
    suites = ET.parse(RESULTS / "focused_tests.xml").getroot().findall("testsuite")
    assert suites and all(
        int(s.get("failures", 0)) == int(s.get("errors", 0)) == int(s.get("skipped", 0)) == 0
        for s in suites
    )
    write_json(RESULTS / "authority.json", authority())
    write_json(
        RESULTS / "package.json",
        dict(
            authority_sha256=content_hash(authority()),
            sources=pre["sources"],
            scenario_sha256=pre["scenario_sha256"],
            parent_sha256=PARENT_SHA,
            evidence={
                n: text_sha(RESULTS / n) for n in ("native_preflight.json", "focused_tests.xml")
            },
        ),
    )


@torch.no_grad()
def skill_evaluation(model):
    result = {}
    before = tensor_hash(model.state_dict())
    model.eval()
    for family, name in enumerate(NAMES):
        n, seed = 64, SEED + 1000 + family
        bank = scenarios(n, seed, family_only=family)
        env = DirectSkillsEnv(
            n, COLLISION, device="cuda:0", seed=seed, ssl_foundation_scenarios=bank
        )
        rows = torch.arange(n, device="cuda:0")
        side = torch.as_tensor(bank.focal_side.astype("int64"), device="cuda:0")
        nexto = NextoPolicyAdapter(n, device=env.device)
        nexto.set_player_index(1 - side)
        alive = torch.ones(n, device=env.device, dtype=torch.bool)
        nexto.activate(alive)
        ns = NextoStateTensors.from_bridge(env.bridge)
        hidden = model.initial_hidden(n * 2)
        reset = torch.ones(n * 2, device=env.device, dtype=torch.bool)
        touches = torch.zeros(n, device=env.device)
        first = torch.full((n,), float("nan"), device=env.device)
        goals, concedes, duration = touches.clone(), touches.clone(), touches.clone()
        events = torch.zeros((n, 7), device=env.device)
        ending = torch.zeros(n, device=env.device, dtype=torch.int64)
        for tick in range(900 if family == 0 else 360):
            logits, hidden = model.forward_actor(
                env.observation.reshape(-1, 182), hidden, reset_before=reset
            )
            action = model.deterministic(logits).reshape(n, 2, 8)

            def provider(_, action=action, ns=ns, nexto=nexto, alive=alive, rows=rows, side=side):
                applied = action.clone()
                kickoff = (ns.ball_pos[:, 0] == 0) & (ns.ball_pos[:, 1] == 0)
                controls, _ = nexto.tick_action(ns, kickoff, active_mask=alive)
                applied[rows, 1 - side] = controls
                return applied

            tr = env.step_with_tick_actions(action, provider)
            contact = env.last_native["touch_count"][rows, side] * alive
            touches += contact
            first = torch.where(
                (contact > 0) & first.isnan(),
                (tick * 4 + env.last_native["first_touch_tick"][rows, side] + 1) / 120,
                first,
            )
            goals += alive & tr.terminated & (env.last_native["scoring_team"] == side)
            concedes += alive & tr.terminated & (env.last_native["scoring_team"] != side)
            events += env.last_skill["events"][rows, side] * alive[:, None]
            duration += alive / 30
            ending += alive * (tr.terminated.long() + 2 * tr.truncated.long())
            alive &= ~tr.reset_mask
            reset = tr.reset_mask[:, None].expand(-1, 2).reshape(-1)
            hidden = hidden.masked_fill(reset[None, :, None], 0)
            if not bool(alive.any()):
                break
        result[name] = dict(
            worlds=n,
            scenario_sha256=scenario_hash(bank),
            goals_for=int(goals.sum()),
            goals_against=int(concedes.sum()),
            touched_worlds=int(first.isfinite().sum()),
            touches=int(touches.sum()),
            touches_per_minute=float(touches.sum() / duration.sum() * 60),
            conditional_first_touch_seconds=float(first[first.isfinite()].mean())
            if bool(first.isfinite().any())
            else None,
            goal_endings=int((ending == 1).sum()),
            truncation_endings=int((ending == 2).sum()),
            unfinished=int(alive.sum()),
            events=dict(
                zip(reward_authority()["events"], events.sum(0).cpu().tolist(), strict=True)
            ),
            raw=dict(
                touches=touches.cpu().tolist(),
                goals=goals.cpu().tolist(),
                concedes=concedes.cpu().tolist(),
                seconds=duration.cpu().tolist(),
                endings=ending.cpu().tolist(),
            ),
            event_semantics="Task proxy predicates, not independent possession/save/fake adjudication",
        )
        print(
            "SKILL_EVAL "
            + name
            + " "
            + json.dumps({k: v for k, v in result[name].items() if k != "raw"}),
            flush=True,
        )
        del env, nexto
        gc.collect()
    assert tensor_hash(model.state_dict()) == before
    return result


def scheduled(offset):
    return offset in (0, 10, 25) or offset % 50 == 0


def run(args):
    package = verify()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    assert not (EXTERNAL / "STOP").exists(), "STOP present"
    assert args.resume or not (EXTERNAL / "latest.json").exists(), "Use explicit latest resume"
    source = Path(args.resume) if args.resume else PARENT
    expected = args.resume_sha256 if args.resume else PARENT_SHA
    assert expected and sha(source) == expected.upper()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if args.resume:
        assert payload["format"] == VERSION + "_CHECKPOINT"
        assert (
            payload["authority_sha256"] == content_hash(authority())
            and payload["parent_sha256"] == PARENT_SHA
        )
    model, optimizer = load_model_optimizer(payload)
    offset = payload["accepted_updates"] if args.resume else 0
    samples = payload["direct_skill_samples"] if args.resume else 0
    ticks = payload["direct_skill_physics_ticks"] if args.resume else 0
    adam_steps = payload["cumulative_optimizer_steps"]
    bank = scenarios(32768)
    assert scenario_hash(bank) == package["scenario_sha256"]
    env = DirectSkillsEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = DirectSkillCollector(env, model, seed=SEED)
    collector.generator.set_state(payload["policy_generator_state"].cpu())
    shuffle = torch.Generator(device="cuda:0")
    shuffle.set_state(payload["shuffle_generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu())
    latest = None

    def state(status, **kwargs):
        write_json(
            EXTERNAL / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                status=status,
                accepted_updates=offset,
                cumulative_optimizer_steps=adam_steps,
                latest_checkpoint=latest,
                **kwargs,
            ),
        )

    def save(path):
        p = dict(payload)
        p.update(
            format=VERSION + "_CHECKPOINT",
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            authority_sha256=content_hash(authority()),
            package=package,
            parent_sha256=PARENT_SHA,
            accepted_updates=offset,
            direct_skill_samples=samples,
            direct_skill_physics_ticks=ticks,
            new_agent_samples=1676472162 + samples,
            new_physics_ticks=3456368640 + ticks,
            source_entity_updates=293,
            cumulative_optimizer_steps=adam_steps,
            fresh_optimizer=False,
            optimizer_lineage="Entity293 Adam preserved; direct skill reward/curriculum authority replaces old experiment",
            runtime_contract_hashes=env.contract_hashes,
            reward_authority=reward_authority(),
            policy_generator_state=collector.generator.get_state(),
            shuffle_generator_state=shuffle.get_state(),
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(),
            last_training_metrics=collector.last_metrics,
            resume_count=payload.get("resume_count", 0) + 1,
            scenario_reset_note=authority()["resume"],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        with tmp.open("wb") as f:
            torch.save(p, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return dict(path=str(path), sha256=sha(path), accepted_updates=offset)

    def checkpoint():
        record = save(EXTERNAL / f"rolling_{offset % 2}.pt")
        write_json(EXTERNAL / "latest.json", record)
        return record

    def evaluate():
        state("evaluating")
        frozen = CHECKPOINTS / f"plus_{offset:06d}.pt"
        if frozen.exists():
            identity = dict(path=str(frozen), sha256=sha(frozen), accepted_updates=offset)
            existing = torch.load(frozen, map_location="cpu", weights_only=False)
            assert tensor_hash(existing["model"]) == tensor_hash(model.state_dict())
            del existing
        else:
            identity = save(frozen)
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        path = RESULTS / f"evaluation_{offset:06d}.json"
        if not path.exists():
            skill = skill_evaluation(model)
            write_json(
                path,
                dict(
                    utc=utc(),
                    accepted_updates=offset,
                    checkpoint=identity,
                    authority_sha256=content_hash(authority()),
                    skills=skill,
                    optimizer_steps=0,
                ),
            )
        match_path = RESULTS / f"full_match_{offset:06d}.json"
        if offset % 50 == 0 and not match_path.exists():
            # Same predeclared method, not a new candidate selection or training opponent.
            with owned_match_stream():
                runner = CandidateMatchRunner(frozen, identity["sha256"], entity=True)
                elapsed = runner.run_ticks(REGULATION_TICKS).seconds
                for _ in range(OVERTIME_CAP_TICKS // 600):
                    if bool(runner.phase_status()["done"].all()):
                        break
                    elapsed += runner.run_ticks(600).seconds
                raw = runner.export()["raw"]
                assert not bool(raw["goal_overflow"].any())
                assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
                write_json(
                    match_path,
                    dict(
                        utc=utc(),
                        accepted_updates=offset,
                        checkpoint=identity,
                        authority_sha256=content_hash(authority()),
                        match_reset_version=MATCH_RESET_VERSION,
                        runtime_package_sha256=content_hash(package),
                        summary=summarize(raw),
                        raw={k: v.tolist() for k, v in raw.items()},
                        wall_seconds=elapsed,
                        hidden_resets=runner.hidden_reset_count.cpu().tolist(),
                        optimizer_steps=0,
                        model_unchanged=True,
                        checkpoint_unchanged=sha(frozen) == identity["sha256"],
                    ),
                )
                print("MATCH_EVAL " + json.dumps(summarize(raw)), flush=True)
                del runner
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        latest = checkpoint()
        if scheduled(offset):
            evaluate()
        while not (EXTERNAL / "STOP").exists():
            state("rollout")
            started = time.monotonic()
            rollout = collector.collect()
            rollout_seconds = time.monotonic() - started
            state("optimizing")
            started = time.monotonic()
            ppo = mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic() - started
            offset += 1
            adam_steps += ppo["optimizer_steps"]
            samples += collector.last_metrics["trainable_agent_samples"]
            ticks += collector.last_metrics["physical_physics_ticks"]
            del rollout
            latest = checkpoint()
            row = dict(
                utc=utc(),
                accepted_updates=offset,
                cumulative_optimizer_steps=adam_steps,
                samples=samples,
                physical_ticks=ticks,
                ppo=ppo,
                training=collector.last_metrics,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
            )
            append_json(EXTERNAL / "training_curve.jsonl", row)
            print("ACCEPTED " + json.dumps(row), flush=True)
            if scheduled(offset):
                evaluate()
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
        state("failed", failure=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "verify", "run"))
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    else:
        with gpu_lease():
            preflight() if args.command == "preflight" else run(args)
