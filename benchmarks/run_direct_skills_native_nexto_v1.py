"""Corrected-opponent learning block; exact preserved parent, no legacy overwrite."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks import evaluate_direct_skills_native_nexto_v1 as evaluation
from benchmarks import run_rival2_direct_skills_v1 as base
from benchmarks.run_direct_skills_shooting_progress_v1 import load, adam_hash
from benchmarks.run_rival2_fresh_ground_30hz_v1 import append_json, sha, tensor_hash, utc, write_json
from rivalsim.direct_skills_exploration_v1 import TEMPERATURE
from rivalsim.direct_skills_finishing_goal_v2 import FinishingGoalEnv, reward_authority
from rivalsim.direct_skills_native_nexto import DirectSkillNativeNextoCollector, COLLECTOR_VERSION
from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios, curriculum_authority
from rivalsim.fresh_ground_30hz import content_hash, ppo_config, scenario_hash
from third_party.nexto.native_v5 import VERSION as CONTROLLER

VERSION = "RIVAL2_DIRECT_SKILLS_NATIVE_NEXTO_CAMPAIGN_V1"
OUT = evaluation.OUT
CKPTS = ROOT / "checkpoints/rival2/direct_skills_native_nexto_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-native-nexto-v1")
LIMIT, EVALUATIONS, NEXTO_SEED = 25, (10, 25), 2026090691
RNG = ("policy_generator_state", "shuffle_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_state")
SOURCES = (
    "benchmarks/run_direct_skills_native_nexto_v1.py", "tests/test_direct_skills_native_nexto_runner.py",
    "benchmarks/run_rival2_direct_skills_v1.py", "benchmarks/run_direct_skills_shooting_progress_v1.py",
    "rivalsim/direct_skills_native_nexto.py", "rivalsim/direct_skills_training.py",
    "rivalsim/direct_skills_finishing_goal_v2.py", "rivalsim/direct_skills_v1.py",
    "rivalsim/direct_skills_shooting_curriculum_v1.py", "rivalsim/direct_skills_exploration_v1.py",
    "rivalsim/ssl_entity_training.py", "rivalsim/ssl_entity_mixed_training.py",
    "rivalsim/rival2_recurrent_ppo.py", "rivalsim/fresh_ground_30hz.py",
)


def selection():
    result = json.loads((OUT / "selection.json").read_text())
    assert result["checkpoint"] == evaluation.CANDIDATES[result["selected_name"]]
    candidates = {name: json.loads((OUT / (name + ".json")).read_text()) for name in evaluation.CANDIDATES}
    assert evaluation.select_parent(candidates) == result["selected_name"]
    for name in candidates:
        assert result["results"][name]["file_sha256"] == sha(OUT / (name + ".json"))
        assert all(evaluation.reduce(OUT / (name + ".json"))["integrity"].values())
    assert sha(ROOT / result["checkpoint"]["path"]) == result["checkpoint"]["sha256"]
    return result


def authority():
    picked = selection()
    return dict(version=VERSION, parent=picked["checkpoint"], selection_sha256=sha(OUT / "selection.json"),
        parent_choice=picked["selected_name"], worlds=32768, child_updates=LIMIT,
        evaluation_boundaries=list(EVALUATIONS), physics_hz=120, policy_hz=30,
        ppo=asdict(ppo_config()), training_temperature=TEMPERATURE,
        opponents=dict(collector=COLLECTOR_VERSION, controller=CONTROLLER, sampling_mode="native_v5",
                       seed=NEXTO_SEED, nexto_world_fraction=.5, nexto_learner_sample_fraction=1/3),
        reward=reward_authority(), reward_sha256=content_hash(reward_authority()),
        curriculum=curriculum_authority(), scenario_seed=base.SEED,
        architecture_change=False, fresh_optimizer=False,
        initialization="Selected preserved model/Adam/counters/four RNG exactly restored; fresh physical episodes and zero recurrent hidden. New native-controller RNG starts from frozen seed; legacy opponent caches are not reused.",
        resume="Same new lineage, exact latest accepted file/hash only. Model/Adam/four RNG and native-opponent RNG restored; new physical episodes/zero hidden, controller caches reset explicitly. Not exact physical-world replay.",
        changes="Corrected native Nexto controller for training and evaluation. Keep latest finishing-goal reward, existing pressure bank, T2 collection/likelihood and PPO semantics. If reference600 selected, it enters the already accepted finishing/T2 curriculum; no new reward design.",
        safety="Existing finite model/gradient/Adam checks and transactional corruption rollback. KL telemetry ONLY; no KL rejection or retention objective.",
        checkpoints="Entry0/first1/permanent10,25 and alternating durable latest after every accepted update. Preserve all old parents.",
        evaluation_spec_sha256=content_hash(evaluation.specification()),
        end="Review after25, not achievement of SSL. No automatic extension of this finite block; ongoing goal continues from measured evidence.")


def make_collector(env, model):
    return DirectSkillNativeNextoCollector(env, model, seed=base.SEED,
        nexto_sampling_mode="native_v5", nexto_seed=NEXTO_SEED)


def preflight():
    evaluation.verify()
    parent = selection()["checkpoint"]
    payload = torch.load(ROOT / parent["path"], map_location="cpu", weights_only=False)
    model, optimizer = load(payload)
    before, adam_before = tensor_hash(model.state_dict()), adam_hash(optimizer)
    env = FinishingGoalEnv(1024, base.COLLISION, device="cuda:0", seed=base.SEED,
                          ssl_foundation_scenarios=scenarios(1024))
    collector = make_collector(env, model)
    rollout = collector.collect()
    data = base.mixed_sequence_data(rollout, ppo_config())
    indices = data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()
    loss, _ = base.joint_sequence_loss(model, data, indices, ppo_config())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.isolated_value(data["observations"][indices]).sum().backward()
    isolated = all(p.grad is None or not bool(p.grad.any()) for name,p in model.named_parameters()
                   if not name.startswith("critic."))
    optimizer.zero_grad(set_to_none=True)
    checks = dict(parent_exact=sha(ROOT / parent["path"]) == parent["sha256"],
        model_unchanged=before == tensor_hash(model.state_dict()), adam_unchanged=adam_before == adam_hash(optimizer),
        optimizer_counters_unchanged={int(s["step"]) for s in optimizer.state.values()} == {payload["cumulative_optimizer_steps"]},
        finite_loss_gradient=bool(torch.isfinite(loss) & torch.isfinite(norm)), critic_isolated=isolated,
        finite_model_adam=base.finite_model_and_optimizer(model, optimizer),
        action_targets_exact=torch.equal(model.action_table[rollout.action_indices][rollout.train_mask], rollout.actions[rollout.train_mask]),
        nexto_fraction_exact=collector.last_metrics["nexto_training_sample_count"]*3 == collector.last_metrics["trainable_agent_samples"],
        native_controller=collector.nexto.checkpoint_state()["version"] == CONTROLLER,
        finishing_reward_identity=env.contract_hashes["reward"] == content_hash(reward_authority()),
        finishing_retired_bonuses_zero=not bool(env.last_skill["weighted"][env.last_skill["roles"] == 2][...,:6].any()),
        finishing_retired_approach_zero=not bool(env.last_skill["approach"][env.last_skill["roles"] == 2].any()),
        no_mechanics_hotpath=env.world.gameplay_v3 is None and env.world.gameplay_120 is None)
    assert all(checks.values()), checks
    assert not (OUT / "training_preflight.json").exists()
    write_json(OUT / "training_preflight.json", dict(utc=utc(), checks=checks, optimizer_steps=0,
        gradient_norm=float(norm), parent=parent, worlds=1024, decisions=90,
        model_sha256=before, adam_sha256=adam_before, training=collector.last_metrics))
    print("PREFLIGHT " + json.dumps(checks), flush=True)


def freeze():
    import xml.etree.ElementTree as ET
    evaluation.verify()
    assert not (OUT / "training_package.json").exists()
    pre = json.loads((OUT / "training_preflight.json").read_text())
    assert all(pre["checks"].values()) and pre["optimizer_steps"] == 0
    suites = ET.parse(OUT / "training_tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    spec = authority()
    write_json(OUT / "training_authority.json", spec)
    write_json(OUT / "training_package.json", dict(authority_sha256=content_hash(spec),
        scenario_sha256=scenario_hash(scenarios(32768)),
        sources={p: evaluation.text_sha(ROOT / p) for p in SOURCES},
        evidence={n: sha(OUT / n) for n in ("selection.json", "training_preflight.json", "training_tests.xml")}))


def verify():
    evaluation.verify()
    package = json.loads((OUT / "training_package.json").read_text())
    assert json.loads((OUT / "training_authority.json").read_text()) == authority()
    assert package["authority_sha256"] == content_hash(authority())
    for p,h in package["sources"].items(): assert evaluation.text_sha(ROOT / p) == h, p
    for p,h in package["evidence"].items(): assert sha(OUT / p) == h, p
    for p in (*SOURCES, *(str((OUT / n).relative_to(ROOT).as_posix()) for n in
                         ("training_authority.json", "training_package.json", *package["evidence"]))):
        remote = subprocess.check_output(["git", "show", "origin/main:"+p], cwd=ROOT)
        assert remote.replace(b"\r\n", b"\n") == (ROOT / p).read_bytes().replace(b"\r\n", b"\n"), p
    return package


def validate_resume(path, digest, spec, external=EXTERNAL):
    if (external / "STOP").exists(): raise RuntimeError("Respect STOP")
    latest_path = external / "latest.json"
    if not latest_path.exists():
        if path is not None or digest is not None: raise RuntimeError("Fresh run must use selected parent")
        if (external / "campaign_state.json").exists(): raise RuntimeError("State without checkpoint needs audit")
        return ROOT / spec["parent"]["path"], spec["parent"]["sha256"], 0
    latest = json.loads(latest_path.read_text())
    state = json.loads((external / "campaign_state.json").read_text())
    step = latest["branch_updates"]
    if type(step) is not int or not 0 <= step <= LIMIT: raise RuntimeError("Outside frozen block")
    if path is None or not digest or digest.upper() != latest["sha256"]: raise RuntimeError("Explicit latest accepted resume required")
    if Path(path).resolve() != Path(latest["path"]).resolve() or sha(Path(path)) != digest.upper(): raise RuntimeError("Resume path/hash mismatch")
    if state["native_nexto_authority_sha256"] != content_hash(spec) or state["branch_updates"] not in (step, step-1):
        raise RuntimeError("Authority/counter discrepancy")
    if state["status"] in ("complete_review", "nonfinite_or_runtime_failure", "stopped_at_accepted_boundary"):
        raise RuntimeError("Stopped, completed or failed run requires explicit audit")
    return Path(path), digest.upper(), step


def run(args):
    package, spec = verify(), authority()
    source, digest, child = validate_resume(args.resume, args.resume_sha256, spec)
    assert sha(source) == digest
    payload = torch.load(source, map_location="cpu", weights_only=False)
    start = spec["parent"]["accepted_updates"]
    assert payload["accepted_updates"] == start + child
    if args.resume:
        assert payload["format"] == VERSION+"_CHECKPOINT"
        assert payload["native_nexto_authority_sha256"] == content_hash(spec)
        assert payload["native_nexto_parent_sha256"] == spec["parent"]["sha256"]
        assert payload["native_nexto_branch_updates"] == child
    model, optimizer = load(payload)
    samples, ticks, adam_steps = (payload[k] for k in ("direct_skill_samples", "direct_skill_physics_ticks", "cumulative_optimizer_steps"))
    bank = scenarios(32768)
    assert scenario_hash(bank) == package["scenario_sha256"]
    env = FinishingGoalEnv(32768, base.COLLISION, device="cuda:0", seed=base.SEED, ssl_foundation_scenarios=bank)
    collector = make_collector(env, model)
    if args.resume: collector.restore_opponent_for_fresh_episodes(payload["opponent_state"])
    collector.generator.set_state(payload[RNG[0]].cpu())
    shuffle = torch.Generator(device="cuda:0"); shuffle.set_state(payload[RNG[1]].cpu())
    torch.set_rng_state(payload[RNG[2]].cpu()); torch.cuda.set_rng_state(payload[RNG[3]].cpu())
    latest = None
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    nexto_hash = tensor_hash(collector.nexto.actor.state_dict())

    def state(status, **kwargs):
        write_json(EXTERNAL / "campaign_state.json", dict(utc=utc(), pid=os.getpid(), status=status,
            branch_updates=child, accepted_updates=start+child, cumulative_optimizer_steps=adam_steps,
            latest_checkpoint=latest, native_nexto_authority_sha256=content_hash(spec), review_boundary=LIMIT, **kwargs))

    def save(path, immutable=False):
        p = dict(payload)
        p.update(format=VERSION+"_CHECKPOINT", model=model.state_dict(), optimizer=optimizer.state_dict(),
            native_nexto_authority_sha256=content_hash(spec), native_nexto_parent_sha256=spec["parent"]["sha256"],
            native_nexto_package=package, native_nexto_branch_updates=child,
            reward_authority=reward_authority(), accepted_updates=start+child,
            direct_skill_samples=samples, direct_skill_physics_ticks=ticks,
            new_agent_samples=payload["new_agent_samples"] + samples-payload["direct_skill_samples"],
            new_physics_ticks=payload["new_physics_ticks"] + ticks-payload["direct_skill_physics_ticks"],
            cumulative_optimizer_steps=adam_steps, fresh_optimizer=False,
            runtime_contract_hashes=env.contract_hashes,
            policy_generator_state=collector.generator.get_state(), shuffle_generator_state=shuffle.get_state(),
            torch_cpu_rng_state=torch.get_rng_state(), torch_cuda_rng_state=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(), last_training_metrics=collector.last_metrics,
            evaluation_spec_sha256=content_hash(evaluation.specification()),
            training_distribution=dict(temperature=TEMPERATURE, on_policy_sampling_and_likelihood=True),
            resume_count=payload.get("resume_count", 0)+1,
            physical_resume_semantics=spec["resume"], effective_training_scenario_sha256=package["scenario_sha256"])
        if immutable and path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=False)
            assert stored["native_nexto_authority_sha256"] == content_hash(spec)
            assert stored["native_nexto_branch_updates"] == child
            assert tensor_hash(stored["model"]) == tensor_hash(p["model"])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".pt.tmp")
            with temp.open("wb") as f: torch.save(p, f); f.flush(); os.fsync(f.fileno())
            os.replace(temp, path)
        return dict(path=str(path), sha256=sha(path), accepted_updates=start+child, branch_updates=child)

    def checkpoint():
        identity = save(EXTERNAL / f"rolling_{child % 2}.pt")
        write_json(EXTERNAL / "latest.json", identity)
        return identity

    def evaluate():
        state("evaluating")
        identity = save(CKPTS / f"child_{child:06d}.pt", immutable=True)
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        evaluation.evaluate(Path(identity["path"]), identity["sha256"], OUT / f"child_{child:06d}.json",
            dict(native_nexto_authority_sha256=content_hash(spec), branch_updates=child))
        torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda)
        gc.collect(); torch.cuda.empty_cache()

    try:
        state("initializing")
        if child == 0:
            entry = save(CKPTS / "entry_000000.pt", immutable=True)
            stored = torch.load(entry["path"], map_location="cpu", weights_only=False)
            checks = dict(model_exact=tensor_hash(stored["model"]) == tensor_hash(payload["model"]),
                optimizer_exact=all(torch.equal(stored["optimizer"]["state"][i][k], v.cpu())
                    for i,s in payload["optimizer"]["state"].items() for k,v in s.items()),
                optimizer_groups_exact=stored["optimizer"]["param_groups"] == payload["optimizer"]["param_groups"],
                four_rng_exact=all(torch.equal(stored[k], payload[k].cpu()) for k in RNG),
                counters_exact=all(stored[k] == payload[k] for k in
                    ("accepted_updates", "direct_skill_samples", "direct_skill_physics_ticks", "cumulative_optimizer_steps")),
                parent_unchanged=sha(source) == digest,
                new_native_opponent=stored["opponent_state"]["controller_version"] == CONTROLLER)
            assert all(checks.values()), checks
            write_json(OUT / "entry_integrity.json", dict(utc=utc(), checks=checks, optimizer_steps=0, checkpoint=entry))
            del stored
        latest = checkpoint()
        # If interrupted after saving an evaluation boundary, finish its evaluation
        # before further training. The evaluator refuses a partially started case.
        if args.resume and child in EVALUATIONS: evaluate()
        while child < LIMIT and not (EXTERNAL / "STOP").exists():
            state("rollout"); torch.cuda.reset_peak_memory_stats()
            start_time = time.monotonic(); rollout = collector.collect(); rollout_seconds = time.monotonic()-start_time
            state("optimizing"); start_time = time.monotonic()
            ppo = base.mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic()-start_time
            child += 1; adam_steps += ppo["optimizer_steps"]
            samples += collector.last_metrics["trainable_agent_samples"]
            ticks += collector.last_metrics["physical_physics_ticks"]
            del rollout
            assert tensor_hash(collector.nexto.actor.state_dict()) == nexto_hash
            latest = checkpoint()
            if child == 1: save(CKPTS / "first_000001.pt", immutable=True)
            row = dict(utc=utc(), branch_updates=child, accepted_updates=start+child,
                cumulative_optimizer_steps=adam_steps, samples=samples, physical_ticks=ticks,
                ppo=ppo, training=collector.last_metrics, rollout_seconds=rollout_seconds, ppo_seconds=ppo_seconds,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), native_nexto_authority_sha256=content_hash(spec))
            append_json(EXTERNAL / "training_curve.jsonl", row)
            print("ACCEPTED " + json.dumps(dict(branch_updates=child, checkpoint=latest, ppo=ppo,
                  training=collector.last_metrics, rollout_seconds=rollout_seconds, ppo_seconds=ppo_seconds)), flush=True)
            if child in EVALUATIONS: evaluate()
        if child == LIMIT:
            state("complete_review")
        else:
            latest = save(CKPTS / f"paused_{child:06d}.pt", immutable=True)
            state("stopped_at_accepted_boundary")
    except Exception as exc:
        failure = dict(utc=utc(), exception=repr(exc), traceback=traceback.format_exc(),
                       latest_checkpoint=latest, branch_updates=child)
        write_json(EXTERNAL / "failure.json", failure)
        state("nonfinite_or_runtime_failure", failure=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preflight", "freeze", "verify", "run"))
    parser.add_argument("--resume", type=Path); parser.add_argument("--resume-sha256")
    args = parser.parse_args(); torch.set_num_threads(8)
    if args.mode == "freeze": freeze()
    elif args.mode == "verify": verify()
    else:
        with base.gpu_lease(): preflight() if args.mode == "preflight" else run(args)
