"""Frozen650 sampling-versus-greedy diagnostic. No learning or deployment edit."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks import run_rival2_direct_skills_v1 as base
from benchmarks import run_direct_skills_exploration_followup_v1 as follow
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash

OUT = base.RESULTS / "sampling_diagnostic_000650"
SOURCE = base.CHECKPOINTS / "plus_000650.pt"
SOURCE_SHA = "4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F"
SEEDS = (20260906651, 20260906652, 20260906653)
SOURCES = (__file__, str(ROOT / "tests/test_direct_skills_sampling_diagnostic.py"))


class SampledEvaluationPolicy(base.EntityJointControlActorCritic):
    """Only this diagnostic's historical `deterministic` interface slot samples.

    Network forward/state_dict remain original. A private explicit RNG samples
    exactly the temperature2 distribution used by the frozen PPO collector.
    This class must not be installed in the production evaluator or deployment.
    """

    def __init__(self, seed, device="cpu"):
        super().__init__()
        self.action_generator = torch.Generator(device=device).manual_seed(seed)

    def deterministic(self, logits):
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite diagnostic actor logits")
        return self.sample(logits / 2.0, self.action_generator)[1]


def authority():
    return dict(
        version="RIVAL2_DIRECT_SKILLS_650_SAMPLING_DIAGNOSTIC_V1",
        checkpoint=SOURCE.relative_to(ROOT).as_posix(), checkpoint_sha256=SOURCE_SHA,
        parent_training_authority_sha256=content_hash(follow.amendment()),
        action_mode="categorical raw logits/2 sampled at30Hz, four physical ticks held",
        sampling_seeds=list(SEEDS), model_selection=False, optimizer_steps=0,
        baseline="Existing completed deterministic650 skill/fullmatch artifacts; never overwrite",
        full_matches="Same ten layouts/sides and fixed physics seed per sampling seed; "
        "300second regulation plus existing bounded overtime; original corrected reset",
        finishing="Same64 initial finishing cases and12second limit per sampling seed; "
        "private RNG restarted from declared seed separately for each evaluation",
        outcomes="Report all three seeds, per-case and aggregate goals/concedes/contacts. "
        "No seed/temperature search, no checkpoint selection, no promotion rule",
        inference="A changed action-selection diagnostic, NOT a same-method learning comparison. "
        "It cannot excuse failed deterministic performance or establish SSL",
        unchanged="Model/Adam/checkpoint, reward/physics/Nexto, observations/actions, "
        "30Hz/120Hz cadence, recurrent lifecycle; no actor or optimizer mutation",
        stop="Complete finite diagnostic once. Preserve failures; no automatic learning/resume",
    )


def prepare():
    assert not (OUT / "authority.json").exists()
    follow.verify()
    assert sha(SOURCE) == SOURCE_SHA
    write_json(OUT / "authority.json", authority())
    write_json(OUT / "package.json", dict(
        authority_sha256=content_hash(authority()),
        sources={Path(p).relative_to(ROOT).as_posix():base.text_sha(Path(p)) for p in SOURCES},
        tests_sha256=base.text_sha(OUT / "tests.xml"),
        baselines={name:base.text_sha(base.RESULTS / name)
                   for name in ("evaluation_000650.json", "full_match_000650.json")},
    ))


def verify():
    follow.verify()
    package = json.loads((OUT / "package.json").read_text())
    assert json.loads((OUT / "authority.json").read_text()) == authority()
    assert package["authority_sha256"] == content_hash(authority())
    assert sha(SOURCE) == SOURCE_SHA
    for name, expected in package["sources"].items():
        assert base.text_sha(ROOT / name) == expected, name
    assert base.text_sha(OUT / "tests.xml") == package["tests_sha256"]
    for name, expected in package["baselines"].items():
        assert base.text_sha(base.RESULTS / name) == expected, name
    paths = [*package["sources"], *[(OUT / name).relative_to(ROOT).as_posix()
                                  for name in ("authority.json", "package.json", "tests.xml")]]
    for name in paths:
        assert subprocess.check_output(["git", "show", "origin/main:"+name], cwd=ROOT).replace(
            b"\r\n", b"\n") == (ROOT/name).read_bytes().replace(b"\r\n", b"\n"), name
    return package


def policy(seed):
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = SampledEvaluationPolicy(seed, "cuda:0").cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    assert tensor_hash(model.state_dict()) == tensor_hash(payload["model"])
    return model


@torch.inference_mode()
def finishing(model):
    n, family = 64, 2
    seed = base.SEED + 1000 + family
    bank = base.scenarios(n, seed, family_only=family)
    env = base.DirectSkillsEnv(n, base.COLLISION, device="cuda:0", seed=seed,
                              ssl_foundation_scenarios=bank)
    rows = torch.arange(n, device=env.device)
    side = torch.as_tensor(bank.focal_side.astype("int64"), device=env.device)
    nexto = base.NextoPolicyAdapter(n, device=env.device)
    nexto.set_player_index(1-side)
    alive = torch.ones(n, device=env.device, dtype=torch.bool)
    nexto.activate(alive)
    ns = base.NextoStateTensors.from_bridge(env.bridge)
    hidden = model.initial_hidden(n*2)
    reset = torch.ones(n*2, device=env.device, dtype=torch.bool)
    touches = torch.zeros(n, device=env.device)
    first = torch.full((n,), float("nan"), device=env.device)
    goals, concedes, duration = touches.clone(), touches.clone(), touches.clone()
    events = torch.zeros((n,7), device=env.device)
    ending = torch.zeros(n, device=env.device, dtype=torch.int64)
    for tick in range(360):
        logits, hidden = model.forward_actor(env.observation.reshape(-1,182), hidden,
                                             reset_before=reset)
        assert bool(torch.isfinite(hidden).all())
        action = model.deterministic(logits).reshape(n,2,8)
        def provider(_, action=action):
            applied = action.clone()
            kickoff = (ns.ball_pos[:,0]==0) & (ns.ball_pos[:,1]==0)
            controls, _ = nexto.tick_action(ns, kickoff, active_mask=alive)
            applied[rows,1-side] = controls
            return applied
        tr = env.step_with_tick_actions(action, provider)
        contact = env.last_native["touch_count"][rows,side]*alive
        touches += contact
        first = torch.where((contact>0)&first.isnan(),
                            (tick*4+env.last_native["first_touch_tick"][rows,side]+1)/120,first)
        goals += alive & tr.terminated & (env.last_native["scoring_team"]==side)
        concedes += alive & tr.terminated & (env.last_native["scoring_team"]!=side)
        events += env.last_skill["events"][rows,side]*alive[:,None]
        duration += alive/30
        ending += alive*(tr.terminated.long()+2*tr.truncated.long())
        alive &= ~tr.reset_mask
        reset = tr.reset_mask[:,None].expand(-1,2).reshape(-1)
        hidden = hidden.masked_fill(reset[None,:,None],0)
        if not bool(alive.any()): break
    result = dict(worlds=n, scenario_sha256=scenario_hash(bank), goals_for=int(goals.sum()),
                  goals_against=int(concedes.sum()), touches=int(touches.sum()),
                  touched_worlds=int(first.isfinite().sum()),
                  timeouts=int((ending==2).sum()), unfinished=int(alive.sum()),
                  events=dict(zip(base.reward_authority()["events"],events.sum(0).cpu().tolist(),strict=True)),
                  raw=dict(touches=touches.cpu().tolist(),goals=goals.cpu().tolist(),
                           concedes=concedes.cpu().tolist(),seconds=duration.cpu().tolist(),
                           endings=ending.cpu().tolist()))
    assert result["scenario_sha256"] == json.loads((base.RESULTS/"evaluation_000650.json").read_text())["skills"]["finishing"]["scenario_sha256"]
    assert result["goals_for"]+result["goals_against"]+result["timeouts"] == 64
    return result


def run():
    package=verify()
    payload=torch.load(SOURCE,map_location="cpu",weights_only=False)
    expected_model=tensor_hash(payload["model"])
    del payload
    for seed in SEEDS:
        destination=OUT/f"seed_{seed}.json"
        assert not destination.exists(), "Never overwrite or silently rerun a completed seed"
        model=policy(seed)
        skill=finishing(model)
        assert tensor_hash(model.state_dict())==expected_model
        del model
        gc.collect()
        with base.owned_match_stream():
            runner=base.CandidateMatchRunner(SOURCE,SOURCE_SHA,entity=True)
            runner.rival_policy=policy(seed)
            runner.checkpoint_identity["action"]="joint90_categorical_temperature2_diagnostic"
            elapsed=runner.run_ticks(base.REGULATION_TICKS).seconds
            for _ in range(base.OVERTIME_CAP_TICKS//600):
                if bool(runner.phase_status()["done"].all()): break
                elapsed+=runner.run_ticks(600).seconds
            raw=runner.export()["raw"]
            assert not bool(raw["goal_overflow"].any())
            assert tensor_hash(runner.rival_policy.state_dict())==expected_model==runner.model_hash_before
            assert sha(SOURCE)==SOURCE_SHA
            result=dict(utc=utc(),accepted_updates=650,sampling_seed=seed,
                        action_mode=authority()["action_mode"],optimizer_steps=0,
                        authority_sha256=content_hash(authority()),
                        runtime_package_sha256=content_hash(package),
                        match_reset_version=base.MATCH_RESET_VERSION,
                        checkpoint=dict(path=str(SOURCE),sha256=SOURCE_SHA,accepted_updates=650),
                        summary=base.summarize(raw),raw={k:v.tolist() for k,v in raw.items()},
                        wall_seconds=elapsed,hidden_resets=runner.hidden_reset_count.cpu().tolist(),
                        model_unchanged=True,checkpoint_unchanged=True,finishing=skill,
                        meaning="Changed sampling diagnostic; NOT same-method learning progress or deployment")
            write_json(destination,result)
            print(json.dumps(dict(seed=seed,matches=result["summary"],finishing={k:skill[k] for k in
                             ("goals_for","goals_against","touches","timeouts")})),flush=True)
            del runner
        gc.collect()
        torch.cuda.empty_cache()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","verify","run"))
    args=parser.parse_args()
    torch.set_num_threads(8)
    if args.command=="prepare": prepare()
    elif args.command=="verify": verify()
    else:
        with base.gpu_lease():
            try: run()
            except Exception as exc:
                write_json(OUT/"failure.json",dict(utc=utc(),exception=repr(exc),
                           traceback=traceback.format_exc(),optimizer_steps=0))
                raise
