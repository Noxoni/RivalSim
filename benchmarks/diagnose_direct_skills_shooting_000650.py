"""Bounded native shooting-pressure comparison; no learning or reward edits."""

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

import numpy as np
import torch

from benchmarks import run_rival2_direct_skills_v1 as base
from benchmarks import run_direct_skills_exploration_followup_v1 as follow
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash
from rivalsim.ssl_foundation_v1 import _set_coherent_ground_route

OUT = base.RESULTS / "shooting_diagnostic_000650"
SOURCE = base.CHECKPOINTS / "plus_000650.pt"
SOURCE_SHA = "4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F"
MODES = ("established_keeper", "recovering_defender", "initial_open_net")
SEED = base.SEED + 1002
SOURCES = ("benchmarks/diagnose_direct_skills_shooting_000650.py",
           "tests/test_direct_skills_shooting_diagnostic.py")


def bank_for(mode):
    if mode not in MODES:
        raise ValueError(mode)
    bank = base.scenarios(64, SEED, family_only=2)
    if mode == "established_keeper":
        return bank
    rng = np.random.default_rng(SEED + 650)
    s = bank.state
    for row, focal in enumerate(bank.focal_side):
        opponent = 1 - focal
        sign = 1 if focal == 0 else -1
        # Alternate flank independently of team. Only opponent pos/vel/quaternion change.
        x = (1 if row % 2 else -1) * (3000 if mode == "initial_open_net" else 2200)
        y = -3500 if mode == "initial_open_net" else s.ball_pos[row, 1] * sign - 1300
        s.car_pos[row, opponent, :2] = np.array([x, y]) * sign
        target = s.ball_pos[row, :2] if mode == "initial_open_net" else np.array([0, 5120]) * sign
        speed = (0, 0) if mode == "initial_open_net" else (600, 600)
        _set_coherent_ground_route(s, row, opponent, rng, target, speed, (0, 0))
    s.validate()
    return bank


def authority():
    return dict(
        version="RIVAL2_DIRECT_SKILLS_650_SHOOTING_PRESSURE_DIAGNOSTIC_V1",
        checkpoint_sha256=SOURCE_SHA, modes=list(MODES), seed=SEED, worlds_per_mode=64,
        horizon_seconds=12, optimizer_steps=0, action="Unchanged raw-logit argmax at30Hz/120Hz physics",
        scope="Matched ball and Rival starts; change only initial opponent position, coherent "
        "velocity and yaw. Same active pinnedNexto, boost, observation, reward, physics and reset semantics",
        established_keeper="Exactly existing64 finishing starts; must reproduce650 per-case outcomes",
        recovering_defender="Opponent at canonical x alternating+/-2200,y=ball_y-1300, "
        "grounded,600uu/s toward own goal center with original coherent-route heading jitter",
        initial_open_net="Opponent at canonical x alternating+/-3000,y=-3500,zero velocity, "
        "facing ball with coherent-route heading jitter. Nexto stays active and may recover; "
        "this is initially open, not an artificially disabled goalkeeper for12seconds",
        evidence="All192 outcomes and30Hz pre/action/post observations, alive masks, native "
        "contact counts/subtick times and goal times. Count actual goals before any opponent "
        "contact and all goals separately. No inferred mechanic detector or reward",
        decision="Determine accessible finishing versus keeper-pressure failure before a small "
        "prospective curriculum adjustment. No automatic learning or choosing a lucky checkpoint",
        safety="ExclusiveGPUlease, finite outputs/states, checkpoint/Nexto/model unchanged; "
        "preserve failure and never overwrite completed mode. Existing650 review STOP remains",
    )


def prepare():
    assert not (OUT / "authority.json").exists()
    follow.verify()
    assert sha(SOURCE) == SOURCE_SHA
    write_json(OUT / "authority.json", authority())
    write_json(OUT / "package.json", dict(
        authority_sha256=content_hash(authority()),
        sources={p:base.text_sha(ROOT / p) for p in SOURCES},
        tests_sha256=base.text_sha(OUT / "tests.xml"),
        baseline_sha256=base.text_sha(base.RESULTS / "evaluation_000650.json"),
        scenario_sha256={mode:scenario_hash(bank_for(mode)) for mode in MODES},
    ))


def verify():
    follow.verify()
    package = json.loads((OUT / "package.json").read_text())
    assert json.loads((OUT / "authority.json").read_text()) == authority()
    assert package["authority_sha256"] == content_hash(authority())
    assert sha(SOURCE) == SOURCE_SHA
    assert base.text_sha(OUT / "tests.xml") == package["tests_sha256"]
    assert base.text_sha(base.RESULTS / "evaluation_000650.json") == package["baseline_sha256"]
    for mode in MODES:
        assert scenario_hash(bank_for(mode)) == package["scenario_sha256"][mode]
    for p, expected in package["sources"].items():
        assert base.text_sha(ROOT / p) == expected, p
    for p in (*SOURCES, *((OUT / n).relative_to(ROOT).as_posix()
                           for n in ("authority.json", "package.json", "tests.xml"))):
        assert subprocess.check_output(["git", "show", "origin/main:"+p]).replace(
            b"\r\n", b"\n") == (ROOT/p).read_bytes().replace(b"\r\n", b"\n"), p
    return package


@torch.inference_mode()
def evaluate(mode, model, package):
    bank = bank_for(mode)
    env = base.DirectSkillsEnv(64, base.COLLISION, device="cuda:0", seed=SEED,
                              ssl_foundation_scenarios=bank)
    rows = torch.arange(64, device=env.device)
    side = torch.as_tensor(bank.focal_side.astype("int64"), device=env.device)
    nexto = base.NextoPolicyAdapter(64, device=env.device)
    nexto.set_player_index(1-side)
    alive = torch.ones(64, device=env.device, dtype=torch.bool)
    nexto.activate(alive)
    nexto_hash = tensor_hash(nexto.actor.state_dict())
    ns = base.NextoStateTensors.from_bridge(env.bridge)
    hidden = model.initial_hidden(128)
    reset = torch.ones(128, device=env.device, dtype=torch.bool)
    records = {k:[] for k in ("before", "action", "after", "alive", "touches", "touch_tick",
                              "goal_tick", "scoring_team", "terminated", "truncated", "events")}
    # Match the published evaluator's accumulation dtype/order exactly.
    touches = torch.zeros(64, device=env.device)
    goals, concedes, duration = touches.clone(), touches.clone(), touches.clone()
    ending = torch.zeros(64, dtype=torch.int64, device=env.device)
    events = torch.zeros((64,7), device=env.device)
    for tick in range(360):
        before = env.observation[rows, side].clone()
        logits, hidden = model.forward_actor(env.observation.reshape(-1,182), hidden, reset_before=reset)
        assert bool(torch.isfinite(logits).all() & torch.isfinite(hidden).all())
        action = model.deterministic(logits).reshape(64,2,8)
        def provider(_):
            applied = action.clone()
            controls, _ = nexto.tick_action(ns, (ns.ball_pos[:,0]==0)&(ns.ball_pos[:,1]==0),
                                            active_mask=alive)
            applied[rows,1-side] = controls
            return applied
        tr = env.step_with_tick_actions(action, provider)
        after = tr.transition_observation[rows,side]
        assert bool(torch.isfinite(after).all())
        # Saved contact columns are focal Rival then opponent, not fixed blue/orange.
        contact = env.last_native["touch_count"].gather(1, torch.stack((side,1-side),1))
        touch_tick = env.last_native["first_touch_tick"].gather(1, torch.stack((side,1-side),1))
        values = dict(before=before, action=action[rows,side], after=after, alive=alive,
                      touches=contact, touch_tick=touch_tick,
                      goal_tick=env.last_native["first_goal_tick"],
                      scoring_team=env.last_native["scoring_team"], terminated=tr.terminated,
                      truncated=tr.truncated, events=env.last_skill["events"][rows,side])
        for key, value in values.items():
            records[key].append(value.clone())
        touches += contact[:,0] * alive
        goals += alive & tr.terminated & (env.last_native["scoring_team"]==side)
        concedes += alive & tr.terminated & (env.last_native["scoring_team"]!=side)
        events += env.last_skill["events"][rows,side] * alive[:,None]
        duration += alive/30
        ending += alive * (tr.terminated.long()+2*tr.truncated.long())
        alive &= ~tr.reset_mask
        reset = tr.reset_mask[:,None].expand(-1,2).reshape(-1)
        hidden = hidden.masked_fill(reset[None,:,None],0)
        if not bool(alive.any()):
            break
    assert not bool(alive.any()) and bool(((ending==1)|(ending==2)).all())
    assert tensor_hash(nexto.actor.state_dict()) == nexto_hash
    arrays = {k:torch.stack(v).cpu().numpy() for k,v in records.items()}
    arrays["focal_side"] = bank.focal_side
    first_own, first_opp, score_time = [], [], []
    for row in range(64):
        for car, times in ((0,first_own),(1,first_opp)):
            indices = np.flatnonzero(arrays['alive'][:,row] & (arrays['touches'][:,row,car]>0))
            t = int(indices[0]) if len(indices) else None
            times.append((t*4+int(arrays['touch_tick'][t,row,car])+1)/120 if t is not None else None)
        indices = np.flatnonzero(arrays['alive'][:,row] & arrays['terminated'][:,row])
        t = int(indices[0]) if len(indices) else None
        score_time.append((t*4+int(arrays['goal_tick'][t,row])+1)/120 if t is not None else None)
    raw = dict(touches=touches.cpu().tolist(), goals=goals.cpu().tolist(),
               concedes=concedes.cpu().tolist(), seconds=duration.cpu().tolist(),
               endings=ending.cpu().tolist())
    if mode == "established_keeper":
        baseline = json.loads((base.RESULTS / "evaluation_000650.json").read_text())["skills"]["finishing"]
        assert raw == baseline["raw"], "Diagnostic must reproduce original per-case greedy evaluation"
    trace_path = OUT / f"{mode}.npz"
    assert not trace_path.exists()
    np.savez_compressed(trace_path, **arrays)
    result = dict(mode=mode, utc=utc(), checkpoint_sha256=SOURCE_SHA,
                  authority_sha256=content_hash(authority()), scenario_sha256=scenario_hash(bank),
                  trace=dict(path=trace_path.relative_to(ROOT).as_posix(), sha256=sha(trace_path)),
                  goals_for=int(goals.sum()), goals_against=int(concedes.sum()),
                  timeouts=int((ending==2).sum()), touched_worlds=sum(t is not None for t in first_own),
                  touches=int(touches.sum()), first_rival_contact_seconds=first_own,
                  first_opponent_contact_seconds=first_opp, goal_seconds=score_time,
                  goals_before_any_opponent_contact=sum(bool(raw['goals'][i]) and
                    (first_opp[i] is None or score_time[i]<first_opp[i]) for i in range(64)),
                  events=dict(zip(base.reward_authority()["events"],events.sum(0).cpu().tolist(),strict=True)),
                  raw=raw, original_baseline_reproduced=(mode=="established_keeper"),
                  nexto_unchanged=True, model_unchanged=True, optimizer_steps=0)
    return result


def run():
    package = verify()
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = base.EntityJointControlActorCritic().cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    expected = tensor_hash(payload["model"])
    for mode in MODES:
        path = OUT / f"{mode}.json"
        assert not path.exists(), "Never overwrite a completed mode"
        result = evaluate(mode, model, package)
        assert tensor_hash(model.state_dict()) == expected and sha(SOURCE) == SOURCE_SHA
        write_json(path, result)
        print(json.dumps({k:result[k] for k in ('mode','goals_for','goals_against','timeouts',
                                               'goals_before_any_opponent_contact')}), flush=True)
        gc.collect()
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    else:
        with base.gpu_lease():
            try:
                run()
            except Exception as exc:
                write_json(OUT / "failure.json", dict(utc=utc(), exception=repr(exc),
                           traceback=traceback.format_exc(), optimizer_steps=0))
                raise
