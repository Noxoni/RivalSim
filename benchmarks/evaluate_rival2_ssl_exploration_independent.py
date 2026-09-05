"""Predeclared independent scenario confirmation after both bounded arms finish.

No learning or checkpoint selection. All three policies are measured on the
same new states; these results must not be used to reopen this pilot's training.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_ssl_exploration_comparison import (
    ARMS,
    CHECKPOINTS,
    EXTERNAL,
    PARENT,
    PARENT_SHA,
    RESULTS,
    sha,
    utc,
    write_json,
)
from rivalsim.fresh_ground_30hz import FreshGroundEnv, policy_config, scenario_hash, scenarios
from rivalsim.fresh_ground_exploration_comparison import ScaledNoiseActorCritic
from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, OBS_FIELD_NAMES
from rivalsim.rival2_unified_policy import deterministic_unified_action
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

CASES = [
    ("acquisition_selfplay", 2, False, 2026091501),
    ("finishing_selfplay", 4, False, 2026091502),
    ("ongoing_nexto", 0, True, 2026091503),
    ("kickoff_nexto", 1, True, 2026091504),
]
WORLDS = 256
DECISIONS = 900


def authority():
    return dict(
        version="RIVAL2_SSL_EXPLORATION_INDEPENDENT_EVAL_V1",
        worlds=WORLDS,
        max_decisions=DECISIONS,
        policy_hz=30,
        physics_hz=120,
        cases=CASES,
        parent_sha256=PARENT_SHA,
        checkpoints=["parent_u000597", "control_plus030", "half_sigma_plus030"],
        mode="deterministic; original episode only; scenario outcomes not full match win rates",
        selection="No reopening completed pilot training or checkpoint selection. Report all three policies on every case, including regressions.",
        no_optimizer_steps=True,
        seeds_not_used_for_training=True,
    )


def prepare():
    path = RESULTS / "independent_evaluation_authority.json"
    if path.exists():
        raise RuntimeError("independent authority already exists")
    write_json(
        path,
        dict(
            authority=authority(),
            utc=utc(),
            evaluator_sha256=sha(Path(__file__)),
            scenario_hashes={
                name: scenario_hash(scenarios(WORLDS, seed, family_only=family))
                for name, family, _, seed in CASES
            },
        ),
    )


@torch.no_grad()
def evaluate_case(model, name, family, use_nexto, seed):
    bank = scenarios(WORLDS, seed, family_only=family)
    env = FreshGroundEnv(
        WORLDS,
        "G:/dev/RLBot-Rival/bot/collision_meshes",
        device="cuda:0",
        seed=seed,
        ssl_foundation_scenarios=bank,
    )
    side = torch.tensor(bank.focal_side.astype("int64"), device=env.device)
    rows = torch.arange(WORLDS, device=env.device)
    alive = torch.ones(WORLDS, device=env.device, dtype=torch.bool)
    first = torch.full((WORLDS,), float("nan"), device=env.device)
    hidden = model.initial_hidden(WORLDS * 2, device=env.device)
    reset = torch.ones(WORLDS * 2, device=env.device, dtype=torch.bool)
    totals = {
        key: torch.zeros(WORLDS, device=env.device)
        for key in (
            "touches",
            "goals_for",
            "goals_against",
            "no_touch",
            "exposure",
            "speed",
            "jump",
            "boost",
            "handbrake",
            "goalward_contacts",
        )
    }
    opponent = NextoPolicyAdapter(WORLDS, device=env.device) if use_nexto else None
    if opponent:
        opponent.set_player_index(1 - side)
        opponent.activate(alive)
        ns = NextoStateTensors.from_bridge(env.bridge)
    speed_index = OBS_FIELD_NAMES.index("self.linear_velocity.x")
    ball_vy_index = OBS_FIELD_NAMES.index("ball.linear_velocity.y")
    for tick in range(DECISIONS):
        observation = env.observation
        actor, hidden = model.forward_actor(
            observation.reshape(-1, 182), hidden, reset_before=reset
        )
        action = deterministic_unified_action(actor).reshape(WORLDS, 2, 8)
        if opponent:

            def provider(_tick, _action=action, _alive=alive, _opponent=opponent, _ns=ns):
                output = _action.clone()
                kickoff = (_ns.ball_pos[:, 0] == 0) & (_ns.ball_pos[:, 1] == 0)
                controls, _ = _opponent.tick_action(_ns, kickoff, active_mask=_alive)
                output[rows, 1 - side] = controls
                return output

            tr = env.step_with_tick_actions(action, provider)
        else:
            tr = env.step(action)
        native = env.last_native
        contacts = native["touch_count"][rows, side] * alive
        first = torch.where(
            (contacts > 0) & first.isnan(),
            (tick * 4 + native["first_touch_tick"][rows, side] + 1) / 120.0,
            first,
        )
        totals["touches"] += contacts
        totals["goalward_contacts"] += contacts * (
            tr.transition_observation[rows, side, ball_vy_index] > 0
        )
        totals["goals_for"] += alive & tr.terminated & (native["scoring_team"] == side)
        totals["goals_against"] += alive & tr.terminated & (native["scoring_team"] != side)
        totals["no_touch"] += alive & tr.truncated & (native["no_touch_ticks"] >= 1800)
        totals["exposure"] += alive / 30
        totals["speed"] += (
            alive
            * observation[rows, side, speed_index : speed_index + 3].norm(dim=-1)
            * CAR_LINEAR_SPEED_SCALE
            / 30
        )
        for i, key in enumerate(("jump", "boost", "handbrake"), 5):
            totals[key] += alive * action[rows, side, i] / 30
        alive &= ~tr.reset_mask
        reset = tr.reset_mask[:, None].expand(-1, 2).reshape(-1)
        hidden = hidden.masked_fill(reset[None, :, None], 0)
        if not alive.any():
            break
    sums = {k: float(v.sum()) for k, v in totals.items()}
    touched = first.isfinite()
    result = dict(
        name=name,
        seed=seed,
        scenario_sha256=scenario_hash(bank),
        worlds=WORLDS,
        focal_cases_with_touch=int(touched.sum()),
        focal_touch_fraction=float(touched.float().mean()),
        median_first_touch_seconds_if_touched=float(first[touched].median())
        if touched.any()
        else None,
        totals=sums,
        touches_per_player_minute=sums["touches"] / sums["exposure"] * 60,
        mean_speed=sums["speed"] / sums["exposure"],
        survivors=int(alive.sum()),
        per_case=[
            dict(
                case=i,
                side=int(side[i]),
                first_touch_seconds=float(first[i]) if touched[i] else None,
                **{k: float(v[i]) for k, v in totals.items()},
            )
            for i in range(WORLDS)
        ],
    )
    del env, opponent
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run():
    frozen = json.loads((RESULTS / "independent_evaluation_authority.json").read_text())
    # JSON normalizes case tuples to lists.
    assert frozen["authority"] == json.loads(json.dumps(authority()))
    assert frozen["evaluator_sha256"] == sha(Path(__file__))
    for arm in ARMS:
        state = json.loads((EXTERNAL / arm / "campaign_state.json").read_text())
        assert state["status"] == "completed" and state["additional_updates"] == 30
    assert sha(PARENT) == PARENT_SHA
    paths = {"parent": PARENT, **{arm: CHECKPOINTS / arm / "plus_030.pt" for arm in ARMS}}
    for name, path in paths.items():
        output = RESULTS / "independent" / f"{name}.json"
        if output.exists():
            raise RuntimeError("independent evaluation already exists; do not silently rerun")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = ScaledNoiseActorCritic(policy_config())
        model.load_state_dict(payload["model"], strict=True)
        model.set_sigma_scale(ARMS.get(name, 1.0))
        model.cuda().eval().requires_grad_(False)
        result = dict(utc=utc(), checkpoint=dict(path=str(path), sha256=sha(path)), cases={})
        for case in CASES:
            result["cases"][case[0]] = evaluate_case(model, *case)
            print(name, case[0], result["cases"][case[0]]["focal_touch_fraction"], flush=True)
        write_json(output, result)
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(8)
    if sys.argv[-1] == "prepare":
        prepare()
    elif sys.argv[-1] == "run":
        run()
    else:
        raise SystemExit("Choose prepare or run")
