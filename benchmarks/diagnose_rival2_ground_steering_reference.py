"""Read-only simulator/observation sanity reference, NEVER Rival or training data.

A simple ground pursuit controller uses only the same final 182 observations.
It tests whether existing inputs/controls can solve basic acquisition at all.
It is not a trained bot, learned competence evidence, a reward, a detector, an
expert prefix, a target dataset, or a controller to deploy against a human.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_ssl_exploration_comparison import (
    EXTERNAL,
    PARENT,
    RESULTS,
    sha,
    utc,
    write_json,
)
from rivalsim.fresh_ground_30hz import SEED, FreshGroundEnv, policy_config, scenario_hash, scenarios
from rivalsim.fresh_ground_exploration_comparison import ScaledNoiseActorCritic
from rivalsim.rival2_contracts import (
    ANGULAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.rival2_unified_policy import deterministic_unified_action


def pursuit_action(observation):
    def vector(name):
        index = OBS_FIELD_NAMES.index(name + ".x")
        return observation[..., index : index + 3]

    delta = vector("relative.ball_position") * observation.new_tensor(POSITION_SCALE)
    forward = vector("self.forward")
    heading_error = torch.atan2(
        forward[..., 0] * delta[..., 1] - forward[..., 1] * delta[..., 0],
        (forward[..., :2] * delta[..., :2]).sum(-1),
    )
    yaw_rate = vector("self.angular_velocity")[..., 2] * ANGULAR_SPEED_SCALE
    speed = vector("self.linear_velocity").norm(dim=-1) * CAR_LINEAR_SPEED_SCALE
    distance = delta.norm(dim=-1)
    output = observation.new_zeros((*observation.shape[:-1], 8))
    output[..., 0] = torch.where(heading_error.abs() < 0.8, 1.0, 0.35)
    output[..., 1] = (2.5 * heading_error - 0.12 * yaw_rate).clamp(-1, 1)
    output[..., 6] = ((heading_error.abs() < 0.12) & (distance > 800) & (speed < 1400)).float()
    output[..., 7] = ((heading_error.abs() > 1.4) & (speed > 600)).float()
    return output


@torch.inference_mode()
def main():
    for arm in ("control", "half_sigma"):
        assert (
            json.loads((EXTERNAL / arm / "campaign_state.json").read_text())["status"]
            == "completed"
        )
    digest = sha(PARENT)
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = ScaledNoiseActorCritic(policy_config())
    model.load_state_dict(payload["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    n = 256
    bank = scenarios(n, SEED + 100, family_only=2)
    env = FreshGroundEnv(
        n,
        "G:/dev/RLBot-Rival/bot/collision_meshes",
        device="cuda:0",
        seed=SEED + 100,
        ssl_foundation_scenarios=bank,
    )
    rows = torch.arange(n, device=env.device)
    side = torch.tensor(bank.focal_side.astype("int64"), device=env.device)
    hidden = model.initial_hidden(n * 2)
    reset = torch.ones(n * 2, dtype=torch.bool, device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    first = torch.full((n,), float("nan"), device=env.device)
    for tick in range(450):
        actor, hidden = model.forward_actor(
            env.observation.reshape(-1, 182), hidden, reset_before=reset
        )
        action = deterministic_unified_action(actor).reshape(n, 2, 8)
        action[rows, side] = pursuit_action(env.observation[rows, side])
        tr = env.step(action)
        contact = env.last_native["touch_count"][rows, side] * alive
        first = torch.where(
            (contact > 0) & first.isnan(),
            (tick * 4 + env.last_native["first_touch_tick"][rows, side] + 1) / 120.0,
            first,
        )
        alive &= ~tr.reset_mask
        reset = tr.reset_mask[:, None].expand(-1, 2).reshape(-1)
        hidden = hidden.masked_fill(reset[None, :, None], 0)
    assert sha(PARENT) == digest
    touched = first.isfinite()
    report = dict(
        utc=utc(),
        verdict="DIAGNOSTIC_REFERENCE_NOT_LEARNED_RIVAL",
        optimizer_steps=0,
        policy_checkpoint_unchanged=True,
        worlds=n,
        seconds=15,
        scenario_sha256=scenario_hash(bank),
        inputs="Only existing final 182D observation; no future/native hidden state",
        opponent="Frozen parent policy; focal controller is explicitly scripted diagnostic only",
        cases_with_touch=int(touched.sum()),
        first_touch_median=float(first[touched].median()) if touched.any() else None,
        included_in_training=False,
        deployed=False,
        per_case_first_touch=[float(v) if torch.isfinite(v) else None for v in first],
    )
    write_json(RESULTS / "diagnostics" / "ground_steering_reference.json", report)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("per_case")}))


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
