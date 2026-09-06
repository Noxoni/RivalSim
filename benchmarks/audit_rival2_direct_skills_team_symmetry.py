"""Bounded CPU-only team-side diagnostic; no rollout or optimizer operation.

Reuses archived native reset views and the production observation function.
Half-turn counterfactuals test canonicalization, not live physics equivalence.
Checkpoint assignment counts are instantaneous, not cumulative side exposure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.rival2_contracts import (  # noqa: E402
    OBS_FIELD_NAMES,
    ORANGE_PAD_REMAP,
    POSITION_SCALE,
)
from rivalsim.rival2_env import Rival2TensorBridge  # noqa: E402
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic  # noqa: E402

RESULTS = ROOT / "results/rival2/direct_skills_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/direct_skills_v1"
CAR_FIELDS = (
    "car_pos",
    "car_vel",
    "car_quat",
    "car_ang_vel",
    "boost",
    "boosting_time",
    "time_since_boosted",
    "on_ground",
    "has_jumped",
    "is_jumping",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "sticky_ticks",
    "jump_time",
    "air_time",
    "air_time_since_jump",
    "flip_time",
    "is_supersonic",
    "supersonic_time",
    "wheel_contact",
    "car_is_demoed",
    "demo_respawn_timer",
    "rival2.previous_action",
    "rival2.touch_count",
    "rival2.demoed_event",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def bridge(views, worlds):
    # No Warp world, device allocation, simulator step or source modification.
    result = object.__new__(Rival2TensorBridge)
    result.num_envs = worlds
    result.device = torch.device("cpu")
    result.views = views
    result.position_scale = torch.tensor(POSITION_SCALE, dtype=torch.float32)
    result.pad_durations = torch.tensor([10.0] * 6 + [4.0] * 28)
    result.blue_pad_remap = torch.arange(34)
    result.orange_pad_remap = torch.tensor(ORANGE_PAD_REMAP)
    result.team_signs = torch.tensor(((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)))
    return result


def half_turn(views, worlds):
    result = {name: value.clone() for name, value in views.items()}
    for name in CAR_FIELDS:
        value = views[name]
        result[name] = value.reshape(worlds, 2, -1).flip(1).reshape_as(value).clone()
    for name in ("car_pos", "car_vel", "car_ang_vel", "ball_pos", "ball_vel", "ball_ang_vel"):
        result[name][..., :2] *= -1
    x, y, z, w = result["car_quat"].unbind(-1)
    result["car_quat"] = torch.stack((-y, x, w, -z), -1)
    result["pad_cooldown"] = views["pad_cooldown"].reshape(worlds, 34)[:, ORANGE_PAD_REMAP].clone()
    return result


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    fixture = RESULTS / "training_reset_contact_fixture.npz"
    assert sha(fixture) == "08C01C48D6FE9AC5A782F548314A48C93DA32EC25258E8D7EA8E735756B1E7C2"
    with np.load(fixture, allow_pickle=False) as saved:
        views = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
    original_views = {name: value.clone() for name, value in views.items()}
    recorded = views["final_observation"]
    worlds = len(recorded)
    native = bridge(views, worlds).observation()
    replay_error = float((native - recorded).abs().max())
    assert replay_error <= 1e-6
    mirrored = bridge(half_turn(views, worlds), worlds).observation().flip(1)
    error = (native - mirrored).abs()
    assert float(error.max()) <= 1e-6
    assert all(torch.equal(value, original_views[name]) for name, value in views.items())
    policy = EntityJointControlActorCritic().eval()
    checkpoint = CHECKPOINTS / "plus_000500.pt"
    expected_sha = "2DF2D29E766E91E01EC08BAB7D50FB526F8EAA41490818470C9686B3760BCF5A"
    assert sha(checkpoint) == expected_sha
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(saved_model["model"], strict=True)
    model_before = {key: value.clone() for key, value in policy.state_dict().items()}
    with torch.no_grad():
        first, value, hidden = policy(native.reshape(-1, 182), policy.initial_hidden(worlds * 2))
        second, other_value, other_hidden = policy(
            mirrored.reshape(-1, 182), policy.initial_hidden(worlds * 2)
        )
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    changed = int((first.argmax(-1) != second.argmax(-1)).sum())
    assert changed == 0
    assert all(torch.equal(value, model_before[key]) for key, value in policy.state_dict().items())
    assignments = []
    for offset in (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500):
        path = CHECKPOINTS / f"plus_{offset:06d}.pt"
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        state = loaded["opponent_state"]
        nexto, side = state["is_nexto"], state["learner_side"]
        assert torch.equal(nexto, torch.arange(len(nexto)).remainder(2) == 0)
        assert torch.equal(state["nexto_cache"]["player_index"], 1 - side)
        counts = torch.bincount(side[nexto], minlength=2).tolist()
        assignments.append(
            dict(
                offset=offset,
                sha256=sha(path),
                nexto_learner_blue_orange=counts,
                nexto_blue_fraction=counts[0] / sum(counts),
                selfplay_focal_blue_orange=torch.bincount(side[~nexto], minlength=2).tolist(),
                nexto_player_index_complement_exact=True,
                fixed_even_nexto_slots_exact=True,
            )
        )
    assert sha(checkpoint) == expected_sha and not torch.cuda.is_initialized()
    report = dict(
        schema="RIVAL2_DIRECT_SKILLS_TEAM_SYMMETRY_DIAGNOSTIC_V1",
        fixture_sha256=sha(fixture),
        checkpoint_sha256=expected_sha,
        worlds=worlds,
        perspectives=worlds * 2,
        cpu_replay_max_observation_error=replay_error,
        half_turn_max_observation_error=float(error.max()),
        fields_with_nonzero_roundoff={
            OBS_FIELD_NAMES[i]: float(v) for i, v in enumerate(error.amax((0, 1))) if v != 0
        },
        deterministic_action_changes=changed,
        maximum_logit_difference=float((first - second).abs().max()),
        maximum_value_difference=float((value - other_value).abs().max()),
        maximum_hidden_difference=float((hidden - other_hidden).abs().max()),
        checkpoint_assignment_snapshots=assignments,
        original_fixture_unchanged=True,
        model_unchanged=True,
        checkpoint_unchanged=True,
        torch_cuda_initialized=False,
        computation_device="cpu",
        optimizer_steps=0,
        native_rollout_steps=0,
        limitations=[
            "Archived 32-world reset fixture, not an all-state or live-physics proof.",
            "Zero recurrent history on both sides; does not explain later closed-loop "
            "trajectories.",
            "Checkpoint counts are instantaneous. Per-side cumulative decisions and role "
            "durations were not stored.",
            "Does not test Nexto native actor team canonicalization or post-kickoff timing.",
            "Does not test the external RLBot deployment observation bridge.",
        ],
    )
    output = RESULTS / "team_symmetry_diagnostic_000500.json"
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output.exists():
        assert json.loads(output.read_text()) == report, "diagnostic changed on rebuild"
    output.write_text(encoded)
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoint_assignment_snapshots"}))
    print(json.dumps(assignments))


if __name__ == "__main__":
    main()
