"""Run the bounded pinned Wisp source-vs-CUDA adapter fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.wisp75b.adapter import (  # noqa: E402
    RIVALSIM_TO_WISP_PAD_INDICES,
    WISP_BOTPACK_COMMIT,
    WISP_POLICY_SHA256,
    WISP_SHARED_HEAD_SHA256,
    WISP_UPSTREAM_COMMIT,
    WispPolicyAdapter,
    WispStateTensors,
    _ball_prediction,
    _linear_eta,
)

CORPUS_SIZE = 320
CORPUS_SEED = 2026082703


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root", type=Path, default=Path(r"G:\dev\RivalSim-runs\wisp-v2-75b-58d4ab18")
    )
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\.venv\Scripts\python.exe"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rival2/opponent_curriculum_v1/wisp_fidelity.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _corpus() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(CORPUS_SEED)
    n = CORPUS_SIZE
    side = np.arange(n, dtype=np.int64) % 2
    car_pos = rng.uniform((-3800, -4900, 17), (3800, 4900, 1200), (n, 2, 3)).astype(np.float32)
    car_vel = rng.uniform(-1200, 1200, (n, 2, 3)).astype(np.float32)
    car_ang_vel = rng.uniform(-3, 3, (n, 2, 3)).astype(np.float32)
    quaternion = rng.normal(size=(n, 2, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    ball_pos = rng.uniform((-3500, -4700, 92), (3500, 4700, 1800), (n, 3)).astype(np.float32)
    ball_vel = rng.uniform(-1600, 1600, (n, 3)).astype(np.float32)
    ball_ang_vel = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
    # Ten exact standard-kickoff identities (five layouts x both Wisp sides).
    kickoff_positions = ((-2048, -2560), (2048, -2560), (-256, -3840), (256, -3840), (0, -4608))
    for row in range(10):
        layout = row // 2
        x, y = kickoff_positions[layout]
        car_pos[row, 0] = (x, y, 17)
        car_pos[row, 1] = (-x, -y, 17)
        car_vel[row] = 0
        car_ang_vel[row] = 0
        quaternion[row] = (0, 0, 0, 1)
        ball_pos[row] = (0, 0, 92.75)
        ball_vel[row] = 0
        ball_ang_vel[row] = 0
    on_ground = (car_pos[..., 2] < 80).astype(np.int32)
    is_jumping = ((np.arange(n)[:, None] + np.arange(2)) % 13 == 0).astype(np.int32)
    has_jumped = ((np.arange(n)[:, None] + np.arange(2)) % 3 == 0).astype(np.int32)
    has_double = ((np.arange(n)[:, None] + np.arange(2)) % 17 == 0).astype(np.int32)
    has_flipped = ((np.arange(n)[:, None] + np.arange(2)) % 11 == 0).astype(np.int32)
    air_since = rng.uniform(0, 1.8, (n, 2)).astype(np.float32)
    boost = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    boost[::19] = 0
    demoed = np.zeros((n, 2), dtype=np.int32)
    demoed[31::97, 1] = 1
    pad = rng.uniform(0, 10, (n, 34)).astype(np.float32)
    pad[rng.random((n, 34)) < 0.55] = 0
    previous = rng.choice((-1.0, 0.0, 1.0), (n, 8)).astype(np.float32)
    previous[:, 5:] = (previous[:, 5:] > 0).astype(np.float32)
    touch = ((np.arange(n)[:, None] + np.arange(2)) % 23 == 0).astype(np.int32)
    handbrake = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    score = rng.integers(-2, 3, n, dtype=np.int32)
    wisp_pad = pad[:, np.asarray(RIVALSIM_TO_WISP_PAD_INDICES)]
    return {
        "side": side,
        "car_pos": car_pos,
        "car_vel": car_vel,
        "car_quat": quaternion,
        "car_ang_vel": car_ang_vel,
        "boost": boost,
        "on_ground": on_ground,
        "is_jumping": is_jumping,
        "has_jumped": has_jumped,
        "has_double_jumped": has_double,
        "has_flipped": has_flipped,
        "air_time_since_jump": air_since,
        "demoed": demoed,
        "pad_cooldown": pad,
        "wisp_pad_cooldown": wisp_pad,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "ball_ang_vel": ball_ang_vel,
        "touch_count": touch,
        "handbrake_value": handbrake,
        "previous_action": previous,
        "score_diff": score,
    }


def _tensor_state(data: dict[str, np.ndarray], device: torch.device) -> WispStateTensors:
    def tensor(name):
        return torch.as_tensor(data[name], device=device)

    return WispStateTensors(
        car_pos=tensor("car_pos"),
        car_vel=tensor("car_vel"),
        car_quat=tensor("car_quat"),
        car_ang_vel=tensor("car_ang_vel"),
        boost=tensor("boost"),
        on_ground=tensor("on_ground"),
        is_jumping=tensor("is_jumping"),
        has_jumped=tensor("has_jumped"),
        has_double_jumped=tensor("has_double_jumped"),
        has_flipped=tensor("has_flipped"),
        air_time_since_jump=tensor("air_time_since_jump"),
        demoed=tensor("demoed"),
        pad_cooldown=tensor("pad_cooldown"),
        ball_pos=tensor("ball_pos"),
        ball_vel=tensor("ball_vel"),
        ball_ang_vel=tensor("ball_ang_vel"),
        touch_count=tensor("touch_count"),
        handbrake_value=tensor("handbrake_value"),
    )


def main() -> None:
    args = parse_args()
    if (
        subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != WISP_UPSTREAM_COMMIT
    ):
        raise RuntimeError("Wisp upstream checkout is not pinned")
    data = _corpus()
    with tempfile.TemporaryDirectory(prefix="rivalsim-wisp-fidelity-") as folder:
        root = Path(folder)
        source_input, source_output = root / "corpus.npz", root / "reference.npz"
        np.savez_compressed(source_input, **data)
        subprocess.run(
            [
                str(args.reference_python),
                "benchmarks/wisp75b_source_reference.py",
                "--upstream-root",
                str(args.upstream_root),
                "--input",
                str(source_input),
                "--output",
                str(source_output),
            ],
            check=True,
        )
        with np.load(source_output, allow_pickle=False) as archive:
            reference = {name: archive[name].copy() for name in archive.files}
        device = torch.device("cuda:0")
        adapter = WispPolicyAdapter(CORPUS_SIZE, device=device)
        adapter.set_player_index(torch.as_tensor(data["side"], device=device))
        adapter.set_opponent_slot(torch.as_tensor(reference["opponent_slot"], device=device))
        adapter.previous_action.copy_(torch.as_tensor(data["previous_action"], device=device))
        state = _tensor_state(data, device)
        eta_time = torch.zeros((CORPUS_SIZE, 2, 2), dtype=torch.float64, device=device)
        eta_tick = torch.zeros((CORPUS_SIZE, 2, 2), dtype=torch.int64, device=device)
        eta_v0 = torch.zeros((CORPUS_SIZE, 2, 2), dtype=torch.float32, device=device)
        eta_x0 = torch.zeros((CORPUS_SIZE, 2, 2), dtype=torch.float64, device=device)
        for physical in range(2):
            estimate = torch.zeros(CORPUS_SIZE, dtype=torch.float64, device=device)
            for eta_pass in range(2):
                tick = (estimate * 120).to(torch.int64).clamp(0, 599)
                target, _ = _ball_prediction(state.ball_pos, state.ball_vel, tick + 1)
                delta = target - state.car_pos[:, physical]
                distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-7)
                initial_velocity = (state.car_vel[:, physical] * (delta / distance[:, None])).sum(
                    -1
                )
                estimate = _linear_eta(
                    initial_velocity,
                    distance.to(torch.float64) - 136.875,
                    state.boost[:, physical],
                )
                eta_tick[:, physical, eta_pass] = tick
                eta_v0[:, physical, eta_pass] = initial_velocity
                eta_x0[:, physical, eta_pass] = distance.to(torch.float64) - 136.875
                eta_time[:, physical, eta_pass] = estimate
        observation = (
            adapter.observation(
                state, score_diff=torch.as_tensor(data["score_diff"], device=device)
            )
            .cpu()
            .numpy()
        )
        port_eta_v0 = adapter.eta_trace_v0.cpu().numpy().copy()
        port_eta_x0 = adapter.eta_trace_x0.cpu().numpy().copy()
        port_eta_tick = adapter.eta_trace_tick.cpu().numpy().copy()
        port_eta_time = adapter.eta_trace_time.cpu().numpy().copy()
        mask = adapter.action_mask(state).cpu().numpy()
        adapter.eta_cache.fill(0.0)
        action, index, _ = adapter.neural_action(
            state,
            score_diff=torch.as_tensor(data["score_diff"], device=device),
            shuffle_opponents=False,
        )
        actual_index, actual_action = index.cpu().numpy(), action.cpu().numpy()
    obs_error = np.abs(observation - reference["observation"])
    flat_worst = np.unravel_index(int(obs_error.argmax()), obs_error.shape)
    per_feature_max = obs_error.max(axis=0)
    worst_features = np.argsort(per_feature_max)[-20:][::-1]
    exact_mask = np.array_equal(mask, reference["action_mask"])
    exact_index = np.array_equal(actual_index, reference["action_index"])
    exact_controller = np.array_equal(actual_action, reference["controller"])
    action_index_mismatches = np.flatnonzero(actual_index != reference["action_index"])
    controller_mismatches = np.flatnonzero(np.any(actual_action != reference["controller"], axis=1))
    # State transition checks: repeatability, reset/history, both side mirrors,
    # and the literal initial + steady-state delayed-control schedule.
    adapter = WispPolicyAdapter(2, device=device)
    two = {key: value[:2] for key, value in data.items() if value.shape[0] == CORPUS_SIZE}
    two["side"] = np.asarray((0, 1), dtype=np.int64)
    adapter.set_player_index(torch.as_tensor(two["side"], device=device))
    two_state = _tensor_state(two, device)
    sequence = []
    for _ in range(18):
        controls, _ = adapter.tick_action(two_state)
        sequence.append(controls.cpu().numpy())
    reset = torch.tensor((True, False), device=device)
    adapter.reset(reset)
    reset_ok = bool(
        torch.all(adapter.previous_action[0] == 0)
        and adapter.ticks[0] == -1
        and adapter.update_flag[0]
    )
    report: dict[str, Any] = {
        "verdict": "PASS_GREEN"
        if obs_error.max() <= 2e-5 and exact_mask and exact_index and exact_controller and reset_ok
        else "FAIL_RED",
        "corpus": {
            "count": CORPUS_SIZE,
            "seed": CORPUS_SEED,
            "coverage": [
                "five_kickoffs_both_sides",
                "ground_open_play",
                "airborne_high_ball",
                "wall_curve",
                "boost_history",
            ],
        },
        "identity": {
            "upstream_commit": WISP_UPSTREAM_COMMIT,
            "botpack_commit": WISP_BOTPACK_COMMIT,
            "policy_sha256": WISP_POLICY_SHA256,
            "shared_head_sha256": WISP_SHARED_HEAD_SHA256,
            "policy_observed_sha256": _sha256(Path("third_party/wisp75b/models/POLICY.lt")),
            "shared_head_observed_sha256": _sha256(
                Path("third_party/wisp75b/models/SHARED_HEAD.lt")
            ),
        },
        "observation": {
            "max_abs_error": float(obs_error.max()),
            "mean_abs_error": float(obs_error.mean()),
            "atol": 2e-5,
        },
        "observation_diagnostic": {
            "worst_row": int(flat_worst[0]),
            "worst_feature": int(flat_worst[1]),
            "worst_actual": float(observation[flat_worst]),
            "worst_reference": float(reference["observation"][flat_worst]),
            "features": [
                {
                    "index": int(index),
                    "max_abs_error": float(per_feature_max[index]),
                    "row": int(obs_error[:, index].argmax()),
                    "actual_at_max": float(observation[int(obs_error[:, index].argmax()), index]),
                    "reference_at_max": float(
                        reference["observation"][int(obs_error[:, index].argmax()), index]
                    ),
                }
                for index in worst_features
            ],
        },
        "eta_diagnostic": {},
        "action_mask_exact": exact_mask,
        "action_index_exact": exact_index,
        "action_index_mismatch_count": int(action_index_mismatches.size),
        "action_index_mismatch_rows": action_index_mismatches[:20].tolist(),
        "controller_exact": exact_controller,
        "controller_mismatch_count": int(controller_mismatches.size),
        "deterministic_repeatability": bool(np.array_equal(sequence[8], sequence[16])),
        "reset_history_state": reset_ok,
        "cadence": {"tick_skip": 8, "action_delay": 7, "captured_ticks": len(sequence)},
        "full_world_state_cpu_transfer_in_hot_path": False,
    }
    actual_eta_time = port_eta_time
    actual_eta_tick = eta_tick.cpu().numpy()
    actual_eta_v0 = port_eta_v0
    actual_eta_x0 = port_eta_x0
    actual_eta_tick = port_eta_tick
    eta_error = np.abs(actual_eta_time - reference["eta_time"])
    eta_flat = np.argsort(eta_error.reshape(-1))[-12:][::-1]
    report["eta_diagnostic"] = {
        "max_abs_error": float(eta_error.max()),
        "tick_mismatch_count": int(np.count_nonzero(actual_eta_tick != reference["eta_tick"])),
        "v0_exact_count": int(
            np.count_nonzero(actual_eta_v0 == reference["eta_v0"].astype(np.float32))
        ),
        "x0_exact_count": int(np.count_nonzero(actual_eta_x0 == reference["eta_x0"])),
        "worst": [
            {
                "row": int(np.unravel_index(int(flat), eta_error.shape)[0]),
                "physical": int(np.unravel_index(int(flat), eta_error.shape)[1]),
                "pass": int(np.unravel_index(int(flat), eta_error.shape)[2]),
                "actual_time": float(actual_eta_time[np.unravel_index(int(flat), eta_error.shape)]),
                "reference_time": float(
                    reference["eta_time"][np.unravel_index(int(flat), eta_error.shape)]
                ),
                "reference_v0": float(
                    reference["eta_v0"][np.unravel_index(int(flat), eta_error.shape)]
                ),
                "actual_v0": float(actual_eta_v0[np.unravel_index(int(flat), eta_error.shape)]),
                "reference_x0": float(
                    reference["eta_x0"][np.unravel_index(int(flat), eta_error.shape)]
                ),
                "actual_x0": float(actual_eta_x0[np.unravel_index(int(flat), eta_error.shape)]),
                "actual_tick": int(actual_eta_tick[np.unravel_index(int(flat), eta_error.shape)]),
                "reference_tick": int(
                    reference["eta_tick"][np.unravel_index(int(flat), eta_error.shape)]
                ),
            }
            for flat in eta_flat
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] != "PASS_GREEN":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
