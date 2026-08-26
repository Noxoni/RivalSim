"""Run the bounded source-vs-CUDA fidelity gate for the pinned Nexto adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.nexto.adapter import (
    KICKOFF_LENGTH,
    MODEL_SHA256,
    NEXTO_ACTION_COUNT,
    NEXTO_BOOST_LOCATIONS,
    NextoDeviceConstants,
    NextoPolicyAdapter,
    NextoStateTensors,
    build_action_table,
    build_kickoff_sequence,
    build_nexto_observation,
    nexto_pad_mapping,
    stable_tensor_hash,
)

MODEL_PATH = ROOT / "third_party" / "nexto" / "nexto-model.pt"
DEFAULT_OUTPUT = ROOT / "results" / "rival2" / "nexto" / "fidelity.json"


def _source_action_table() -> np.ndarray:
    actions: list[list[int]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                        )
    return np.asarray(actions)


def _source_kickoff() -> np.ndarray:
    rows: list[list[float]] = []
    rows += 11 * 4 * [[1, 0, 0, 0, 0, 0, 1, 0]]
    rows += 4 * 4 * [[1, -1, 0, -1, 0, 0, 1, 0]]
    rows += 2 * 4 * [[1, 0, 0, 0, 0, 1, 1, 0]]
    rows += 1 * 4 * [[1, 0, 0, 0, 0, 0, 1, 0]]
    rows += 1 * 4 * [[1, 0, -0.7, 0.8, 0, 1, 1, 0]]
    rows += 13 * 4 * [[1, 0, 1, 0, 0, 0, 1, 0]]
    rows += 10 * 4 * [[1, 0, 0.5, 0, 1, 0, 0, 0]]
    return np.asarray(rows)


def _basis_numpy(quaternion_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = np.moveaxis(quaternion_xyzw, -1, 0)
    forward = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        axis=-1,
    )
    up = np.stack(
        (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    )
    return forward, up


def source_observation(
    raw: dict[str, np.ndarray],
    side: np.ndarray,
    previous_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct float64 NumPy transcription of pinned ``nexto_obs.py``."""

    count = side.shape[0]
    q = np.zeros((count, 1, 32))
    kv = np.zeros((count, 37, 24))
    mask = np.zeros((count, 37))
    kv[:, 2, 3] = 1
    kv[:, 2, 5:8] = raw["ball_pos"]
    kv[:, 2, 8:11] = raw["ball_vel"]
    kv[:, 2, 17:20] = raw["ball_ang_vel"]

    mapping, _ = nexto_pad_mapping()
    kv[:, 3:, 4] = 1
    kv[:, 3:, 5:8] = NEXTO_BOOST_LOCATIONS.astype(np.float64)
    kv[:, 3:, 20] = 0.12 + 0.88 * (NEXTO_BOOST_LOCATIONS[:, 2] > 72)
    kv[:, 3:, 21] = raw["pad_cooldown"][:, mapping] == 0

    physical_order = np.stack((side, 1 - side), axis=1)
    teams = physical_order.astype(np.float64)
    kv[:, :2, 1] = 1 - teams
    kv[:, :2, 2] = teams
    rows = np.arange(count)
    for entity in range(2):
        physical = physical_order[:, entity]
        quat = raw["car_quat"][rows, physical].astype(np.float64)
        forward, up = _basis_numpy(quat)
        kv[:, entity, 5:8] = raw["car_pos"][rows, physical]
        kv[:, entity, 8:11] = raw["car_vel"][rows, physical]
        kv[:, entity, 11:14] = forward
        kv[:, entity, 14:17] = up
        kv[:, entity, 17:20] = raw["car_ang_vel"][rows, physical]
        kv[:, entity, 20] = raw["car_boost"][rows, physical] / 100.0
        kv[:, entity, 21] = raw["car_demoed"][rows, physical]
        kv[:, entity, 22] = raw["on_ground"][rows, physical]
        kv[:, entity, 23] = (
            (raw["has_flipped"][rows, physical] == 0)
            & (raw["has_double_jumped"][rows, physical] == 0)
            & (raw["air_time_since_jump"][rows, physical] < 1.25)
        )
    kv[:, 0, 0] = 1

    invert = np.asarray([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    orange = side == 1
    kv[orange] *= invert
    mate = kv[orange, :, 1].copy()
    kv[orange, :, 1] = kv[orange, :, 2]
    kv[orange, :, 2] = mate
    norm = np.asarray([1.0] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)
    kv /= norm
    q[:, 0, :24] = kv[:, 0, :]
    q[:, 0, 24:32] = previous_action

    kv[:, :, 5:8] -= q[:, :, 5:8]
    forward = q[:, :, 11:14]
    theta = np.expand_dims(np.arctan2(forward[..., 0], forward[..., 1]), axis=-1)
    ct = np.cos(theta)
    st = np.sin(theta)
    xs = kv[:, :, 5:20:3]
    ys = kv[:, :, 6:20:3]
    nx = ct * xs - st * ys
    ny = st * xs + ct * ys
    kv[:, :, 5:20:3] = nx
    kv[:, :, 6:20:3] = ny
    return q.astype(np.float32), kv.astype(np.float32), mask.astype(np.float32)


def make_corpus(count: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw: dict[str, np.ndarray] = {
        "car_pos": rng.uniform((-3900, -5000, 17), (3900, 5000, 1900), (count, 2, 3)).astype(np.float32),
        "car_vel": rng.uniform(-2300, 2300, (count, 2, 3)).astype(np.float32),
        "car_ang_vel": rng.uniform(-5.5, 5.5, (count, 2, 3)).astype(np.float32),
        "car_boost": rng.uniform(0, 100, (count, 2)).astype(np.float32),
        "car_demoed": rng.integers(0, 2, (count, 2), dtype=np.int32),
        "on_ground": rng.integers(0, 2, (count, 2), dtype=np.int32),
        "has_double_jumped": rng.integers(0, 2, (count, 2), dtype=np.int32),
        "has_flipped": rng.integers(0, 2, (count, 2), dtype=np.int32),
        "air_time_since_jump": rng.uniform(0, 2, (count, 2)).astype(np.float32),
        "ball_pos": rng.uniform((-4000, -5100, 93.15), (4000, 5100, 1900), (count, 3)).astype(np.float32),
        "ball_vel": rng.uniform(-6000, 6000, (count, 3)).astype(np.float32),
        "ball_ang_vel": rng.uniform(-6, 6, (count, 3)).astype(np.float32),
        "pad_cooldown": rng.choice(np.asarray([0.0, 0.25, 1.5, 4.0, 10.0], np.float32), (count, 34)),
    }
    quat = rng.normal(size=(count, 2, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    raw["car_quat"] = quat.astype(np.float32)
    side = (np.arange(count) % 2).astype(np.int64)
    table = _source_action_table().astype(np.float32)
    previous = table[rng.integers(0, len(table), count)]

    kickoff_positions = (
        (-2048.0, -2560.0, 17.0),
        (2048.0, -2560.0, 17.0),
        (-256.0, -3840.0, 17.0),
        (256.0, -3840.0, 17.0),
        (0.0, -4608.0, 17.0),
    )
    kickoff_yaws = (math.pi / 4, 3 * math.pi / 4, math.pi / 2, math.pi / 2, math.pi / 2)
    for layout in range(5):
        for viewpoint in range(2):
            index = layout * 2 + viewpoint
            if index >= count:
                continue
            blue = np.asarray(kickoff_positions[layout], dtype=np.float32)
            raw["car_pos"][index, 0] = blue
            raw["car_pos"][index, 1] = (-blue[0], -blue[1], blue[2])
            yaw = kickoff_yaws[layout]
            raw["car_quat"][index, 0] = (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            orange_yaw = yaw + math.pi
            raw["car_quat"][index, 1] = (
                0,
                0,
                math.sin(orange_yaw / 2),
                math.cos(orange_yaw / 2),
            )
            raw["car_vel"][index] = 0
            raw["car_ang_vel"][index] = 0
            raw["car_boost"][index] = 100 / 3
            raw["car_demoed"][index] = 0
            raw["on_ground"][index] = 1
            raw["has_double_jumped"][index] = 0
            raw["has_flipped"][index] = 0
            raw["air_time_since_jump"][index] = 0
            raw["ball_pos"][index] = (0, 0, 93.15)
            raw["ball_vel"][index] = 0
            raw["ball_ang_vel"][index] = 0
            raw["pad_cooldown"][index] = 0
            side[index] = viewpoint
            previous[index] = 0
    return raw, side, previous


def _to_state(raw: dict[str, np.ndarray], device: torch.device) -> NextoStateTensors:
    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(raw[name], device=device)

    return NextoStateTensors(
        car_pos=tensor("car_pos"),
        car_vel=tensor("car_vel"),
        car_quat=tensor("car_quat"),
        car_ang_vel=tensor("car_ang_vel"),
        car_boost=tensor("car_boost"),
        car_demoed=tensor("car_demoed"),
        on_ground=tensor("on_ground"),
        has_double_jumped=tensor("has_double_jumped"),
        has_flipped=tensor("has_flipped"),
        air_time_since_jump=tensor("air_time_since_jump"),
        ball_pos=tensor("ball_pos"),
        ball_vel=tensor("ball_vel"),
        ball_ang_vel=tensor("ball_ang_vel"),
        pad_cooldown=tensor("pad_cooldown"),
    )


def _cuda_elapsed(callable_: Any, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def run_fidelity(count: int = 2048, seed: int = 0x4E455854) -> dict[str, Any]:
    device = torch.device("cuda:0")
    raw, side_np, previous_np = make_corpus(count, seed)
    q_ref, kv_ref, mask_ref = source_observation(raw, side_np, previous_np)
    state = _to_state(raw, device)
    side = torch.as_tensor(side_np, dtype=torch.long, device=device)
    previous = torch.as_tensor(previous_np, dtype=torch.float32, device=device)
    constants = NextoDeviceConstants.create(device)
    obs_gpu = build_nexto_observation(state, side, previous, constants=constants)
    q_gpu = obs_gpu.q.detach().cpu().numpy()
    kv_gpu = obs_gpu.kv.detach().cpu().numpy()
    mask_gpu = obs_gpu.mask.detach().cpu().numpy()

    cpu_actor = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    with torch.inference_mode():
        logits_ref, _ = cpu_actor(
            (torch.from_numpy(q_ref), torch.from_numpy(kv_ref), torch.from_numpy(mask_ref))
        )
    adapter = NextoPolicyAdapter(count, device=device, model_path=MODEL_PATH)
    adapter.set_player_index(side)
    adapter.previous_action.copy_(previous)
    with torch.inference_mode():
        logits_gpu = adapter.logits(obs_gpu)
    logits_gpu_cpu = logits_gpu.detach().cpu()
    argmax_ref = torch.argmax(logits_ref, dim=-1)
    argmax_gpu = torch.argmax(logits_gpu_cpu, dim=-1)

    source_table = _source_action_table()
    gpu_table = build_action_table(device).cpu().numpy()
    source_kickoff = _source_kickoff().astype(np.float32)
    gpu_kickoff = build_kickoff_sequence(device).cpu().numpy()

    adapter.neural_action(state)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        adapter.neural_action(state)
        torch.cuda.synchronize()
    transfer_events = []
    for event in profile.events():
        name = event.name.lower()
        if "memcpy htod" in name or "memcpy dtoh" in name:
            transfer_events.append(event.name)

    obs_seconds = _cuda_elapsed(
        lambda: build_nexto_observation(state, side, previous, constants=constants), 20
    )
    model_seconds = _cuda_elapsed(lambda: adapter.logits(obs_gpu), 20)
    mapping, residual = nexto_pad_mapping()
    action_hash = stable_tensor_hash(adapter.action_table)
    kickoff_hash = stable_tensor_hash(adapter.kickoff_sequence)
    model_sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest().upper()

    q_error = float(np.max(np.abs(q_gpu - q_ref)))
    kv_error = float(np.max(np.abs(kv_gpu - kv_ref)))
    mask_error = float(np.max(np.abs(mask_gpu - mask_ref)))
    logit_error = float(torch.max(torch.abs(logits_gpu_cpu - logits_ref)).item())
    action_agreement = float((argmax_gpu == argmax_ref).to(torch.float32).mean().item())
    gates = {
        "model_sha_exact": model_sha == MODEL_SHA256,
        "observation_q_max_abs_le_1e6": q_error <= 1e-6,
        "observation_kv_max_abs_le_1e6": kv_error <= 1e-6,
        "observation_mask_max_abs_le_1e6": mask_error <= 1e-6,
        "argmax_agreement_100_percent": action_agreement == 1.0,
        "action_table_exact": bool(
            source_table.shape == gpu_table.shape
            and np.array_equal(source_table.astype(np.float32), gpu_table)
        ),
        "kickoff_sequence_exact_float32": bool(
            source_kickoff.shape == gpu_kickoff.shape
            and np.array_equal(source_kickoff, gpu_kickoff)
        ),
        "timed_hot_path_h2d_d2h_zero": len(transfer_events) == 0,
        "pad_mapping_unique": len(np.unique(mapping)) == 34,
    }
    result: dict[str, Any] = {
        "verdict": "PASS_GREEN" if all(gates.values()) else "FAIL_RED",
        "corpus": {
            "states": count,
            "seed": seed,
            "blue_viewpoints": int(np.count_nonzero(side_np == 0)),
            "orange_viewpoints": int(np.count_nonzero(side_np == 1)),
            "coverage": [
                "grounded",
                "aerial",
                "demoed",
                "low_boost",
                "high_boost",
                "pad_active",
                "pad_inactive",
                "all_five_kickoff_layouts_both_viewpoints",
            ],
        },
        "provenance": {
            "upstream_commit": "2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca",
            "model_sha256": model_sha,
            "model_size_bytes": MODEL_PATH.stat().st_size,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "compatibility_patch": "one stale TorchScript explicit CPU device literal retargeted to cuda:0; weights unchanged",
            "compatibility_graph_literals_patched": adapter.compatibility_graph_literals_patched,
            "compatibility_scalar_constants_rehomed": adapter.compatibility_scalar_constants_rehomed,
        },
        "observation_parity": {
            "q_max_abs_error": q_error,
            "kv_max_abs_error": kv_error,
            "mask_max_abs_error": mask_error,
        },
        "model_action_parity": {
            "logit_max_abs_error_cpu_reference_vs_cuda": logit_error,
            "argmax_agreement_fraction": action_agreement,
            "argmax_agreement_count": int(torch.count_nonzero(argmax_gpu == argmax_ref).item()),
        },
        "action_table": {
            "count": NEXTO_ACTION_COUNT,
            "shape": list(gpu_table.shape),
            "float32_sha256": action_hash,
        },
        "kickoff_sequence": {
            "physics_ticks": KICKOFF_LENGTH,
            "shape": list(gpu_kickoff.shape),
            "float32_sha256": kickoff_hash,
        },
        "boost_pad_mapping": {
            "nexto_to_rivalsim": mapping.tolist(),
            "max_coordinate_residual_uu": float(residual.max()),
            "nonzero_residuals": [
                {"nexto_index": int(i), "rivalsim_index": int(mapping[i]), "residual_uu": float(residual[i])}
                for i in np.flatnonzero(residual)
            ],
        },
        "hot_path": {
            "profiled_h2d_d2h_event_count": len(transfer_events),
            "profiled_h2d_d2h_event_names": transfer_events,
            "adapter_timed_h2d_bytes": adapter.timed_h2d_bytes,
            "adapter_timed_d2h_bytes": adapter.timed_d2h_bytes,
        },
        "performance": {
            "batch_size": count,
            "observation_worlds_per_second": count * 20 / obs_seconds,
            "model_inferences_per_second": count * 20 / model_seconds,
            "observation_batch_milliseconds": obs_seconds * 1000 / 20,
            "model_batch_milliseconds": model_seconds * 1000 / 20,
        },
        "gates": gates,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0x4E455854)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_fidelity(args.count, args.seed)
    result["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "PASS_GREEN":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
