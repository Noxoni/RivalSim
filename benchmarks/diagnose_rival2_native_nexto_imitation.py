"""Measure a Rival checkpoint's native-state imitation of pinned Nexto actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_codex_autonomous_v1 import sha256_file, write_json
from rivalsim.full_match import FullMatchRunner
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ENVIRONMENT_CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
ENVIRONMENT_CHECKPOINT_SHA256 = (
    "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
)
PREVIOUS_ACTION_SLICE = slice(167, 175)


def _empty(device: torch.device) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "absolute_error": torch.zeros(8, device=device, dtype=torch.float64),
        "squared_error": torch.zeros(8, device=device, dtype=torch.float64),
        "button_correct": torch.zeros(3, device=device, dtype=torch.float64),
        "button_bce": torch.zeros(3, device=device, dtype=torch.float64),
    }


def _update(
    state: dict[str, torch.Tensor | int],
    actor: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = torch.cat((torch.tanh(actor[:, :5]), torch.sigmoid(actor[:, 10:13])), 1)
    error = prediction - target
    state["count"] = int(state["count"]) + target.shape[0]
    state["absolute_error"] += error.abs().sum(0, dtype=torch.float64)
    state["squared_error"] += error.square().sum(0, dtype=torch.float64)
    state["button_correct"] += (
        (prediction[:, 5:] >= 0.5) == (target[:, 5:] >= 0.5)
    ).sum(0, dtype=torch.float64)
    state["button_bce"] += F.binary_cross_entropy(
        prediction[:, 5:].clamp(1e-7, 1.0 - 1e-7),
        target[:, 5:],
        reduction="none",
    ).sum(0, dtype=torch.float64)


def _finish(state: dict[str, torch.Tensor | int]) -> dict[str, Any]:
    count = int(state["count"])
    absolute = state["absolute_error"] / count
    squared = state["squared_error"] / count
    return {
        "samples": count,
        "complete_action_mae": float(absolute.mean().item()),
        "complete_action_rmse": float(squared.mean().sqrt().item()),
        "per_channel_mae": absolute.detach().cpu().tolist(),
        "per_channel_rmse": squared.sqrt().detach().cpu().tolist(),
        "button_accuracy": (state["button_correct"] / count).detach().cpu().tolist(),
        "button_bce": (state["button_bce"] / count).detach().cpu().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2_026_090_219)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    args = parser.parse_args()
    expected = args.checkpoint_sha256.upper()
    if sha256_file(args.checkpoint) != expected:
        raise RuntimeError("diagnostic checkpoint SHA-256 mismatch")
    if sha256_file(ENVIRONMENT_CHECKPOINT) != ENVIRONMENT_CHECKPOINT_SHA256:
        raise RuntimeError("diagnostic environment checkpoint changed")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Rival2PolicyConfig(**payload["policy_config"])
    model = Rival2ActorCritic(config).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    layout = torch.randint(5, (args.worlds,), generator=generator).numpy()
    side = torch.randint(2, (args.worlds,), generator=generator).numpy()
    runner = FullMatchRunner(
        args.worlds,
        str(args.collision_root),
        ENVIRONMENT_CHECKPOINT,
        starting_layout=layout,
        rival_side=side,
        stochastic_rival=True,
        evaluation_seed=args.seed,
        device=args.device,
    )
    native = _empty(torch.device(args.device))
    neutral_previous = _empty(torch.device(args.device))
    with torch.inference_mode():
        for tick in range(1, args.ticks + 1):
            observation = runner.rival_observation[
                runner.batch_index, runner.nexto_side
            ].clone()
            runner.tick()
            if tick % args.sample_every:
                continue
            target = runner.nexto.previous_action
            actor, _ = model(observation)
            _update(native, actor, target)
            observation[:, PREVIOUS_ACTION_SLICE] = 0.0
            actor_zero, _ = model(observation)
            _update(neutral_previous, actor_zero, target)
    result = {
        "format": "RIVAL2_NATIVE_NEXTO_IMITATION_DIAGNOSTIC",
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": expected,
            "format": payload.get("format"),
        },
        "environment_checkpoint": {
            "path": ENVIRONMENT_CHECKPOINT.relative_to(ROOT).as_posix(),
            "sha256": ENVIRONMENT_CHECKPOINT_SHA256,
        },
        "teacher": "pinned deterministic Nexto exact consumed actions",
        "observation": "native RIVAL2_OBS_V2_120HZ immediately before action",
        "worlds": args.worlds,
        "physics_ticks": args.ticks,
        "sample_every_physics_ticks": args.sample_every,
        "seed": args.seed,
        "with_native_previous_action": _finish(native),
        "with_previous_action_forced_zero": _finish(neutral_previous),
        "policy_mutation": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
