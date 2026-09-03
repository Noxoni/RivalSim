"""Validate exact reconstruction and deterministic outcomes of V18 handoffs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from benchmarks import run_rival2_ground_to_air_selfplay_v12 as v12  # noqa: E402
from benchmarks.capture_rival2_ground_to_air_natural_handoffs_v18 import (  # noqa: E402
    OPTION_SHA256,
)
from benchmarks.run_rival2_ground_to_air_entry_probe_v11 import (  # noqa: E402
    human_envelope_config,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_entry_probe_v11 import (  # noqa: E402
    GroundToAirEntryProbeV11,
)
from rivalsim.rival2_natural_handoff_replay_v18 import (  # noqa: E402
    load_natural_handoff_corpus,
    restore_corpus_runtime,
    state_snapshot_from_corpus,
)
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_HANDOFF_VALIDATION_V18"
DEFAULT_CORPUS = (
    ROOT
    / "results/rival2/ground_to_air_natural_handoffs_v18"
    / "natural_handoffs.pt"
)
DEFAULT_OUTPUT = DEFAULT_CORPUS.with_name("validation.json")
DEFAULT_COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def collect_side(
    payload: dict[str, Any],
    model: torch.nn.Module,
    defenders: dict[int, torch.nn.Module],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    horizon: int,
    seed: int,
    device: str,
    collision_root: Path,
) -> dict[str, Any]:
    indices = torch.nonzero(payload["attacker_side"] == side).flatten()
    count = int(indices.numel())
    state = state_snapshot_from_corpus(payload, indices)
    lifecycle_rows = indices.numpy()
    kickoff_selector = payload["lifecycle"]["kickoff_selector"][lifecycle_rows]
    env = Rival2Env(
        count,
        str(collision_root),
        device=device,
        seed=seed ^ side,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    observation = restore_corpus_runtime(env, payload, indices)
    expected_observation = payload["observation"].index_select(0, indices).to(device)
    observation_difference = (observation - expected_observation).abs()

    other = 1 - side
    with torch.inference_mode():
        option_actor, _ = model(observation[:, side])
        first_option_action = deterministic_hybrid_action(
            option_actor, model.config
        )
        defender_actor, _ = defenders[other](observation[:, other])
        first_defender_action = deterministic_hybrid_action(
            defender_actor, defenders[other].config
        )
    expected_option = payload["option_action"].index_select(0, indices)[:, side].to(device)
    expected_base = payload["base_action"].index_select(0, indices)[:, other].to(device)

    setup = payload["route"].index_select(0, indices).to(device)
    probe = GroundToAirEntryProbeV11(
        setup,
        attacker_side=side,
        envelope_config=human_envelope_config(),
        continuation_ticks=min(horizon, 420),
        separation_ticks=4,
        maximum_contacts=6,
    )
    active = torch.ones(count, dtype=torch.bool, device=device)
    goals_for = torch.zeros((), dtype=torch.int64, device=device)
    goals_against = torch.zeros_like(goals_for)
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.inference_mode():
            actor, _ = model(observation[:, side])
            option_action = deterministic_hybrid_action(actor, model.config)
            defender_actor, _ = defenders[other](observation[:, other])
            defender_action = deterministic_hybrid_action(
                defender_actor, defenders[other].config
            )
        action = torch.zeros((count, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], option_action, 0.0)
        action[:, other] = torch.where(active_before[:, None], defender_action, 0.0)
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == side)
        goal_against = active_before & transition.terminated & (scoring_team == other)
        events = probe.step(
            observation,
            transition.transition_observation,
            tick=tick,
            active=active_before,
            goal_for_attacker=goal_for,
        )
        goals_for += goal_for.sum()
        goals_against += goal_against.sum()
        terminal = (
            transition.terminated
            | transition.truncated
            | events.ball_ground_failure
            | events.contact_budget_exceeded
        )
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "side": side,
        "samples": count,
        "source_indices": indices.tolist(),
        "observation_reconstruction": {
            "maximum_absolute_difference": float(observation_difference.max()),
            "nonzero_elements": int((observation_difference != 0).sum()),
            "exact": bool((observation_difference == 0).all()),
        },
        "first_action_reconstruction": {
            "option_maximum_absolute_difference": float(
                (first_option_action - expected_option).abs().max()
            ),
            "defender_maximum_absolute_difference": float(
                (first_defender_action - expected_base).abs().max()
            ),
        },
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "horizon_timeouts": int(active.sum()),
        "telemetry": probe.telemetry(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    corpus = args.corpus.resolve()
    corpus_sha256 = v12.sha256_file(corpus)
    payload = load_natural_handoff_corpus(corpus)
    if payload["sources"]["aerial_v3"] != OPTION_SHA256:
        raise RuntimeError("V18 corpus is not bound to the protected V3 scorer")
    option_payload = torch.load(
        natural_v4.PARENT, map_location="cpu", weights_only=False
    )
    model = natural_v4.make_model(option_payload, args.device).eval().requires_grad_(False)
    defenders = natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_root)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows = [
        collect_side(
            payload,
            model,
            defenders,
            geometry,
            meshes,
            side=side,
            horizon=args.horizon,
            seed=args.seed,
            device=args.device,
            collision_root=args.collision_root,
        )
        for side in (0, 1)
    ]
    total = sum(row["samples"] for row in rows)
    summary = {
        name: sum(
            row["samples"] * float(row["telemetry"]["fractions"][name])
            for row in rows
        )
        / total
        for name in (
            "entry_airborne_contact",
            "second_airborne_contact",
            "goal_within_contact_budget",
            "ball_ground_failure",
        )
    }
    summary.update(
        {
            "goals_for": sum(row["goals_for"] for row in rows),
            "goals_against": sum(row["goals_against"] for row in rows),
            "all_observations_exact": all(
                row["observation_reconstruction"]["exact"] for row in rows
            ),
            "maximum_observation_absolute_difference": max(
                row["observation_reconstruction"]["maximum_absolute_difference"]
                for row in rows
            ),
            "maximum_first_option_action_absolute_difference": max(
                row["first_action_reconstruction"][
                    "option_maximum_absolute_difference"
                ]
                for row in rows
            ),
        }
    )
    result = {
        "format": VERSION,
        "created_utc": utc_now(),
        "corpus": {
            "path": corpus.relative_to(ROOT).as_posix(),
            "sha256": corpus_sha256,
            "semantic_sha256": payload["semantic_sha256"],
        },
        "checkpoint": {
            "path": natural_v4.PARENT.relative_to(ROOT).as_posix(),
            "sha256": OPTION_SHA256,
            "unchanged": v12.sha256_file(natural_v4.PARENT) == OPTION_SHA256,
        },
        "horizon": args.horizon,
        "seed": args.seed,
        "rows": rows,
        "summary": summary,
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
        "verdict": "PASS" if summary["all_observations_exact"] else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if result["verdict"] == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2_026_090_329)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
