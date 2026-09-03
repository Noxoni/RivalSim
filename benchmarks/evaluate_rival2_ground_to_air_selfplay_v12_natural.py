"""Deterministic natural V23 self-play evaluation of the selected V12 option."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.census_rival2_v23_ground_to_air_selfplay import (  # noqa: E402
    BLUE,
    BLUE_SHA256,
    COLLISION_ROOT,
    ORANGE,
    ORANGE_SHA256,
    SideSpecializedSelfPlayRunner,
    sha256_file,
    write_json,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    AerialOptionRouterConfig,
    AerialOptionSelfPlayRouter,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_NATURAL_EVALUATION"
SELECTED = (
    ROOT
    / "checkpoints/rival2/ground_to_air_selfplay_v12"
    / "rival2_ground_to_air_selfplay_v12_u0060.pt"
)
SELECTED_SHA256 = "0A80DD35040D5FE354240D4E4E4F4B2CD50EB342CC95985647D3B0947DB092B2"
OUTPUT = ROOT / "results/rival2/ground_to_air_selfplay_v12/natural_selfplay_u0060.json"


class V12NaturalSelfPlayRunner(SideSpecializedSelfPlayRunner):
    """Both V23 sides share the selected direct aerial option and router."""

    def __init__(
        self,
        *args: Any,
        option_checkpoint: Path = SELECTED,
        option_sha256: str = SELECTED_SHA256,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        option_checkpoint = Path(option_checkpoint).resolve()
        observed_sha256 = sha256_file(option_checkpoint)
        if observed_sha256 != str(option_sha256).upper():
            raise RuntimeError("natural self-play option identity mismatch")
        payload = torch.load(option_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") not in {
            "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_CHECKPOINT",
            "RIVAL2_GROUND_TO_AIR_SELF_IMITATION_V13_CHECKPOINT",
        }:
            raise RuntimeError("unexpected aerial-option checkpoint format")
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != self.checkpoint_identity["policy_config"]:
            raise RuntimeError("selected option architecture differs from V23")
        self.option_policy = Rival2ActorCritic(config).to(self.device)
        self.option_policy.load_state_dict(payload["model"], strict=True)
        self.option_policy.eval().requires_grad_(False)
        self.aerial_router = AerialOptionSelfPlayRouter(
            self.num_worlds * 2,
            device=self.device,
            router_config=AerialOptionRouterConfig(**payload["router_config"]),
            reward_config=AerialSelfPlayRewardConfig(**payload["aerial_reward_config"]),
        )
        self._goal_scored = wp.to_torch(self.world.lifecycle.goal_scored)
        self._scoring_team = wp.to_torch(self.world.lifecycle.scoring_team).to(torch.int64)
        self._before = torch.zeros_like(self.rival_observation)
        self._active = torch.zeros(
            (self.num_worlds, 2), dtype=torch.bool, device=self.device
        )

    def _update_all_actions(self) -> None:
        observation = self.rival_observation
        flat = observation.reshape(-1, 182)
        lifecycle = self.match_views["kickoff_active"][:, None].expand(-1, 2)
        done = self.match_views["done"][:, None].expand(-1, 2)
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation[:, 0])
            orange_actor, _ = self.orange_policy(observation[:, 1])
            base_actions = torch.stack(
                (
                    deterministic_hybrid_action(blue_actor),
                    deterministic_hybrid_action(orange_actor),
                ),
                dim=1,
            )
            option_actor, _ = self.option_policy(flat)
            option_actions = deterministic_hybrid_action(option_actor).reshape(
                self.num_worlds, 2, 8
            )
            selection = self.aerial_router.select(
                flat,
                kickoff_active=(lifecycle != 0).reshape(-1),
                match_done=(done != 0).reshape(-1),
            )
            active = selection.active.reshape(self.num_worlds, 2)
            self.actions.copy_(torch.where(active[..., None], option_actions, base_actions))
            self._before.copy_(observation)
            self._active.copy_(active)

    def tick(self) -> None:
        self._activate_stream()
        self._update_all_actions()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        after = self.bridge.observation()
        side = torch.arange(2, dtype=torch.int64, device=self.device)[None, :]
        goal_for = (self._goal_scored[:, None] != 0) & (
            self._scoring_team[:, None] == side
        )
        self.aerial_router.observe(
            self._before.reshape(-1, 182),
            after.reshape(-1, 182),
            active_before=self._active.reshape(-1),
            goal_for_lane=goal_for.reshape(-1),
        )
        self.match_views["rival_scheduler_tick"].add_(1).remainder_(
            self.rival_cadence_ticks
        )
        self.match_views["nexto_scheduler_tick"].add_(1).remainder_(1)
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            wp.copy(self.world.rival2.reset_mask, self.match.pending_reset)
            self.world.apply_interval_resets()
            self.telemetry.after_resets(self.world, self.world.rival2.reset_mask)
        self.rival_observation = self.bridge.observation()


def evaluate_checkpoint(
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    worlds: int,
    ticks: int,
    seed: int,
    device: str,
    collision_root: Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    identities = {
        "blue_v23": sha256_file(BLUE),
        "orange_v23": sha256_file(ORANGE),
        "aerial_option": sha256_file(checkpoint),
    }
    expected = {
        "blue_v23": BLUE_SHA256,
        "orange_v23": ORANGE_SHA256,
        "aerial_option": str(checkpoint_sha256).upper(),
    }
    if identities != expected:
        raise RuntimeError(f"natural self-play identity mismatch: {identities}")
    runner = V12NaturalSelfPlayRunner(
        worlds,
        str(Path(collision_root).resolve()),
        BLUE,
        starting_layout=np.arange(worlds, dtype=np.int32) % 5,
        rival_side=np.arange(worlds, dtype=np.int32) % 2,
        stochastic_rival=False,
        evaluation_seed=int(seed),
        orange_checkpoint=ORANGE,
        option_checkpoint=checkpoint,
        option_sha256=checkpoint_sha256,
        device=device,
    )
    timing = runner.run_ticks(int(ticks))
    raw = runner.export()["raw"]
    touch_count = raw["touch_count"]
    demo_count = raw["demo_count"]
    status = runner.phase_status()
    router = runner.aerial_router.telemetry()
    player_minutes = worlds * 2 * int(ticks) / (120.0 * 60.0)
    payload = {
        "format": VERSION,
        "diagnostic_only": True,
        "policy_mutation": False,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "checkpoint": identities,
        "worlds": worlds,
        "ticks": int(ticks),
        "seed": int(seed),
        "timing_seconds": timing.seconds,
        "touches": {
            "total": int(touch_count.sum()),
            "per_player_minute": float(touch_count.sum() / player_minutes),
            "players_without_touch": int((touch_count == 0).sum()),
        },
        "scoring": {
            "blue": int(status["blue_score"].sum()),
            "orange": int(status["orange_score"].sum()),
            "total": int(status["blue_score"].sum() + status["orange_score"].sum()),
        },
        "demolitions": {
            "total": int(demo_count.sum()),
            "per_player_minute": float(demo_count.sum() / player_minutes),
        },
        "router": router,
        "derived": {
            "entry_fraction_per_activation": int(
                router["counters"]["entry_airborne_contacts"]
            )
            / max(int(router["counters"]["activations"]), 1),
            "second_touch_fraction_per_activation": int(
                router["counters"]["second_airborne_contacts"]
            )
            / max(int(router["counters"]["activations"]), 1),
            "goal_fraction_per_activation": int(
                router["counters"]["goals_within_contact_budget"]
            )
            / max(int(router["counters"]["activations"]), 1),
        },
    }
    del runner
    return payload


def run(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    payload = evaluate_checkpoint(
        checkpoint,
        args.checkpoint_sha256,
        worlds=int(args.worlds),
        ticks=int(args.ticks),
        seed=int(args.seed),
        device=args.device,
        collision_root=args.collision_root,
    )
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=1_024)
    parser.add_argument("--ticks", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=2_026_090_301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=SELECTED)
    parser.add_argument("--checkpoint-sha256", default=SELECTED_SHA256)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
