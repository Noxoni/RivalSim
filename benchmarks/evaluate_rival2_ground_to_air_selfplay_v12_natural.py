"""Deterministic natural V23 self-play evaluation of compatible aerial options."""

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
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ANGULAR_SPEED_SCALE,
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    ROUTE_NAMES,
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

AERIAL_OPTION_CHECKPOINT_FORMATS = frozenset(
    {
        "RIVAL2_CHECKPOINT_V1",
        "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_CHECKPOINT",
        "RIVAL2_GROUND_TO_AIR_SELF_IMITATION_V13_CHECKPOINT",
        "RIVAL2_GROUND_TO_AIR_MIXED_SELFPLAY_V14_CHECKPOINT",
        "RIVAL2_GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_CHECKPOINT",
    }
)

HANDOFF_FEATURE_NAMES = (
    "ball_y_uu",
    "ball_height_uu",
    "ball_goalward_speed_uu_per_second",
    "ball_vertical_speed_uu_per_second",
    "self_y_uu",
    "self_height_uu",
    "self_goalward_speed_uu_per_second",
    "self_vertical_speed_uu_per_second",
    "relative_ball_x_uu",
    "relative_ball_y_uu",
    "relative_ball_z_uu",
    "relative_ball_goalward_speed_uu_per_second",
    "relative_ball_vertical_speed_uu_per_second",
    "planar_distance_uu",
    "distance_3d_uu",
    "opponent_ball_planar_distance_uu",
    "forward_alignment",
    "self_forward_z",
    "self_up_z",
    "self_angular_speed_radians_per_second",
    "boost_fraction",
    "on_ground",
    "jump_available",
    "dodge_available",
    "action_throttle",
    "action_steer",
    "action_pitch",
    "action_yaw",
    "action_roll",
    "action_jump",
    "action_boost",
    "action_handbrake",
)


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack(
        [observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1
    )


def handoff_features(
    observation: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Return physical, policy-visible handoff features for read-only audit."""

    if observation.ndim != 2 or observation.shape[1] != 182:
        raise ValueError("handoff observation must be [N,182]")
    if action.shape != (observation.shape[0], 8):
        raise ValueError("handoff action must be [N,8]")
    position_scale = torch.as_tensor(
        POSITION_SCALE, dtype=observation.dtype, device=observation.device
    )
    ball_position = _vector(observation, "ball.position") * position_scale
    self_position = _vector(observation, "self.position") * position_scale
    relative_position = _vector(observation, "relative.ball_position") * position_scale
    ball_velocity = (
        _vector(observation, "ball.linear_velocity") * BALL_LINEAR_SPEED_SCALE
    )
    self_velocity = (
        _vector(observation, "self.linear_velocity") * CAR_LINEAR_SPEED_SCALE
    )
    relative_velocity = (
        _vector(observation, "relative.ball_velocity") * BALL_LINEAR_SPEED_SCALE
    )
    opponent_position = _vector(observation, "opponent.position") * position_scale
    forward = _vector(observation, "self.forward")
    up = _vector(observation, "self.up")
    angular = (
        _vector(observation, "self.angular_velocity") * ANGULAR_SPEED_SCALE
    )
    planar = torch.linalg.vector_norm(relative_position[:, :2], dim=-1)
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    opponent_ball_planar = torch.linalg.vector_norm(
        (ball_position - opponent_position)[:, :2], dim=-1
    )
    planar_direction = relative_position[:, :2] / planar[:, None].clamp_min(1.0e-6)
    forward_planar = forward[:, :2]
    forward_planar = forward_planar / torch.linalg.vector_norm(
        forward_planar, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = (forward_planar * planar_direction).sum(dim=-1)
    values = (
        ball_position[:, 1],
        ball_position[:, 2],
        ball_velocity[:, 1],
        ball_velocity[:, 2],
        self_position[:, 1],
        self_position[:, 2],
        self_velocity[:, 1],
        self_velocity[:, 2],
        relative_position[:, 0],
        relative_position[:, 1],
        relative_position[:, 2],
        relative_velocity[:, 1],
        relative_velocity[:, 2],
        planar,
        distance,
        opponent_ball_planar,
        alignment,
        forward[:, 2],
        up[:, 2],
        torch.linalg.vector_norm(angular, dim=-1),
        observation[:, FIELD["self.boost"]],
        observation[:, FIELD["self.on_ground"]],
        observation[:, FIELD["self.jump_available"]],
        observation[:, FIELD["self.dodge_available"]],
        *(action[:, channel] for channel in range(8)),
    )
    result = torch.stack(values, dim=-1)
    if result.shape[1] != len(HANDOFF_FEATURE_NAMES):
        raise RuntimeError("handoff feature contract mismatch")
    return result


class RouteFeatureMoments:
    """Device-resident per-route feature moments without retaining frames."""

    def __init__(self, *, device: torch.device) -> None:
        routes = len(ROUTE_NAMES)
        features = len(HANDOFF_FEATURE_NAMES)
        self.device = device
        self.count = torch.zeros(routes, dtype=torch.int64, device=device)
        self.total = torch.zeros((routes, features), dtype=torch.float64, device=device)
        self.total_square = torch.zeros_like(self.total)
        self._route_ids = torch.arange(routes, dtype=torch.int64, device=device)

    def add(
        self,
        values: torch.Tensor,
        *,
        mask: torch.Tensor,
        route: torch.Tensor,
    ) -> None:
        if values.ndim != 2 or values.shape[1] != len(HANDOFF_FEATURE_NAMES):
            raise ValueError("route feature matrix mismatch")
        if mask.shape != route.shape or mask.shape != (values.shape[0],):
            raise ValueError("route feature mask mismatch")
        membership = (
            mask[:, None]
            & (route[:, None].to(torch.int64) == self._route_ids[None, :])
        ).to(torch.float64)
        numeric = values.to(torch.float64)
        self.count += membership.sum(dim=0).to(torch.int64)
        self.total += membership.transpose(0, 1) @ numeric
        self.total_square += membership.transpose(0, 1) @ numeric.square()

    def export(self) -> dict[str, Any]:
        count = self.count.detach().cpu()
        total = self.total.detach().cpu()
        total_square = self.total_square.detach().cpu()
        result: dict[str, Any] = {}
        for route_id, route_name in enumerate(ROUTE_NAMES):
            n = int(count[route_id])
            if n == 0:
                result[route_name] = {"count": 0, "features": {}}
                continue
            mean = total[route_id] / n
            variance = (total_square[route_id] / n - mean.square()).clamp_min(0.0)
            result[route_name] = {
                "count": n,
                "features": {
                    name: {
                        "mean": float(mean[index]),
                        "standard_deviation": float(variance[index].sqrt()),
                    }
                    for index, name in enumerate(HANDOFF_FEATURE_NAMES)
                },
            }
        return result


class V12NaturalSelfPlayRunner(SideSpecializedSelfPlayRunner):
    """Both V23 sides share the selected direct aerial option and router."""

    def __init__(
        self,
        *args: Any,
        option_checkpoint: Path = SELECTED,
        option_sha256: str = SELECTED_SHA256,
        option_router_config: AerialOptionRouterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        option_checkpoint = Path(option_checkpoint).resolve()
        observed_sha256 = sha256_file(option_checkpoint)
        if observed_sha256 != str(option_sha256).upper():
            raise RuntimeError("natural self-play option identity mismatch")
        payload = torch.load(option_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")
        if checkpoint_format not in AERIAL_OPTION_CHECKPOINT_FORMATS:
            raise RuntimeError("unexpected aerial-option checkpoint format")
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != self.checkpoint_identity["policy_config"]:
            raise RuntimeError("selected option architecture differs from V23")
        self.option_policy = Rival2ActorCritic(config).to(self.device)
        self.option_policy.load_state_dict(payload["model"], strict=True)
        self.option_policy.eval().requires_grad_(False)
        # The protected V3 scorer is a standard production checkpoint.  It has
        # no router metadata because routing was added later; use the frozen V12
        # defaults only for this read-only natural evaluation.  The policy
        # state itself is loaded byte-for-byte from the explicitly hashed input.
        router_config = option_router_config
        if router_config is None:
            router_config = (
                AerialOptionRouterConfig()
                if checkpoint_format == "RIVAL2_CHECKPOINT_V1"
                else AerialOptionRouterConfig(**payload["router_config"])
            )
        reward_config = (
            AerialSelfPlayRewardConfig()
            if checkpoint_format == "RIVAL2_CHECKPOINT_V1"
            else AerialSelfPlayRewardConfig(**payload["aerial_reward_config"])
        )
        self.aerial_router = AerialOptionSelfPlayRouter(
            self.num_worlds * 2,
            device=self.device,
            router_config=router_config,
            reward_config=reward_config,
        )
        self._goal_scored = wp.to_torch(self.world.lifecycle.goal_scored)
        self._scoring_team = wp.to_torch(self.world.lifecycle.scoring_team).to(torch.int64)
        self._before = torch.zeros_like(self.rival_observation)
        self._active = torch.zeros(
            (self.num_worlds, 2), dtype=torch.bool, device=self.device
        )
        self._route_before = torch.full(
            (self.num_worlds, 2),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._route_goals_for = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self._route_goals_against = torch.zeros_like(self._route_goals_for)
        self._activation_features = RouteFeatureMoments(device=self.device)
        self._entry_features = RouteFeatureMoments(device=self.device)
        self._second_features = RouteFeatureMoments(device=self.device)

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
            self._activation_features.add(
                handoff_features(flat, option_actions.reshape(-1, 8)),
                mask=selection.activated,
                route=selection.route,
            )
            active = selection.active.reshape(self.num_worlds, 2)
            self.actions.copy_(torch.where(active[..., None], option_actions, base_actions))
            self._before.copy_(observation)
            self._active.copy_(active)
            self._route_before.copy_(selection.route.reshape(self.num_worlds, 2))

    def tick(self) -> None:
        self._activate_stream()
        self._update_all_actions()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        after = self.bridge.observation()
        side = torch.arange(2, dtype=torch.int64, device=self.device)[None, :]
        goal_event = self._goal_scored[:, None] != 0
        goal_for = goal_event & (
            self._scoring_team[:, None] == side
        )
        goal_against = goal_event & (self._scoring_team[:, None] != side)
        self._route_goals_for += (goal_for & self._active).sum()
        self._route_goals_against += (goal_against & self._active).sum()
        outcome = self.aerial_router.observe(
            self._before.reshape(-1, 182),
            after.reshape(-1, 182),
            active_before=self._active.reshape(-1),
            goal_for_lane=goal_for.reshape(-1),
        )
        after_features = handoff_features(after.reshape(-1, 182), self.actions.reshape(-1, 8))
        route_before = self._route_before.reshape(-1)
        self._entry_features.add(
            after_features,
            mask=outcome.entry_airborne_contact,
            route=route_before,
        )
        self._second_features.add(
            after_features,
            mask=outcome.second_airborne_contact,
            route=route_before,
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
    router_config: AerialOptionRouterConfig | None = None,
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
        option_router_config=router_config,
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
        "router_config": asdict(router_config or runner.aerial_router.config),
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
        "route_goal_outcomes": {
            "goals_for_while_active": int(runner._route_goals_for.item()),
            "goals_against_while_active": int(runner._route_goals_against.item()),
        },
        "handoff_features": {
            "feature_names": list(HANDOFF_FEATURE_NAMES),
            "activation": runner._activation_features.export(),
            "entry_airborne_contact": runner._entry_features.export(),
            "second_airborne_contact": runner._second_features.export(),
        },
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
    router_config = None
    if args.router_config_json is not None:
        router_payload = json.loads(args.router_config_json.read_text(encoding="utf-8"))
        router_config = AerialOptionRouterConfig(**router_payload)
    payload = evaluate_checkpoint(
        checkpoint,
        args.checkpoint_sha256,
        worlds=int(args.worlds),
        ticks=int(args.ticks),
        seed=int(args.seed),
        device=args.device,
        collision_root=args.collision_root,
        router_config=router_config,
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
    parser.add_argument("--router-config-json", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
