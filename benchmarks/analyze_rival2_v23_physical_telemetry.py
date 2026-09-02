"""Read-only physical-behavior telemetry for the frozen Rival V23 bundle.

This diagnostic does not define rewards or production mechanic detectors.  It
replays the already-selected deterministic V23 policies against pinned Nexto
and reports literal state/action sequences: height, wheel state, speed,
touches, flips, landings, and scoring context.  Wavedash-like results are
explicitly labelled as physics proxies rather than mechanic adjudications.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_codex_autonomous_v1 import sha256_file  # noqa: E402
from benchmarks.run_rival2_codex_autonomous_v23 import (  # noqa: E402
    BLUE_SHA256,
    ORANGE_SHA256,
    SideSpecializedFullMatchRunner,
)
from rivalsim.full_match import (  # noqa: E402
    FullMatchRunner,
    NEXTO_CADENCE_TICKS,
    REGULATION_TICKS,
)

BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
DEFAULT_OUTPUT = (
    ROOT / "results/rival2/codex_autonomous_v23/physical_behavior_telemetry.json"
)
PHYSICS_HZ = 120
SUPERSONIC_UU_PER_SECOND = 2200.0


class _PhysicalTelemetryMixin:
    """Bounded post-physics trace capture shared by uniform and V23 runners."""

    def __init__(self, *args: Any, trace_ticks: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_ticks = int(trace_ticks)
        self.trace_index = 0
        shape = (self.trace_ticks, self.num_worlds)
        float_names = (
            "car_x",
            "car_y",
            "car_z",
            "opponent_x",
            "opponent_y",
            "ball_x",
            "ball_y",
            "ball_z",
            "horizontal_speed",
            "speed_3d",
            "vertical_speed",
            "angular_speed",
            "air_time",
            "up_z",
            "ball_speed_after",
            "ball_velocity_y_before",
            "ball_velocity_y_after",
            "world_contact_normal_x",
            "world_contact_normal_y",
            "world_contact_normal_z",
        )
        int_names = (
            "on_ground",
            "wheel_contacts",
            "has_flipped",
            "is_flipping",
            "is_supersonic",
            "rival_hit_raw",
            "nexto_hit_raw",
            "goal_scored",
            "scoring_team",
            "pre_tick_first_car",
            "rival_demo_count",
        )
        self.trace: dict[str, torch.Tensor] = {
            name: torch.empty(shape, dtype=torch.float32, device=self.device)
            for name in float_names
        }
        self.trace.update(
            {
                name: torch.empty(shape, dtype=torch.int16, device=self.device)
                for name in int_names
            }
        )
        self.trace["action"] = torch.empty(
            (*shape, 8), dtype=torch.float32, device=self.device
        )
        self._goal_scored = wp.to_torch(self.world.lifecycle.goal_scored)
        self._scoring_team = wp.to_torch(self.world.lifecycle.scoring_team)
        self._hit_a = wp.to_torch(self.world.car_ball.hit_this_tick)
        self._hit_b = wp.to_torch(self.world.car_ball_b.hit_this_tick)
        self._pre_tick_first_car = wp.to_torch(
            self.world.car_car.pre_tick_first_car
        )
        self._world_contact_normal = wp.to_torch(
            self.world.vehicle.world_contact_normal
        )
        self._cumulative_demo_count = wp.to_torch(
            self.telemetry.demo_count
        ).reshape(self.num_worlds, 2)

    def _rival_car(self, name: str) -> torch.Tensor:
        value = self.bridge.views[name]
        trailing = value.shape[1:]
        reshaped = value.reshape(self.num_worlds, 2, *trailing)
        return reshaped[self.batch_index, self.rival_side]

    def _opponent_car(self, name: str) -> torch.Tensor:
        value = self.bridge.views[name]
        trailing = value.shape[1:]
        reshaped = value.reshape(self.num_worlds, 2, *trailing)
        return reshaped[self.batch_index, self.nexto_side]

    def _record_post_physics(
        self, ball_velocity_before: torch.Tensor
    ) -> None:
        index = self.trace_index
        if index >= self.trace_ticks:
            raise RuntimeError("physical telemetry trace capacity exceeded")
        car_pos = self._rival_car("car_pos")
        car_vel = self._rival_car("car_vel")
        car_ang = self._rival_car("car_ang_vel")
        car_quat = self._rival_car("car_quat")
        opponent_pos = self._opponent_car("car_pos")
        ball_pos = self.bridge.views["ball_pos"]
        ball_vel = self.bridge.views["ball_vel"]
        x, y, _z, _w = car_quat.unbind(-1)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        hit_rival = torch.where(self.rival_side == 0, self._hit_a, self._hit_b)
        hit_nexto = torch.where(self.rival_side == 0, self._hit_b, self._hit_a)

        self.trace["car_x"][index].copy_(car_pos[:, 0])
        self.trace["car_y"][index].copy_(car_pos[:, 1])
        self.trace["car_z"][index].copy_(car_pos[:, 2])
        self.trace["opponent_x"][index].copy_(opponent_pos[:, 0])
        self.trace["opponent_y"][index].copy_(opponent_pos[:, 1])
        self.trace["ball_x"][index].copy_(ball_pos[:, 0])
        self.trace["ball_y"][index].copy_(ball_pos[:, 1])
        self.trace["ball_z"][index].copy_(ball_pos[:, 2])
        self.trace["horizontal_speed"][index].copy_(
            torch.linalg.vector_norm(car_vel[:, :2], dim=1)
        )
        self.trace["speed_3d"][index].copy_(
            torch.linalg.vector_norm(car_vel, dim=1)
        )
        self.trace["vertical_speed"][index].copy_(car_vel[:, 2])
        self.trace["angular_speed"][index].copy_(
            torch.linalg.vector_norm(car_ang, dim=1)
        )
        self.trace["air_time"][index].copy_(
            self._rival_car("air_time").to(torch.float32)
        )
        self.trace["up_z"][index].copy_(up_z)
        self.trace["ball_speed_after"][index].copy_(
            torch.linalg.vector_norm(ball_vel, dim=1)
        )
        self.trace["ball_velocity_y_before"][index].copy_(
            ball_velocity_before[:, 1]
        )
        self.trace["ball_velocity_y_after"][index].copy_(ball_vel[:, 1])
        self.trace["on_ground"][index].copy_(
            self._rival_car("on_ground").to(torch.int16)
        )
        wheel = self.bridge.views["wheel_contact"].reshape(
            self.num_worlds, 2, 4
        )[self.batch_index, self.rival_side]
        self.trace["wheel_contacts"][index].copy_(
            (wheel != 0).sum(dim=1).to(torch.int16)
        )
        for name in ("has_flipped", "is_flipping", "is_supersonic"):
            self.trace[name][index].copy_(
                self._rival_car(name).to(torch.int16)
            )
        self.trace["rival_hit_raw"][index].copy_(hit_rival.to(torch.int16))
        self.trace["nexto_hit_raw"][index].copy_(hit_nexto.to(torch.int16))
        self.trace["goal_scored"][index].copy_(
            self._goal_scored.to(torch.int16)
        )
        self.trace["scoring_team"][index].copy_(
            self._scoring_team.to(torch.int16)
        )
        self.trace["pre_tick_first_car"][index].copy_(
            self._pre_tick_first_car.to(torch.int16)
        )
        self.trace["rival_demo_count"][index].copy_(
            self._cumulative_demo_count[
                self.batch_index, self.rival_side
            ].to(torch.int16)
        )
        contact_normal = self._world_contact_normal.reshape(
            self.num_worlds, 2, 3
        )[self.batch_index, self.rival_side]
        for axis, name in enumerate(
            ("world_contact_normal_x", "world_contact_normal_y", "world_contact_normal_z")
        ):
            self.trace[name][index].copy_(contact_normal[:, axis])
        self.trace["action"][index].copy_(self.rival_action)
        self.trace_index += 1

    def tick(self) -> None:
        """Exact base scheduler with capture between physics and reset."""

        self._activate_stream()
        if self.host_tick % self.rival_cadence_ticks == 0:
            self._update_rival_action()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        kickoff_active = self.match_views["kickoff_active"] != 0
        nexto_action, _indices = self.nexto.tick_action(
            self.nexto_state, kickoff_active
        )
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        ball_velocity_before = self.bridge.views["ball_vel"].clone()
        self.world.step_graph(1)
        self._record_post_physics(ball_velocity_before)

        self.match_views["rival_scheduler_tick"].add_(1).remainder_(
            self.rival_cadence_ticks
        )
        self.match_views["nexto_scheduler_tick"].add_(1).remainder_(
            NEXTO_CADENCE_TICKS
        )
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            wp.copy(self.world.rival2.reset_mask, self.match.pending_reset)
            reset_mask = self.match_views["pending_reset"] != 0
            self.nexto.notify_kickoff(reset_mask)
            self.world.apply_interval_resets()
            self.telemetry.after_resets(self.world, self.world.rival2.reset_mask)
        if self.host_tick % self.rival_cadence_ticks == 0:
            self.rival_observation = self.bridge.observation()

    def trace_numpy(self) -> dict[str, np.ndarray]:
        torch.cuda.synchronize(self.device)
        return {
            name: value[: self.trace_index].detach().cpu().numpy().copy()
            for name, value in self.trace.items()
        }


class PhysicalTelemetryRunner(_PhysicalTelemetryMixin, SideSpecializedFullMatchRunner):
    """V23 side-specialized runner retained for the historical diagnostic."""

    orange_checkpoint = ORANGE


class UniformPhysicalTelemetryRunner(_PhysicalTelemetryMixin, FullMatchRunner):
    """One-checkpoint runner for prospective capability candidates."""


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _rising(value: np.ndarray) -> np.ndarray:
    previous = np.zeros_like(value, dtype=bool)
    previous[1:] = value[:-1]
    return value & ~previous


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _analyze_subset(
    trace: dict[str, np.ndarray], world_indices: np.ndarray, rival_side: np.ndarray
) -> dict[str, Any]:
    selected = np.asarray(world_indices, dtype=np.int64)
    ground = trace["on_ground"][:, selected] != 0
    airborne = ~ground
    speed = trace["horizontal_speed"][:, selected]
    car_z = trace["car_z"][:, selected]
    ball_z = trace["ball_z"][:, selected]
    action = trace["action"][:, selected]
    raw_rival_hit = trace["rival_hit_raw"][:, selected] != 0
    raw_nexto_hit = trace["nexto_hit_raw"][:, selected] != 0
    rival_touch = _rising(raw_rival_hit)
    nexto_touch = _rising(raw_nexto_hit)
    jump_rising = _rising(action[..., 5] >= 0.5)
    flip_rising = _rising(trace["has_flipped"][:, selected] != 0)
    goal = trace["goal_scored"][:, selected] != 0

    ground_speed = speed[ground]
    air_speed = speed[airborne]
    ground_touch = rival_touch & ground
    airborne_touch = rival_touch & airborne
    elevated_touch = airborne_touch & (car_z >= 100.0) & (ball_z >= 150.0)
    high_touch = airborne_touch & (car_z >= 300.0) & (ball_z >= 300.0)
    sign = np.where(rival_side[selected] == 0, 1.0, -1.0)[None, :]
    longitudinal_delta = sign * (
        trace["ball_velocity_y_after"][:, selected]
        - trace["ball_velocity_y_before"][:, selected]
    )

    episode_max_heights: list[float] = []
    episode_durations: list[int] = []
    low_air_episodes = 0
    high_air_episodes = 0
    ceiling_air_episodes = 0
    multi_elevated_touch_episodes = 0
    landing_count = 0
    flip_active_landings = 0
    short_flip_landings = 0
    speed_gain_flip_landings = 0
    productive_floor_landings = 0
    productive_wall_landings = 0
    productive_landing_events: list[tuple[int, int]] = []
    landing_speed_ratios: list[float] = []
    for local_world in range(selected.size):
        for start, end in _episodes(airborne[:, local_world]):
            if end <= start:
                continue
            maximum = float(car_z[start:end, local_world].max())
            duration = end - start
            episode_max_heights.append(maximum)
            episode_durations.append(duration)
            low_air_episodes += int(maximum >= 100.0)
            high_air_episodes += int(maximum >= 300.0)
            ceiling_air_episodes += int(maximum >= 1000.0)
            multi_elevated_touch_episodes += int(
                elevated_touch[start:end, local_world].sum() >= 2
            )
            if end < ground.shape[0]:
                landing_count += 1
                recent_start = max(start, end - 12)
                recent_flip = bool(
                    np.any(trace["has_flipped"][recent_start:end, selected[local_world]] != 0)
                    or np.any(trace["is_flipping"][recent_start:end, selected[local_world]] != 0)
                )
                flip_active_landings += int(recent_flip)
                short_flip_landings += int(recent_flip and duration <= 90)
                before_index = max(start, end - 6)
                before_speed = float(speed[before_index, local_world])
                after_speed = float(speed[end, local_world])
                if before_speed > 1.0:
                    landing_speed_ratios.append(after_speed / before_speed)
                speed_gain_flip_landings += int(
                    recent_flip and after_speed - before_speed >= 100.0
                )
                normal_z = float(
                    trace["world_contact_normal_z"][end, selected[local_world]]
                )
                productive = bool(
                    recent_flip
                    and duration <= 90
                    and after_speed - before_speed >= 100.0
                )
                floor = normal_z >= 0.70
                wall = abs(normal_z) <= 0.30
                productive_floor_landings += int(productive and floor)
                productive_wall_landings += int(productive and wall)
                if productive:
                    productive_landing_events.append((local_world, end))

    productive_dash_chains = 0
    by_world: dict[int, list[int]] = {}
    for local_world, tick in productive_landing_events:
        by_world.setdefault(local_world, []).append(tick)
    for ticks in by_world.values():
        productive_dash_chains += sum(
            int(current - previous <= 90)
            for previous, current in zip(ticks, ticks[1:])
        )

    scoring = trace["scoring_team"][:, selected]
    first_car = trace["pre_tick_first_car"][:, selected]
    scoring_goals = 0
    last_touch_rival_goals = 0
    airborne_last_touch_goals = 0
    elevated_aerial_goals = 0
    high_aerial_goals = 0
    goal_touch_age_ticks: list[int] = []
    goal_last_touch_car_z: list[float] = []
    goal_last_touch_ball_z: list[float] = []
    for local_world, world in enumerate(selected.tolist()):
        side = int(rival_side[world])
        last_touch: dict[str, Any] | None = None
        for tick in range(goal.shape[0]):
            rival_event = bool(rival_touch[tick, local_world])
            nexto_event = bool(nexto_touch[tick, local_world])
            event_order: list[bool] = []
            if rival_event and nexto_event:
                first_is_rival = int(first_car[tick, local_world]) == side
                event_order = [first_is_rival, not first_is_rival]
            elif rival_event:
                event_order = [True]
            elif nexto_event:
                event_order = [False]
            for is_rival in event_order:
                last_touch = {
                    "rival": is_rival,
                    "tick": tick,
                    "airborne": bool(airborne[tick, local_world]),
                    "car_z": float(car_z[tick, local_world]) if is_rival else None,
                    "ball_z": float(ball_z[tick, local_world]),
                }
            if goal[tick, local_world]:
                if int(scoring[tick, local_world]) == side:
                    scoring_goals += 1
                    if last_touch is not None and last_touch["rival"]:
                        last_touch_rival_goals += 1
                        age = tick - int(last_touch["tick"])
                        goal_touch_age_ticks.append(age)
                        goal_last_touch_car_z.append(float(last_touch["car_z"]))
                        goal_last_touch_ball_z.append(float(last_touch["ball_z"]))
                        airborne_last_touch_goals += int(last_touch["airborne"])
                        elevated_aerial_goals += int(
                            last_touch["airborne"]
                            and float(last_touch["car_z"]) >= 100.0
                            and float(last_touch["ball_z"]) >= 150.0
                        )
                        high_aerial_goals += int(
                            last_touch["airborne"]
                            and float(last_touch["car_z"]) >= 300.0
                            and float(last_touch["ball_z"]) >= 300.0
                        )
                last_touch = None

    cumulative_demos = trace["rival_demo_count"][:, selected].astype(np.int64)
    previous_demos = np.zeros_like(cumulative_demos)
    previous_demos[1:] = cumulative_demos[:-1]
    demo_delta = np.maximum(cumulative_demos - previous_demos, 0)
    demo_ticks = np.argwhere(demo_delta > 0)
    demos_total = int(demo_delta.sum())
    demos_opponent_half = 0
    demos_offensive_route = 0
    demos_followed_by_touch = 0
    demos_followed_by_goal = 0
    demo_opponent_distances: list[float] = []
    five_seconds = 5 * PHYSICS_HZ
    for tick, local_world in demo_ticks.tolist():
        world = int(selected[local_world])
        multiplicity = int(demo_delta[tick, local_world])
        side = int(rival_side[world])
        direction = 1.0 if side == 0 else -1.0
        car_y = float(trace["car_y"][tick, world])
        ball_y_value = float(trace["ball_y"][tick, world])
        opponent_dx = float(
            trace["car_x"][tick, world] - trace["opponent_x"][tick, world]
        )
        opponent_dy = float(car_y - trace["opponent_y"][tick, world])
        demo_opponent_distances.extend(
            [float(np.hypot(opponent_dx, opponent_dy))] * multiplicity
        )
        opponent_half = direction * ball_y_value >= 0.0
        route = opponent_half or direction * car_y >= direction * ball_y_value
        demos_opponent_half += multiplicity * int(opponent_half)
        demos_offensive_route += multiplicity * int(route)
        end = min(goal.shape[0], tick + five_seconds + 1)
        demos_followed_by_touch += multiplicity * int(
            bool(rival_touch[tick:end, local_world].any())
        )
        goal_rows = goal[tick:end, local_world]
        scoring_rows = scoring[tick:end, local_world]
        demos_followed_by_goal += multiplicity * int(
            bool(np.any(goal_rows & (scoring_rows == side)))
        )

    total_ticks = int(ground.size)
    match_minutes = total_ticks / PHYSICS_HZ / 60.0
    return {
        "worlds": int(selected.size),
        "match_minutes": match_minutes,
        "air": {
            "airborne_tick_fraction": float(airborne.mean()),
            "maximum_car_height_uu": float(car_z.max()),
            "airborne_horizontal_speed_uu_per_second": _distribution(air_speed),
            "air_episode_count": len(episode_durations),
            "air_episode_duration_seconds": _distribution(
                np.asarray(episode_durations, dtype=np.float64) / PHYSICS_HZ
            ),
            "air_episode_max_height_uu": _distribution(episode_max_heights),
            "episodes_reaching_100uu": low_air_episodes,
            "episodes_reaching_300uu": high_air_episodes,
            "episodes_reaching_1000uu": ceiling_air_episodes,
            "multi_elevated_touch_air_episodes": multi_elevated_touch_episodes,
        },
        "touches": {
            "total": int(rival_touch.sum()),
            "per_minute": float(rival_touch.sum() / match_minutes),
            "grounded": int(ground_touch.sum()),
            "airborne_any_height": int(airborne_touch.sum()),
            "elevated_aerial_proxy": int(elevated_touch.sum()),
            "high_aerial_proxy": int(high_touch.sum()),
            "grounded_fraction": float(ground_touch.sum() / max(1, rival_touch.sum())),
            "grounded_forward_momentum_fraction": float(
                (longitudinal_delta[ground_touch] >= 100.0).mean()
            ) if ground_touch.any() else None,
            "grounded_longitudinal_ball_velocity_delta_uu_per_second": _distribution(
                longitudinal_delta[ground_touch]
            ),
            "grounded_post_touch_ball_speed_uu_per_second": _distribution(
                trace["ball_speed_after"][:, selected][ground_touch]
            ),
        },
        "scoring": {
            "goals": scoring_goals,
            "goals_with_rival_as_last_toucher": last_touch_rival_goals,
            "goals_last_touched_while_airborne_any_height": airborne_last_touch_goals,
            "goals_from_elevated_aerial_proxy": elevated_aerial_goals,
            "goals_from_high_aerial_proxy": high_aerial_goals,
            "last_touch_to_goal_seconds": _distribution(
                np.asarray(goal_touch_age_ticks, dtype=np.float64) / PHYSICS_HZ
            ),
            "last_touch_car_height_uu": _distribution(goal_last_touch_car_z),
            "last_touch_ball_height_uu": _distribution(goal_last_touch_ball_z),
        },
        "ground": {
            "grounded_tick_fraction": float(ground.mean()),
            "horizontal_speed_uu_per_second": _distribution(ground_speed),
            "fraction_above_1400uu_per_second": float(
                (ground_speed >= 1400.0).mean()
            ),
            "fraction_above_2000uu_per_second": float(
                (ground_speed >= 2000.0).mean()
            ),
            "supersonic_state_fraction": float(
                (trace["is_supersonic"][:, selected][ground] != 0).mean()
            ),
            "full_throttle_fraction": float((action[..., 0][ground] >= 0.9).mean()),
            "boost_fraction": float((action[..., 6][ground] >= 0.5).mean()),
            "handbrake_fraction": float((action[..., 7][ground] >= 0.5).mean()),
        },
        "jump_flip_recovery": {
            "jump_onsets": int(jump_rising.sum()),
            "jump_onsets_per_minute": float(jump_rising.sum() / match_minutes),
            "flip_onsets": int(flip_rising.sum()),
            "flip_onsets_per_minute": float(flip_rising.sum() / match_minutes),
            "landings": landing_count,
            "flip_active_landings": flip_active_landings,
            "short_air_flip_active_landings": short_flip_landings,
            "flip_active_landings_with_100uu_speed_gain": speed_gain_flip_landings,
            "productive_floor_landing_proxy": productive_floor_landings,
            "productive_wall_landing_proxy": productive_wall_landings,
            "productive_dash_chain_proxy": productive_dash_chains,
            "landing_speed_retention_ratio": _distribution(landing_speed_ratios),
            "wavedash_interpretation": (
                "Flip-active landing counts are a broad physics proxy. They include "
                "ordinary dodge landings and are not proof of intentional wavedashes."
            ),
        },
        "demolitions": {
            "total": demos_total,
            "per_minute": float(demos_total / match_minutes),
            "opponent_half": demos_opponent_half,
            "offensive_route_context": demos_offensive_route,
            "followed_by_rival_touch_within_5_seconds": demos_followed_by_touch,
            "followed_by_rival_goal_within_5_seconds": demos_followed_by_goal,
            "opponent_distance_uu": _distribution(demo_opponent_distances),
            "interpretation": (
                "Counts are authoritative simulator demolition events. Offensive "
                "context and five-second follow-ups are descriptive telemetry, not rewards."
            ),
        },
    }


def run(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint is None:
        if sha256_file(BLUE) != BLUE_SHA256:
            raise RuntimeError("V23 Blue checkpoint identity changed")
        if sha256_file(ORANGE) != ORANGE_SHA256:
            raise RuntimeError("V23 Orange checkpoint identity changed")
        runner_type = PhysicalTelemetryRunner
        runner_checkpoint = BLUE
        checkpoint_report: dict[str, Any] = {
            "blue": {"path": BLUE.relative_to(ROOT).as_posix(), "sha256": BLUE_SHA256},
            "orange": {
                "path": ORANGE.relative_to(ROOT).as_posix(),
                "sha256": ORANGE_SHA256,
            },
        }
        report_format = "RIVAL2_V23_PHYSICAL_BEHAVIOR_TELEMETRY_V2"
    else:
        expected = str(args.checkpoint_sha256 or "").upper()
        observed = sha256_file(checkpoint)
        if not expected:
            raise RuntimeError("--checkpoint-sha256 is required with --checkpoint")
        if observed != expected:
            raise RuntimeError(
                f"candidate checkpoint SHA-256 mismatch: {observed} != {expected}"
            )
        runner_type = UniformPhysicalTelemetryRunner
        runner_checkpoint = checkpoint
        try:
            relative = checkpoint.relative_to(ROOT).as_posix()
        except ValueError:
            relative = checkpoint.as_posix()
        checkpoint_report = {
            "uniform": {"path": relative, "sha256": observed}
        }
        report_format = "RIVAL2_PHYSICAL_CAPABILITY_TELEMETRY_V1"
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = runner_type(
        10,
        str(Path(args.collision_root).resolve()),
        runner_checkpoint,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=int(args.seed),
        trace_ticks=REGULATION_TICKS,
    )
    print("physical telemetry: running 10 deterministic regulation matches", flush=True)
    timing = runner.run_ticks(REGULATION_TICKS)
    status = runner.phase_status()
    if np.any(status["done"] == 0):
        raise RuntimeError("physical telemetry authority expected no overtime")
    trace = runner.trace_numpy()
    exported = runner.export()
    raw = exported["raw"]
    observed_touches = int(
        raw["touch_count"][np.arange(10), rival_side].sum()
    )
    traced_touch_count = int(_rising(trace["rival_hit_raw"] != 0).sum())
    if observed_touches != traced_touch_count:
        raise RuntimeError(
            f"touch trace mismatch: {traced_touch_count} != {observed_touches}"
        )
    blue_worlds = np.flatnonzero(rival_side == 0)
    orange_worlds = np.flatnonzero(rival_side == 1)
    report = {
        "format": report_format,
        "diagnostic_only": True,
        "policy_mutation": False,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "checkpoint": checkpoint_report,
        "evaluation": {
            "opponent": "pinned Nexto",
            "action_mode": "deterministic",
            "seed": int(args.seed),
            "matrix": "five standard kickoff layouts x both Rival sides",
            "physics_hz": PHYSICS_HZ,
            "physics_ticks": REGULATION_TICKS,
            "seconds": timing.seconds,
            "score": {
                "wins": int((status["winner"] == rival_side).sum()),
                "losses": int((status["winner"] != rival_side).sum()),
                "rival_goals": int(
                    np.where(
                        rival_side == 0, status["blue_score"], status["orange_score"]
                    ).sum()
                ),
                "nexto_goals": int(
                    np.where(
                        rival_side == 0, status["orange_score"], status["blue_score"]
                    ).sum()
                ),
            },
            "touch_trace_matches_authoritative_telemetry": True,
        },
        "thresholds": {
            "elevated_aerial_proxy": "airborne and car_z >= 100 uu and ball_z >= 150 uu",
            "high_aerial_proxy": "airborne and car_z >= 300 uu and ball_z >= 300 uu",
            "wavedash_like_proxy": (
                "airborne-to-ground transition with flip state in preceding 12 ticks; "
                "speed-gain subset requires >=100 uu/s over preceding 6 ticks"
            ),
            "productive_floor_landing_proxy": (
                "the wavedash-like proxy plus first grounded world-contact normal z >= 0.70"
            ),
            "productive_wall_landing_proxy": (
                "the wavedash-like proxy plus absolute first grounded world-contact normal z <= 0.30"
            ),
            "productive_dash_chain_proxy": (
                "two productive landing proxies for one car no more than 90 physics ticks apart"
            ),
            "supersonic_speed_reference_uu_per_second": SUPERSONIC_UU_PER_SECOND,
        },
        "overall": _analyze_subset(trace, np.arange(10), rival_side),
        "rival_as_blue": _analyze_subset(trace, blue_worlds, rival_side),
        "rival_as_orange": _analyze_subset(trace, orange_worlds, rival_side),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--seed", type=int, default=2_026_090_206)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
