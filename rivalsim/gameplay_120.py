"""Production 120 Hz physical bad-flip guard with no named-mechanics detector.

This module intentionally contains only the two trusted physical exemptions for
the clean 120 Hz gameplay line: a genuine contested challenge and a
dodge-powered power contact.  Gameplay V3's inferred named-mechanics and
controlled-flick machinery remain historical and are never launched here.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from rivalsim.rival2_contracts import GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY

CONTEST_CONTACT_WINDOW_TICKS = 3
CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX = 26.472905158996582
CONTEST_OPPONENT_DISTANCE_MAX = 436.1062316894531
CONTEST_SELF_CLOSING_SPEED_MIN = 727.6748657226562
CONTEST_OPPONENT_CLOSING_SPEED_MIN = 590.4700012207031
CONTEST_TIME_TO_BALL_DELTA_MAX = 0.23664760693567713

POWER_TOTAL_CLOSING_SPEED_MIN = 95.31548309326172
POWER_ROTATIONAL_CLOSING_SPEED_MIN = 164.38546752929688
POWER_ROTATIONAL_SHARE_MIN = 0.2549042037005253
POWER_BALL_DELTA_V_MIN = 303.5580596923828

OUTCOME_NONE = 0
OUTCOME_EXEMPT_CONTESTED_50 = 1
OUTCOME_EXEMPT_POWER_CONTACT = 2
OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT = 3
OUTCOME_COUNT = 4


def physical_flip_outcome(*, contested_50: bool, power_contact: bool) -> int:
    """Pure deterministic precedence used by focused contract tests."""

    if contested_50:
        return OUTCOME_EXEMPT_CONTESTED_50
    if power_contact:
        return OUTCOME_EXEMPT_POWER_CONTACT
    return OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT


@wp.func
def _safe_unit(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length <= 1.0e-6:
        return wp.vec3(0.0)
    return value / length


@wp.func
def _resolve_pending(
    car: int,
    pending_active: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    bad_flip_total: wp.array(dtype=wp.int32),
    contest_exempt_total: wp.array(dtype=wp.int32),
    power_exempt_total: wp.array(dtype=wp.int32),
):
    if pending_contest[car] != 0:
        interval_contest_exempt[car] = interval_contest_exempt[car] + 1
        contest_exempt_total[car] = contest_exempt_total[car] + 1
    elif pending_power[car] != 0:
        interval_power_exempt[car] = interval_power_exempt[car] + 1
        power_exempt_total[car] = power_exempt_total[car] + 1
    else:
        interval_bad_flip[car] = interval_bad_flip[car] + 1
        bad_flip_total[car] = bad_flip_total[car] + 1
    pending_active[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0
    pending_age[car] = 0
    pending_self_contact_tick[car] = -1


@wp.kernel(enable_backward=False)
def gameplay_120_begin_decision(
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    bad_flip_component: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    env = car // 2
    interval_bad_flip[car] = 0
    interval_contest_exempt[car] = 0
    interval_power_exempt[car] = 0
    if car % 2 == 0:
        bad_flip_component[env] = 0.0


@wp.kernel(enable_backward=False)
def gameplay_120_track_tick(
    episode_ticks: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    car_a_hit: wp.array(dtype=wp.int32),
    car_b_hit: wp.array(dtype=wp.int32),
    car_a_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_b_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_a_contact_point_bt: wp.array(dtype=wp.vec3),
    car_b_contact_point_bt: wp.array(dtype=wp.vec3),
    car_a_ball_delta_v: wp.array(dtype=wp.vec3),
    car_b_ball_delta_v: wp.array(dtype=wp.vec3),
    touch_latched: wp.array(dtype=wp.int32),
    opponent_touch_latched: wp.array(dtype=wp.int32),
    recent_opponent_contact_tick: wp.array(dtype=wp.int32),
    recent_opponent_contact_ball_position: wp.array(dtype=wp.vec3),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    pending_ball_position: wp.array(dtype=wp.vec3),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    legitimate_touch_total: wp.array(dtype=wp.int32),
    flip_touch_total: wp.array(dtype=wp.int32),
    bad_flip_total: wp.array(dtype=wp.int32),
    contest_exempt_total: wp.array(dtype=wp.int32),
    power_exempt_total: wp.array(dtype=wp.int32),
):
    car = wp.tid()
    env = car // 2
    local = car % 2
    other = env * 2 + (1 - local)
    tick = episode_ticks[env] + 1
    reports = car_a_hit[env] if local == 0 else car_b_hit[env]
    other_reports = car_b_hit[env] if local == 0 else car_a_hit[env]
    touch_onset = reports != 0 and touch_latched[car] == 0
    opponent_touch_onset = other_reports != 0 and opponent_touch_latched[car] == 0
    touch_latched[car] = wp.int32(reports != 0)
    opponent_touch_latched[car] = wp.int32(other_reports != 0)

    if opponent_touch_onset:
        recent_opponent_contact_tick[car] = tick
        recent_opponent_contact_ball_position[car] = ball_pos[env]

    if pending_active[car] != 0:
        candidate_age = tick - pending_self_contact_tick[car]
        if (
            opponent_touch_onset
            and candidate_age <= CONTEST_CONTACT_WINDOW_TICKS
            and wp.length(ball_pos[env] - pending_ball_position[car])
            <= CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX
        ):
            pending_contest[car] = 1
        pending_age[car] = pending_age[car] + 1
        if pending_age[car] >= CONTEST_CONTACT_WINDOW_TICKS or touch_onset:
            _resolve_pending(
                car,
                pending_active,
                pending_contest,
                pending_power,
                pending_age,
                pending_self_contact_tick,
                interval_bad_flip,
                interval_contest_exempt,
                interval_power_exempt,
                bad_flip_total,
                contest_exempt_total,
                power_exempt_total,
            )

    if touch_onset:
        legitimate_touch_total[car] = legitimate_touch_total[car] + 1
        directional_dodge = (
            is_flipping[car] != 0
            and has_flipped[car] != 0
            and wp.length(flip_rel_torque[car]) > 0.25
        )
        if directional_dodge:
            flip_touch_total[car] = flip_touch_total[car] + 1
            pre_velocity_bt = (
                car_a_pre_velocity_bt[env] if local == 0 else car_b_pre_velocity_bt[env]
            )
            opponent_pre_velocity_bt = (
                car_b_pre_velocity_bt[env] if local == 0 else car_a_pre_velocity_bt[env]
            )
            pre_ball_velocity_bt = (
                car_a_pre_ball_velocity_bt[env]
                if local == 0
                else car_b_pre_ball_velocity_bt[env]
            )
            pre_angular = (
                car_a_pre_angular_velocity[env]
                if local == 0
                else car_b_pre_angular_velocity[env]
            )
            point_bt = (
                car_a_contact_point_bt[env]
                if local == 0
                else car_b_contact_point_bt[env]
            )
            ball_delta_vector = (
                car_a_ball_delta_v[env] if local == 0 else car_b_ball_delta_v[env]
            )
            pre_velocity = pre_velocity_bt * 50.0
            opponent_pre_velocity = opponent_pre_velocity_bt * 50.0
            pre_ball_velocity = pre_ball_velocity_bt * 50.0
            contact_world = point_bt * 50.0
            offset = contact_world - car_pos[car]
            direction = _safe_unit(ball_pos[env] - car_pos[car])
            rotational = wp.max(
                wp.dot(wp.cross(pre_angular, offset), direction), 0.0
            )
            translational = wp.max(
                wp.dot(pre_velocity - pre_ball_velocity, direction), 0.0
            )
            total_closing = rotational + translational
            rotational_share = rotational / wp.max(total_closing, 1.0e-6)
            ball_delta = wp.length(ball_delta_vector)

            opponent_direction = _safe_unit(ball_pos[env] - car_pos[other])
            self_closing = wp.dot(pre_velocity - pre_ball_velocity, direction)
            opponent_closing = wp.dot(
                opponent_pre_velocity - pre_ball_velocity, opponent_direction
            )
            self_distance = wp.length(ball_pos[env] - car_pos[car])
            opponent_distance = wp.length(ball_pos[env] - car_pos[other])
            self_time = self_distance / wp.max(self_closing, 1.0e-6)
            opponent_time = opponent_distance / wp.max(opponent_closing, 1.0e-6)
            convergence = (
                opponent_distance <= CONTEST_OPPONENT_DISTANCE_MAX
                and self_closing >= CONTEST_SELF_CLOSING_SPEED_MIN
                and opponent_closing >= CONTEST_OPPONENT_CLOSING_SPEED_MIN
                and wp.abs(self_time - opponent_time) <= CONTEST_TIME_TO_BALL_DELTA_MAX
            )
            power = (
                total_closing >= POWER_TOTAL_CLOSING_SPEED_MIN
                and rotational >= POWER_ROTATIONAL_CLOSING_SPEED_MIN
                and rotational_share >= POWER_ROTATIONAL_SHARE_MIN
                and ball_delta >= POWER_BALL_DELTA_V_MIN
            )
            recent_tick = recent_opponent_contact_tick[car]
            recent_age = tick - recent_tick
            adjacent_recent = (
                recent_tick >= 0
                and recent_age >= 0
                and recent_age <= CONTEST_CONTACT_WINDOW_TICKS
                and wp.length(
                    ball_pos[env] - recent_opponent_contact_ball_position[car]
                )
                <= CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX
            )
            pending_active[car] = 1
            pending_age[car] = 0
            pending_contest[car] = wp.int32(adjacent_recent or convergence)
            pending_power[car] = wp.int32(power)
            pending_self_contact_tick[car] = tick
            pending_ball_position[car] = ball_pos[env]


@wp.kernel(enable_backward=False)
def gameplay_120_compose_reward(
    interval_tick: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    bad_flip_component: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if interval_tick[env] != 1:
        return
    car_base = env * 2
    component = GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY * float(
        interval_bad_flip[car_base] - interval_bad_flip[car_base + 1]
    )
    blue = reward[car_base] + component
    reward[car_base] = blue
    reward[car_base + 1] = -blue
    bad_flip_component[env] = component


@wp.kernel(enable_backward=False)
def gameplay_120_reset(
    reset_mask: wp.array(dtype=wp.int32),
    touch_latched: wp.array(dtype=wp.int32),
    opponent_touch_latched: wp.array(dtype=wp.int32),
    recent_opponent_contact_tick: wp.array(dtype=wp.int32),
    recent_opponent_contact_ball_position: wp.array(dtype=wp.vec3),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    pending_ball_position: wp.array(dtype=wp.vec3),
):
    car = wp.tid()
    env = car // 2
    if reset_mask[env] == 0:
        return
    touch_latched[car] = 0
    opponent_touch_latched[car] = 0
    recent_opponent_contact_tick[car] = -1
    recent_opponent_contact_ball_position[car] = wp.vec3(0.0)
    pending_active[car] = 0
    pending_age[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0
    pending_self_contact_tick[car] = -1
    pending_ball_position[car] = wp.vec3(0.0)


class Rival2Gameplay120State:
    """Bounded device state for only the production physical guardrail."""

    def __init__(self, world: Any):
        self.world = world
        self.device = world.device
        self.num_worlds = world.num_envs
        self.car_count = self.num_worlds * 2
        self._inventory: dict[str, int] = {}

        def ints(name: str, count: int, *, fill: int = 0) -> None:
            value = (
                wp.zeros(count, dtype=wp.int32, device=self.device)
                if fill == 0
                else wp.full(count, fill, dtype=wp.int32, device=self.device)
            )
            setattr(self, name, value)
            self._inventory[name] = count * 4

        def floats(name: str, count: int) -> None:
            setattr(self, name, wp.zeros(count, dtype=wp.float32, device=self.device))
            self._inventory[name] = count * 4

        def vectors(name: str, count: int) -> None:
            setattr(self, name, wp.zeros(count, dtype=wp.vec3, device=self.device))
            self._inventory[name] = count * 12

        ints("touch_latched", self.car_count)
        ints("opponent_touch_latched", self.car_count)
        ints("recent_opponent_contact_tick", self.car_count, fill=-1)
        vectors("recent_opponent_contact_ball_position", self.car_count)
        ints("pending_active", self.car_count)
        ints("pending_age", self.car_count)
        ints("pending_contest", self.car_count)
        ints("pending_power", self.car_count)
        ints("pending_self_contact_tick", self.car_count, fill=-1)
        vectors("pending_ball_position", self.car_count)
        ints("interval_bad_flip", self.car_count)
        ints("interval_contest_exempt", self.car_count)
        ints("interval_power_exempt", self.car_count)
        ints("legitimate_touch_total", self.car_count)
        ints("flip_touch_total", self.car_count)
        ints("bad_flip_total", self.car_count)
        ints("contest_exempt_total", self.car_count)
        ints("power_exempt_total", self.car_count)
        floats("bad_flip_component", self.num_worlds)
        self.reset(wp.ones(self.num_worlds, dtype=wp.int32, device=self.device))

    @property
    def logical_bytes(self) -> int:
        return sum(self._inventory.values())

    def memory_inventory(self) -> dict[str, Any]:
        return {
            "logical_bytes": self.logical_bytes,
            "named_mechanics_arrays": 0,
            "controlled_flick_arrays": 0,
            "arrays": dict(self._inventory),
        }

    def begin_decision(self) -> None:
        wp.launch(
            gameplay_120_begin_decision,
            dim=self.car_count,
            inputs=[
                self.interval_bad_flip,
                self.interval_contest_exempt,
                self.interval_power_exempt,
                self.bad_flip_component,
            ],
            device=self.device,
        )

    def launch_tick(self) -> None:
        world = self.world
        wp.launch(
            gameplay_120_track_tick,
            dim=self.car_count,
            inputs=[
                world.rival2.episode_ticks,
                world.state.car_pos,
                world.state.car_vel,
                world.state.has_flipped,
                world.state.is_flipping,
                world.state.flip_rel_torque,
                world.state.ball_pos,
                world.state.ball_vel,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_car_velocity_bt,
                world.car_ball_b.pre_car_velocity_bt,
                world.car_ball.pre_ball_velocity_bt,
                world.car_ball_b.pre_ball_velocity_bt,
                world.car_ball.pre_car_angular_velocity,
                world.car_ball_b.pre_car_angular_velocity,
                world.car_ball.contact_point_a_bt,
                world.car_ball_b.contact_point_a_bt,
                world.car_ball.extra_hit_velocity_uu,
                world.car_ball_b.extra_hit_velocity_uu,
                self.touch_latched,
                self.opponent_touch_latched,
                self.recent_opponent_contact_tick,
                self.recent_opponent_contact_ball_position,
                self.pending_active,
                self.pending_age,
                self.pending_contest,
                self.pending_power,
                self.pending_self_contact_tick,
                self.pending_ball_position,
                self.interval_bad_flip,
                self.interval_contest_exempt,
                self.interval_power_exempt,
                self.legitimate_touch_total,
                self.flip_touch_total,
                self.bad_flip_total,
                self.contest_exempt_total,
                self.power_exempt_total,
            ],
            device=self.device,
        )

    def compose_reward(self) -> None:
        wp.launch(
            gameplay_120_compose_reward,
            dim=self.num_worlds,
            inputs=[
                self.world.rival2.interval_tick,
                self.world.rival2.reward,
                self.interval_bad_flip,
                self.bad_flip_component,
            ],
            device=self.device,
        )

    def reset(self, mask: wp.array) -> None:
        wp.launch(
            gameplay_120_reset,
            dim=self.car_count,
            inputs=[
                mask,
                self.touch_latched,
                self.opponent_touch_latched,
                self.recent_opponent_contact_tick,
                self.recent_opponent_contact_ball_position,
                self.pending_active,
                self.pending_age,
                self.pending_contest,
                self.pending_power,
                self.pending_self_contact_tick,
                self.pending_ball_position,
            ],
            device=self.device,
        )


__all__ = [
    "OUTCOME_EXEMPT_CONTESTED_50",
    "OUTCOME_EXEMPT_POWER_CONTACT",
    "OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT",
    "Rival2Gameplay120State",
    "physical_flip_outcome",
]
