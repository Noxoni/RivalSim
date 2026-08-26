"""Device-resident Rival 2.0 environment and Warp/PyTorch tensor bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from rivalsim.constants import DOUBLEJUMP_MAX_DELAY
from rivalsim.kernels.rival2 import (
    PHYSICS_TICKS_PER_DECISION,
    rival2_accumulate_tick,
    rival2_after_interval_reset,
    rival2_begin_decision,
    rival2_interval_reset,
)
from rivalsim.rival2_contracts import (
    AIR_TIME_SCALE,
    ANGULAR_SPEED_SCALE,
    APPROACH_DISTANCE_SCALE,
    BALL_LINEAR_SPEED_SCALE,
    BOOST_SCALE,
    BOOSTING_TIME_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    DEMO_TIMER_SCALE,
    EPISODE_AGE_SCALE_TICKS,
    FLIP_TIME_SCALE,
    JUMP_TIME_SCALE,
    NO_TOUCH_AGE_SCALE_TICKS,
    OBS_DIM,
    OBS_FIELD_NAMES,
    ORANGE_PAD_REMAP,
    POSITION_SCALE,
    RIVAL2_REWARD_V2_VERSION,
    RIVAL2_REWARD_VERSION,
    STICKY_TICK_SCALE,
    SUPERSONIC_TIME_SCALE,
    TIME_SINCE_BOOSTED_SCALE,
    contract_hashes_for_reward,
)
from rivalsim.static_world import CompleteWorldSim

_RELATIVE_BALL_POSITION_START = OBS_FIELD_NAMES.index("relative.ball_position.x")


class Rival2EpisodeState:
    """Trainer-owned state kept alongside the accepted v0.4 world."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        for name in (
            "interval_tick",
            "episode_ticks",
            "no_touch_ticks",
            "goal_latched",
            "terminated",
            "truncated",
            "reset_mask",
        ):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.int32, device=device))
        self.scoring_team_latched = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.kickoff_indicator = wp.ones(num_envs, dtype=wp.int32, device=device)
        self.ball_y_before = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.ball_y_after = wp.zeros(num_envs, dtype=wp.float32, device=device)
        car_count = num_envs * 2
        self.touch_count = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.touch_contact_latched = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.demo_by_count = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.demoed_event = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.reward = wp.zeros(car_count, dtype=wp.float32, device=device)
        self.previous_action = wp.zeros(car_count * 8, dtype=wp.float32, device=device)

    @property
    def logical_bytes(self) -> int:
        return self.num_envs * ((9 + 2 * 4) * 4 + 2 * 8 * 4)


class Rival2WorldSim(CompleteWorldSim):
    """CompleteWorldSim plus policy-neutral 30 Hz interval event accounting."""

    def __init__(self, *args: Any, **kwargs: Any):
        if kwargs.get("auto_kickoff") not in (None, False):
            raise ValueError("Rival2WorldSim owns reset timing at decision boundaries")
        kwargs["auto_kickoff"] = False
        super().__init__(*args, **kwargs)
        self.rival2 = Rival2EpisodeState(self.num_envs, self.device)

    @property
    def logical_state_bytes(self) -> int:
        state = getattr(self, "rival2", None)
        return super().logical_state_bytes + (0 if state is None else state.logical_bytes)

    def begin_decision(self) -> None:
        state = self.rival2
        wp.launch(
            rival2_begin_decision,
            dim=self.num_envs,
            inputs=[
                self.state.ball_pos,
                state.interval_tick,
                state.ball_y_before,
                state.ball_y_after,
                state.touch_count,
                state.demo_by_count,
                state.demoed_event,
                state.goal_latched,
                state.scoring_team_latched,
                state.terminated,
                state.truncated,
                state.reset_mask,
                state.reward,
                state.kickoff_indicator,
            ],
            device=self.device,
        )

    def _launch_tick(self) -> None:
        super()._launch_tick()
        state = self.rival2
        wp.launch(
            rival2_accumulate_tick,
            dim=self.num_envs,
            inputs=[
                self.state.ball_pos,
                self.lifecycle.goal_scored,
                self.lifecycle.scoring_team,
                self.car_ball.hit_this_tick,
                self.car_ball_b.hit_this_tick,
                self.car_car.event_count,
                self.car_car.event_bumper,
                self.car_car.event_victim,
                self.car_car.event_is_demo,
                state.interval_tick,
                state.episode_ticks,
                state.no_touch_ticks,
                state.ball_y_before,
                state.ball_y_after,
                state.touch_count,
                state.touch_contact_latched,
                state.demo_by_count,
                state.demoed_event,
                state.goal_latched,
                state.scoring_team_latched,
                state.terminated,
                state.truncated,
                state.reset_mask,
                state.reward,
            ],
            device=self.device,
        )

    def apply_interval_resets(self) -> None:
        """Reset only completed worlds, without advancing physics or the clock."""

        lifecycle = self.lifecycle
        state = self.state
        pair = self.car_car
        ball = self.ball_world
        vehicle = self.vehicle
        mask = self.rival2.reset_mask
        wp.launch(
            rival2_interval_reset,
            dim=self.num_envs,
            inputs=[
                mask,
                lifecycle.episode_tick,
                lifecycle.kickoff_reset,
                lifecycle.kickoff_layout,
                lifecycle.kickoff_selector,
                lifecycle.reset_required,
                lifecycle.ball_scored_last,
                lifecycle.demo_respawn_timer,
                lifecycle.demo_held_valid,
                lifecycle.demo_request,
                lifecycle.respawn_pending,
                pair.car_is_demoed,
                pair.car_contact_id,
                pair.car_contact_cooldown,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.boost,
                state.boosting_time,
                state.time_since_boosted,
                state.on_ground,
                state.air_control_disabled,
                state.has_jumped,
                state.is_jumping,
                state.has_double_jumped,
                state.has_flipped,
                state.is_flipping,
                state.sticky_ticks,
                state.jump_time,
                state.air_time,
                state.air_time_since_jump,
                state.flip_time,
                state.flip_rel_torque,
                state.is_boosting,
                state.is_supersonic,
                state.supersonic_time,
                state.prev_throttle,
                state.prev_steer,
                state.prev_pitch,
                state.prev_yaw,
                state.prev_roll,
                state.prev_jump,
                state.prev_boost,
                state.prev_handbrake,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_position,
                vehicle.solver_orientation,
                vehicle.solver_velocity,
                vehicle.solver_angular_velocity,
                self.boost_pad_cooldown,
                self.boost_pad_previous_locked_car,
            ],
            device=self.device,
        )
        wp.launch(
            rival2_after_interval_reset,
            dim=self.num_envs,
            inputs=[
                mask,
                self.rival2.episode_ticks,
                self.rival2.no_touch_ticks,
                self.rival2.kickoff_indicator,
                self.rival2.touch_count,
                self.rival2.touch_contact_latched,
                self.rival2.demo_by_count,
                self.rival2.demoed_event,
                self.rival2.previous_action,
            ],
            device=self.device,
        )


@dataclass(frozen=True, slots=True)
class Rival2Step:
    observation: torch.Tensor
    transition_observation: torch.Tensor
    emitted_action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reset_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class Rival2ProfiledStep:
    step: Rival2Step
    milliseconds: dict[str, float]


class Rival2TensorBridge:
    """Persistent zero-copy Torch views over Warp world and controller storage."""

    def __init__(self, sim: Rival2WorldSim):
        self.sim = sim
        self.num_envs = sim.num_envs
        self.device = torch.device(sim.device)
        self.views: dict[str, torch.Tensor] = {}
        self._warp_arrays: dict[str, wp.array] = {}
        self._bind_world_views()
        self.position_scale = torch.tensor(POSITION_SCALE, dtype=torch.float32, device=self.device)
        self.pad_durations = torch.tensor(
            [10.0] * 6 + [4.0] * 28, dtype=torch.float32, device=self.device
        )
        self.orange_pad_remap = torch.tensor(ORANGE_PAD_REMAP, dtype=torch.long, device=self.device)
        self.blue_pad_remap = torch.arange(34, dtype=torch.long, device=self.device)
        self.team_signs = torch.tensor(
            ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)),
            dtype=torch.float32,
            device=self.device,
        )

    def _bind(self, name: str, array: wp.array) -> torch.Tensor:
        tensor = wp.to_torch(array)
        if tensor.data_ptr() != array.ptr:
            raise RuntimeError(f"Warp/Torch storage alias failed for {name}")
        if tensor.device != self.device:
            raise RuntimeError(f"device mismatch for {name}: {tensor.device}")
        self.views[name] = tensor
        self._warp_arrays[name] = array
        return tensor

    def _bind_world_views(self) -> None:
        state = self.sim.state
        for name in (
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
            "ball_pos",
            "ball_vel",
            "ball_ang_vel",
        ):
            self._bind(name, getattr(state, name))
        self._bind("wheel_contact", self.sim.vehicle.wheel_contact)
        self._bind("pad_cooldown", self.sim.boost_pad_cooldown)
        self._bind("car_is_demoed", self.sim.car_car.car_is_demoed)
        self._bind("demo_respawn_timer", self.sim.lifecycle.demo_respawn_timer)
        episode = self.sim.rival2
        for name in (
            "episode_ticks",
            "no_touch_ticks",
            "ball_y_before",
            "ball_y_after",
            "touch_count",
            "touch_contact_latched",
            "demoed_event",
            "kickoff_indicator",
            "reward",
            "terminated",
            "truncated",
            "reset_mask",
            "previous_action",
        ):
            self._bind(f"rival2.{name}", getattr(episode, name))
        controls = self.sim.controls
        for name in ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost"):
            self._bind(f"control.{name}", getattr(controls, name))
        self._bind("control.handbrake", controls.handbrake)

    def alias_report(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "warp_ptr": int(self._warp_arrays[name].ptr),
                "torch_ptr": int(tensor.data_ptr()),
                "aliases": bool(self._warp_arrays[name].ptr == tensor.data_ptr()),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "contiguous": bool(tensor.is_contiguous()),
            }
            for name, tensor in self.views.items()
        }

    def set_actions(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.num_envs, 2, 8):
            raise ValueError(f"expected action shape {(self.num_envs, 2, 8)}")
        if action.device != self.device or action.dtype != torch.float32:
            raise ValueError("actions must be float32 on the RivalSim CUDA device")
        analog = action[..., :5].clamp(-1.0, 1.0)
        buttons = (action[..., 5:] >= 0.5).to(torch.float32)
        emitted = torch.cat((analog, buttons), dim=-1).contiguous()
        flat = emitted.reshape(-1, 8)
        for index, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll")):
            self.views[f"control.{name}"].copy_(flat[:, index])
        for index, name in enumerate(("jump", "boost", "handbrake"), start=5):
            self.views[f"control.{name}"].copy_(flat[:, index].to(torch.int32))
        self.views["rival2.previous_action"].reshape(self.num_envs, 2, 8).copy_(emitted)
        return emitted

    @staticmethod
    def _basis(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, z, w = quaternion.unbind(-1)
        forward = torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
            ),
            dim=-1,
        )
        up = torch.stack(
            (
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        )
        return forward, up

    @staticmethod
    def _timer(value: torch.Tensor, scale: float) -> torch.Tensor:
        return (value / scale).clamp(0.0, 1.0).unsqueeze(-1)

    def _car_block(self, car: int, sign: torch.Tensor) -> torch.Tensor:
        count = self.num_envs
        pos = self.views["car_pos"].reshape(count, 2, 3)[:, car]
        vel = self.views["car_vel"].reshape(count, 2, 3)[:, car]
        quat = self.views["car_quat"].reshape(count, 2, 4)[:, car]
        angular = self.views["car_ang_vel"].reshape(count, 2, 3)[:, car]
        forward, up = self._basis(quat)

        def scalar(name: str) -> torch.Tensor:
            return self.views[name].reshape(count, 2)[:, car].to(torch.float32)

        on_ground = scalar("on_ground")
        has_jumped = scalar("has_jumped")
        has_double = scalar("has_double_jumped")
        has_flipped = scalar("has_flipped")
        air_since = scalar("air_time_since_jump")
        jump_available = (has_jumped == 0).to(torch.float32)
        dodge_available = (
            (has_double == 0)
            & (has_flipped == 0)
            & ((on_ground != 0) | ((has_jumped != 0) & (air_since < float(DOUBLEJUMP_MAX_DELAY))))
        ).to(torch.float32)
        wheel = self.views["wheel_contact"].reshape(count, 2, 4)[:, car].to(torch.float32)
        demoed = self.views["car_is_demoed"].reshape(count, 2)[:, car].to(torch.float32)
        demo_timer = self.views["demo_respawn_timer"].reshape(count, 2)[:, car]
        return torch.cat(
            (
                pos * sign / self.position_scale,
                vel * sign / CAR_LINEAR_SPEED_SCALE,
                forward * sign,
                up * sign,
                angular * sign / ANGULAR_SPEED_SCALE,
                (scalar("boost") / BOOST_SCALE).unsqueeze(-1),
                on_ground.unsqueeze(-1),
                has_jumped.unsqueeze(-1),
                scalar("is_jumping").unsqueeze(-1),
                has_double.unsqueeze(-1),
                has_flipped.unsqueeze(-1),
                scalar("is_flipping").unsqueeze(-1),
                jump_available.unsqueeze(-1),
                dodge_available.unsqueeze(-1),
                demoed.unsqueeze(-1),
                self._timer(demo_timer, DEMO_TIMER_SCALE),
                wheel,
                self._timer(scalar("jump_time"), JUMP_TIME_SCALE),
                self._timer(scalar("air_time"), AIR_TIME_SCALE),
                self._timer(air_since, AIR_TIME_SCALE),
                self._timer(scalar("flip_time"), FLIP_TIME_SCALE),
                self._timer(scalar("boosting_time"), BOOSTING_TIME_SCALE),
                self._timer(scalar("time_since_boosted"), TIME_SINCE_BOOSTED_SCALE),
                scalar("is_supersonic").unsqueeze(-1),
                self._timer(scalar("supersonic_time"), SUPERSONIC_TIME_SCALE),
                self._timer(scalar("sticky_ticks"), STICKY_TICK_SCALE),
            ),
            dim=-1,
        )

    def observation(self) -> torch.Tensor:
        count = self.num_envs
        car_pos = self.views["car_pos"].reshape(count, 2, 3)
        car_vel = self.views["car_vel"].reshape(count, 2, 3)
        ball_pos = self.views["ball_pos"].reshape(count, 3)
        ball_vel = self.views["ball_vel"].reshape(count, 3)
        ball_angular = self.views["ball_ang_vel"].reshape(count, 3)
        previous = self.views["rival2.previous_action"].reshape(count, 2, 8)
        touch = self.views["rival2.touch_count"].reshape(count, 2)
        demoed_event = self.views["rival2.demoed_event"].reshape(count, 2)
        kickoff = self.views["rival2.kickoff_indicator"].to(torch.float32).unsqueeze(-1)
        episode_age = self._timer(
            self.views["rival2.episode_ticks"].to(torch.float32),
            EPISODE_AGE_SCALE_TICKS,
        )
        no_touch_age = self._timer(
            self.views["rival2.no_touch_ticks"].to(torch.float32),
            NO_TOUCH_AGE_SCALE_TICKS,
        )
        pad_cooldown = self.views["pad_cooldown"].reshape(count, 34)
        observations: list[torch.Tensor] = []
        for agent in range(2):
            sign = self.team_signs[agent]
            opponent = 1 - agent
            ball = torch.cat(
                (
                    ball_pos * sign / self.position_scale,
                    ball_vel * sign / BALL_LINEAR_SPEED_SCALE,
                    ball_angular * sign / ANGULAR_SPEED_SCALE,
                ),
                dim=-1,
            )
            relative = torch.cat(
                (
                    (ball_pos - car_pos[:, agent]) * sign / self.position_scale,
                    (ball_vel - car_vel[:, agent]) * sign / BALL_LINEAR_SPEED_SCALE,
                    (car_pos[:, opponent] - car_pos[:, agent]) * sign / self.position_scale,
                    (car_vel[:, opponent] - car_vel[:, agent]) * sign / CAR_LINEAR_SPEED_SCALE,
                ),
                dim=-1,
            )
            pad_index = self.blue_pad_remap if agent == 0 else self.orange_pad_remap
            cooldown = pad_cooldown.index_select(1, pad_index)
            durations = self.pad_durations.index_select(0, pad_index)
            pads = torch.stack(
                ((cooldown == 0.0).to(torch.float32), (cooldown / durations).clamp(0.0, 1.0)),
                dim=-1,
            ).reshape(count, 68)
            lifecycle = torch.cat(
                (
                    kickoff,
                    (touch[:, agent] > 0).to(torch.float32).unsqueeze(-1),
                    (touch[:, opponent] > 0).to(torch.float32).unsqueeze(-1),
                    (demoed_event[:, agent] > 0).to(torch.float32).unsqueeze(-1),
                    (demoed_event[:, opponent] > 0).to(torch.float32).unsqueeze(-1),
                    episode_age,
                    no_touch_age,
                ),
                dim=-1,
            )
            observation = torch.cat(
                (
                    ball,
                    self._car_block(agent, sign),
                    self._car_block(opponent, sign),
                    relative,
                    pads,
                    previous[:, agent],
                    lifecycle,
                ),
                dim=-1,
            )
            observations.append(observation)
        result = torch.stack(observations, dim=1).contiguous()
        if result.shape != (count, 2, OBS_DIM):
            raise RuntimeError(f"observation schema mismatch: {result.shape}")
        return result

    def approach_reward(
        self,
        decision_observation: torch.Tensor,
        transition_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Reward V2's per-agent true-distance delta entirely on device."""

        if decision_observation.shape != transition_observation.shape:
            raise ValueError("approach observations must have identical shapes")
        if decision_observation.shape[-1] != OBS_DIM:
            raise ValueError("approach observations must use RIVAL2_OBS_V1")
        if (
            decision_observation.device != self.device
            or transition_observation.device != self.device
        ):
            raise ValueError("approach observations must remain on the RivalSim CUDA device")
        relative_slice = slice(
            _RELATIVE_BALL_POSITION_START,
            _RELATIVE_BALL_POSITION_START + 3,
        )
        before_relative = decision_observation[..., relative_slice] * self.position_scale
        after_relative = transition_observation[..., relative_slice] * self.position_scale
        distance_before = torch.linalg.vector_norm(before_relative, dim=-1)
        distance_after = torch.linalg.vector_norm(after_relative, dim=-1)
        return (distance_before - distance_after) / APPROACH_DISTANCE_SCALE


class Rival2Env:
    """30 Hz Rival 2.0 environment with no host data path."""

    def __init__(
        self,
        num_envs: int,
        collision_root: str,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        reward_version: str = RIVAL2_REWARD_VERSION,
        **world_kwargs: Any,
    ):
        self.reward_version = reward_version
        self.contract_hashes = contract_hashes_for_reward(reward_version)
        self.world = Rival2WorldSim(
            num_envs,
            collision_root,
            device=device,
            seed=seed,
            **world_kwargs,
        )
        self.device = torch.device(self.world.device)
        self._activate_torch_stream()
        self.bridge = Rival2TensorBridge(self.world)
        self.observation = self.bridge.observation()
        self.decision_count = 0

    @property
    def num_envs(self) -> int:
        return self.world.num_envs

    def _activate_torch_stream(self) -> None:
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        wp.set_stream(stream, device=self.world.device, sync=False)

    def _step_impl(
        self,
        action: torch.Tensor,
        markers: list[torch.cuda.Event] | None = None,
    ) -> Rival2Step:
        self._activate_torch_stream()
        if markers is not None:
            markers[0].record()
        decision_observation = self.observation
        self.world.begin_decision()
        emitted = self.bridge.set_actions(action)
        self.world.step(PHYSICS_TICKS_PER_DECISION)
        if markers is not None:
            markers[1].record()
        transition_observation = self.bridge.observation().clone()
        reward = self.bridge.views["rival2.reward"].reshape(self.num_envs, 2).clone()
        if self.reward_version == RIVAL2_REWARD_V2_VERSION:
            reward.add_(
                self.bridge.approach_reward(decision_observation, transition_observation)
            )
        terminated = self.bridge.views["rival2.terminated"].to(torch.bool).clone()
        truncated = self.bridge.views["rival2.truncated"].to(torch.bool).clone()
        reset_mask = self.bridge.views["rival2.reset_mask"].to(torch.bool).clone()
        if markers is not None:
            markers[2].record()
        self.world.apply_interval_resets()
        if markers is not None:
            markers[3].record()
        observation = self.bridge.observation()
        if markers is not None:
            markers[4].record()
        self.observation = observation
        self.decision_count += 1
        return Rival2Step(
            observation=observation,
            transition_observation=transition_observation,
            emitted_action=emitted,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            reset_mask=reset_mask,
        )

    def step(self, action: torch.Tensor) -> Rival2Step:
        return self._step_impl(action)

    def step_profiled(self, action: torch.Tensor) -> Rival2ProfiledStep:
        """Explicit offline phase timing; not used by ordinary rollout."""

        markers = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        result = self._step_impl(action, markers)
        markers[-1].synchronize()
        milliseconds = {
            "physics_reward_action_copy": markers[0].elapsed_time(markers[1]),
            "transition_observation_and_outputs": markers[1].elapsed_time(markers[2]),
            "selective_reset": markers[2].elapsed_time(markers[3]),
            "post_reset_observation": markers[3].elapsed_time(markers[4]),
        }
        return Rival2ProfiledStep(result, milliseconds)

    def hot_path_transfer_bytes(self) -> dict[str, int]:
        return {
            "h2d": int(self.world.host_to_device_bytes),
            "d2h": int(self.world.device_to_host_bytes),
        }

    def reset_transfer_counters(self) -> None:
        self.world.reset_transfer_counters()


__all__ = [
    "Rival2Env",
    "Rival2EpisodeState",
    "Rival2ProfiledStep",
    "Rival2Step",
    "Rival2TensorBridge",
    "Rival2WorldSim",
]
