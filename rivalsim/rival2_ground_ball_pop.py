"""Physical ground-ball pop scenarios and a pre-contact launch controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.state import StateSnapshot

GROUND_BALL_POP_VERSION = "RIVAL2_GROUND_BALL_POP_V1"


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


def build_ground_ball_pop_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
) -> StateSnapshot:
    """Create deterministic moving ground-ball approaches near the target goal."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid ground-ball pop scenario request")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    forward = _yaw_quat(sign * math.pi / 2.0)
    reverse = _yaw_quat(-sign * math.pi / 2.0)
    for world in range(worlds):
        x = float(rng.uniform(-750.0, 750.0))
        y = float(rng.uniform(2_000.0, 2_900.0))
        ball_speed = float(rng.uniform(100.0, 350.0))
        approach_distance = float(rng.uniform(300.0, 550.0))
        approach_speed = float(ball_speed + rng.uniform(400.0, 800.0))
        state.ball_pos[world] = (x, sign * y, 92.75)
        state.ball_vel[world] = (
            float(rng.uniform(-25.0, 25.0)),
            sign * ball_speed,
            0.0,
        )
        state.car_pos[world, attacker_side] = (
            x + float(rng.uniform(-18.0, 18.0)),
            sign * (y - approach_distance),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-20.0, 20.0)),
            sign * approach_speed,
            0.0,
        )
        state.car_pos[world, other] = (
            float(rng.uniform(-850.0, 850.0)),
            -sign * float(rng.uniform(3_900.0, 4_500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
    state.validate()
    return state


@dataclass(frozen=True, slots=True)
class PrecontactPopConfig:
    trigger_distance_uu: float = 175.0
    first_jump_hold_ticks: int = 8
    jump_release_ticks: int = 6
    second_jump: bool = True
    pitch: float = 0.5
    approach_steer_gain: float = 3.5
    use_approach_boost: bool = False

    def __post_init__(self) -> None:
        if self.trigger_distance_uu <= 130.0:
            raise ValueError("pre-contact trigger must precede ordinary contact")
        if self.first_jump_hold_ticks <= 0 or self.jump_release_ticks <= 0:
            raise ValueError("pre-contact jump timing must be positive")
        if not -1.0 <= self.pitch <= 1.0:
            raise ValueError("pre-contact pitch must be in [-1,1]")
        if self.approach_steer_gain == 0.0:
            raise ValueError("pre-contact steer gain cannot be zero")

    @property
    def second_jump_tick(self) -> int:
        return self.first_jump_hold_ticks + self.jump_release_ticks

    @property
    def learned_start_tick(self) -> int:
        return self.second_jump_tick + int(self.second_jump)


@dataclass(frozen=True, slots=True)
class PrecontactPopStep:
    action: torch.Tensor
    launch_started: torch.Tensor
    approaching: torch.Tensor
    primitive: torch.Tensor
    learned_control: torch.Tensor
    planar_distance_uu: torch.Tensor


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack([observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1)


class PrecontactPopController:
    """Approach a ground ball, jump before contact, then hand off to the actor."""

    def __init__(
        self,
        worlds: int,
        *,
        device: str | torch.device,
        config: PrecontactPopConfig,
    ) -> None:
        if worlds <= 0:
            raise ValueError("pre-contact controller needs at least one world")
        self.worlds = int(worlds)
        self.device = torch.device(device)
        self.config = config
        self.launch_age = torch.full((worlds,), -1, dtype=torch.int64, device=self.device)

    def step(self, learned_action: torch.Tensor, observation: torch.Tensor) -> PrecontactPopStep:
        if learned_action.shape != (self.worlds, 8):
            raise ValueError("learned pre-contact action must be [worlds,8]")
        if observation.shape != (self.worlds, 182):
            raise ValueError("pre-contact observation must be [worlds,182]")
        scale = torch.as_tensor(POSITION_SCALE, dtype=observation.dtype, device=observation.device)
        relative = _vector(observation, "relative.ball_position") * scale
        planar = torch.linalg.vector_norm(relative[:, :2], dim=-1)
        planar_direction = relative[:, :2] / planar[:, None].clamp_min(1.0e-6)
        forward = _vector(observation, "self.forward")
        forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(
            1.0e-6
        )
        up = _vector(observation, "self.up")
        up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True).clamp_min(1.0e-6)
        right = torch.linalg.cross(forward, up, dim=-1)
        right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1.0e-6)
        local_right = (right[:, :2] * planar_direction).sum(dim=-1)
        launch_started = (self.launch_age < 0) & (planar <= self.config.trigger_distance_uu)
        self.launch_age.masked_fill_(launch_started, 0)
        approaching = self.launch_age < 0
        first_hold = (self.launch_age >= 0) & (self.launch_age < self.config.first_jump_hold_ticks)
        release = (self.launch_age >= self.config.first_jump_hold_ticks) & (
            self.launch_age < self.config.second_jump_tick
        )
        second = self.config.second_jump & (self.launch_age == self.config.second_jump_tick)
        primitive = first_hold | release | second
        learned_control = self.launch_age >= self.config.learned_start_tick

        action = learned_action.clone()
        action[approaching] = 0.0
        action[approaching, 0] = 1.0
        action[approaching, 1] = (local_right[approaching] * self.config.approach_steer_gain).clamp(
            -1.0, 1.0
        )
        action[approaching, 6] = float(self.config.use_approach_boost)
        action[primitive] = 0.0
        action[primitive, 0] = 1.0
        action[first_hold, 2] = self.config.pitch
        action[first_hold, 5] = 1.0
        action[release, 2] = self.config.pitch
        action[second, 5] = 1.0
        self.launch_age += (self.launch_age >= 0).to(torch.int64)
        return PrecontactPopStep(
            action=action,
            launch_started=launch_started,
            approaching=approaching,
            primitive=primitive,
            learned_control=learned_control,
            planar_distance_uu=planar,
        )


__all__ = [
    "GROUND_BALL_POP_VERSION",
    "PrecontactPopConfig",
    "PrecontactPopController",
    "PrecontactPopStep",
    "build_ground_ball_pop_scenarios",
]
