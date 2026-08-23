"""v0.2 shared-arena simulator with selectable B0/B1/B2/B3 layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.vehicle import (
    chassis_contacts,
    increment_tick_counter,
    load_action_tape,
    wheel_pre_tick,
)
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot
from rivalsim.vehicle_state import VehicleSnapshot, VehicleState

VARIANT_LEVEL = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}


@dataclass(frozen=True, slots=True)
class ActionTape:
    throttle: np.ndarray
    steer: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    jump: np.ndarray
    boost: np.ndarray
    handbrake: np.ndarray
    hold_ticks: int = 4

    @classmethod
    def deterministic(cls, length: int = 64, hold_ticks: int = 4) -> ActionTape:
        if length <= 0 or hold_ticks <= 0:
            raise ValueError("action tape length and hold_ticks must be positive")
        index = np.arange(length, dtype=np.float32)
        throttle = np.sin(index * np.float32(0.37)).astype(np.float32)
        steer = np.sin(index * np.float32(0.73) + np.float32(0.2)).astype(np.float32)
        pitch = (np.sin(index * np.float32(0.19)) * np.float32(0.35)).astype(np.float32)
        yaw = (np.cos(index * np.float32(0.23)) * np.float32(0.25)).astype(np.float32)
        roll = (np.sin(index * np.float32(0.29)) * np.float32(0.2)).astype(np.float32)
        jump = ((np.arange(length) % 29) == 0).astype(np.int32)
        boost = ((np.arange(length) % 11) < 3).astype(np.int32)
        handbrake = ((np.arange(length) % 17) < 4).astype(np.int32)
        return cls(throttle, steer, pitch, yaw, roll, jump, boost, handbrake, hold_ticks)

    @property
    def length(self) -> int:
        return len(self.throttle)

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.throttle,
                self.steer,
                self.pitch,
                self.yaw,
                self.roll,
                self.jump,
                self.boost,
                self.handbrake,
            )
        )


class DeviceActionTape:
    def __init__(self, tape: ActionTape, device: str):
        self.length = tape.length
        self.hold_ticks = tape.hold_ticks
        for name in ("throttle", "steer", "pitch", "yaw", "roll"):
            setattr(self, name, wp.array(getattr(tape, name), dtype=wp.float32, device=device))
        for name in ("jump", "boost", "handbrake"):
            setattr(self, name, wp.array(getattr(tape, name), dtype=wp.int32, device=device))


class StaticWorldSim(RivalSim):
    """RivalSim v0.2 with one shared immutable arena mesh per device."""

    def __init__(
        self,
        num_envs: int,
        collision_root: str,
        *,
        variant: str = "B3",
        device: str = "cuda:0",
        seed: int = 0,
        initial: StateSnapshot | None = None,
        geometry: ArenaGeometry | None = None,
        meshes: WarpArenaMeshes | None = None,
        action_tape: ActionTape | None = None,
    ):
        if variant not in VARIANT_LEVEL:
            raise ValueError(f"unknown benchmark variant: {variant}")
        super().__init__(num_envs, device=device, seed=seed, randomize=initial is None)
        if initial is not None:
            self.reset(initial)
        self.variant = variant
        self.variant_level = VARIANT_LEVEL[variant]
        self.geometry = geometry or ArenaGeometry.load_soccar(collision_root)
        self.meshes = meshes or WarpArenaMeshes(self.geometry, self.device)
        self.vehicle = VehicleState(self.state.car_count, self.device)
        self.tick_counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.action_tape: DeviceActionTape | None = None
        if action_tape is not None:
            self.set_action_tape(action_tape)

    @property
    def logical_state_bytes(self) -> int:
        base = StateSnapshot.empty(self.num_envs).nbytes
        vehicle = getattr(self, "vehicle", None)
        return base + (0 if vehicle is None else vehicle.logical_bytes)

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        super().reset(state, seed=seed)
        if hasattr(self, "vehicle"):
            self.vehicle = VehicleState(self.state.car_count, self.device)
        self.tick_counter = wp.zeros(1, dtype=wp.int32, device=self.device)

    def set_action_tape(self, tape: ActionTape) -> None:
        self.action_tape = DeviceActionTape(tape, self.device)
        self.host_to_device_bytes += tape.nbytes
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def disable_action_tape(self) -> None:
        self.action_tape = None
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def vehicle_snapshot(self) -> VehicleSnapshot:
        self.synchronize()
        result = self.vehicle.snapshot()
        self.device_to_host_bytes += self.vehicle.logical_bytes
        return result

    def _launch_tick(self) -> None:
        if self.action_tape is not None:
            tape = self.action_tape
            controls = self.controls
            wp.launch(
                load_action_tape,
                dim=self.state.car_count,
                inputs=[
                    self.tick_counter,
                    tape.hold_ticks,
                    tape.length,
                    tape.throttle,
                    tape.steer,
                    tape.pitch,
                    tape.yaw,
                    tape.roll,
                    tape.jump,
                    tape.boost,
                    tape.handbrake,
                    controls.throttle,
                    controls.steer,
                    controls.pitch,
                    controls.yaw,
                    controls.roll,
                    controls.jump,
                    controls.boost,
                    controls.handbrake,
                ],
                device=self.device,
            )
        if self.variant_level >= 1:
            state = self.state
            controls = self.controls
            vehicle = self.vehicle
            wp.launch(
                wheel_pre_tick,
                dim=state.car_count,
                inputs=[
                    self.meshes.cubql.id,
                    int(self.variant_level >= 2),
                    state.car_pos,
                    state.car_vel,
                    state.car_quat,
                    state.car_ang_vel,
                    state.on_ground,
                    state.boost,
                    controls.throttle,
                    controls.steer,
                    controls.boost,
                    controls.handbrake,
                    vehicle.wheel_ray_start,
                    vehicle.wheel_direction,
                    vehicle.wheel_hit_point,
                    vehicle.wheel_hit_normal,
                    vehicle.wheel_hit_distance,
                    vehicle.suspension_length,
                    vehicle.suspension_velocity,
                    vehicle.suspension_clipped_factor,
                    vehicle.suspension_force,
                    vehicle.suspension_pushback,
                    vehicle.engine_acceleration,
                    vehicle.brake_acceleration,
                    vehicle.steer_angle,
                    vehicle.lateral_friction,
                    vehicle.longitudinal_friction,
                    vehicle.wheel_contact,
                    vehicle.wheel_world_contact,
                    vehicle.handbrake_value,
                    vehicle.wheels_with_contact,
                ],
                device=self.device,
            )
        super()._launch_tick()
        if self.variant_level >= 3:
            state = self.state
            vehicle = self.vehicle
            wp.launch(
                chassis_contacts,
                dim=state.car_count,
                inputs=[
                    self.meshes.default.id,
                    state.car_pos,
                    state.car_vel,
                    state.car_quat,
                    state.car_ang_vel,
                    vehicle.candidate_count,
                    vehicle.contact_count,
                    vehicle.contact_point,
                    vehicle.contact_normal,
                    vehicle.contact_penetration,
                ],
                device=self.device,
            )
        wp.launch(
            increment_tick_counter,
            dim=1,
            inputs=[self.tick_counter],
            device=self.device,
        )


def make_contact_rich_state(num_envs: int, seed: int = 20260823) -> StateSnapshot:
    """Deterministic floor/ramp/wall/ceiling/landing/body-contact mixture."""

    state = StateSnapshot.empty(num_envs)
    rng = np.random.default_rng(seed)
    patterns = np.arange(num_envs * 2).reshape(num_envs, 2) % 8
    for env in range(num_envs):
        for car in range(2):
            pattern = int(patterns[env, car])
            yaw = rng.uniform(-np.pi, np.pi)
            pitch = 0.0
            roll = 0.0
            pos = np.array(
                (rng.uniform(-2500, 2500), rng.uniform(-3500, 3500), 17.0),
                dtype=np.float32,
            )
            velocity = rng.uniform(-250, 250, 3).astype(np.float32)
            velocity[2] *= 0.15
            if pattern == 1:  # tilted / partial landing
                pos[2] = 32.0
                pitch = 0.28
                roll = -0.18
                velocity[2] = -350.0
            elif pattern == 2:  # side wall
                pos[:] = (-4070.0, rng.uniform(-3500, 3500), rng.uniform(300, 1500))
                pitch = np.pi / 2.0
                velocity[:] = (40.0, rng.uniform(-500, 500), rng.uniform(-100, 100))
            elif pattern == 3:  # opposite wall / scrape
                pos[:] = (4070.0, rng.uniform(-3500, 3500), rng.uniform(200, 1400))
                pitch = -np.pi / 2.0
                velocity[:] = (-40.0, rng.uniform(-500, 500), rng.uniform(-100, 100))
            elif pattern == 4:  # ceiling
                pos[:] = (rng.uniform(-2500, 2500), rng.uniform(-3500, 3500), 2028.0)
                roll = np.pi
                velocity[:] = (rng.uniform(-300, 300), rng.uniform(-300, 300), 15.0)
            elif pattern == 5:  # corner / ramp approach
                pos[:] = (
                    rng.choice((-3850.0, 3850.0)),
                    rng.choice((-4600.0, 4600.0)),
                    rng.uniform(90, 300),
                )
                pitch = rng.choice((-0.35, 0.35))
                velocity[:] = rng.uniform(-600, 600, 3)
            elif pattern == 6:  # back wall / goal family
                pos[:] = (
                    rng.uniform(-700, 700),
                    rng.choice((-5100.0, 5100.0)),
                    rng.uniform(120, 700),
                )
                roll = rng.choice((-np.pi / 2.0, np.pi / 2.0))
                velocity[:] = rng.uniform(-350, 350, 3)
            elif pattern == 7:  # off-center body strike
                pos[2] = 4.0
                pitch = 0.55
                roll = 0.35
                velocity[:] = (rng.uniform(-500, 500), rng.uniform(-500, 500), -180.0)
            state.car_pos[env, car] = pos
            state.car_vel[env, car] = velocity
            state.car_ang_vel[env, car] = rng.uniform(-0.8, 0.8, 3)
            state.car_quat[env, car] = _quat_from_euler(roll, pitch, yaw)
            state.boost[env, car] = rng.uniform(20.0, 100.0)
    state.validate()
    return state


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )
