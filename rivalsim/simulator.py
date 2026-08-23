"""High-level owner for GPU-resident RivalSim v0.1 state."""

from __future__ import annotations

import warp as wp

from rivalsim.controls import ControlBatch
from rivalsim.kernels.integrate import integrate_tick
from rivalsim.state import GpuControls, GpuState, StateSnapshot


class RivalSim:
    """Exactly two cars and one free ball per batched world."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        randomize: bool = True,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        wp.init()
        self.device = str(wp.get_device(device))
        self.num_envs = num_envs
        initial = (
            StateSnapshot.random(num_envs, seed) if randomize else StateSnapshot.empty(num_envs)
        )
        self.state = GpuState(initial, self.device)
        self.controls = GpuControls(ControlBatch.zeros(num_envs), self.device)
        self._captured_graph: wp.Graph | None = None
        self._captured_graph_ticks = 0
        self.tick_count = 0
        self.host_to_device_bytes = initial.nbytes + ControlBatch.zeros(num_envs).nbytes
        self.device_to_host_bytes = 0

    @property
    def logical_state_bytes(self) -> int:
        # The allocation uses the same FP32/int32 footprint as the host snapshot.
        return StateSnapshot.empty(self.num_envs).nbytes

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        if state is None:
            state = StateSnapshot.random(self.num_envs, 0 if seed is None else seed)
        if state.num_envs != self.num_envs:
            raise ValueError("reset world count differs")
        self.state = GpuState(state, self.device)
        self._captured_graph = None
        self._captured_graph_ticks = 0
        self.host_to_device_bytes += state.nbytes
        self.tick_count = 0

    def set_controls(self, controls: ControlBatch) -> None:
        if controls.num_envs != self.num_envs:
            raise ValueError("control world count differs")
        clamped = controls.clamped()
        self.controls = GpuControls(clamped, self.device)
        self._captured_graph = None
        self._captured_graph_ticks = 0
        self.host_to_device_bytes += clamped.nbytes

    def reset_transfer_counters(self) -> None:
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0

    def step(self, ticks: int = 1, *, synchronize: bool = False) -> None:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        for _ in range(ticks):
            self._launch_tick()
        self.tick_count += ticks
        if synchronize:
            self.synchronize()

    def capture_graph(self, block_ticks: int = 8) -> None:
        """Capture a fixed resident-control tick block for low-overhead replay."""

        if block_ticks <= 0:
            raise ValueError("block_ticks must be positive")
        self.synchronize()
        wp.capture_begin(device=self.device)
        for _ in range(block_ticks):
            self._launch_tick()
        self._captured_graph = wp.capture_end(device=self.device)
        self._captured_graph_ticks = block_ticks

    def step_graph(self, ticks: int = 1, *, synchronize: bool = False) -> None:
        """Advance with a captured block plus direct launches for any remainder."""

        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        if self._captured_graph is None:
            raise RuntimeError("capture_graph() must be called before step_graph()")
        blocks, remainder = divmod(ticks, self._captured_graph_ticks)
        for _ in range(blocks):
            wp.capture_launch(self._captured_graph)
        for _ in range(remainder):
            self._launch_tick()
        self.tick_count += ticks
        if synchronize:
            self.synchronize()

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    def snapshot(self) -> StateSnapshot:
        self.synchronize()
        result = self.state.snapshot()
        self.device_to_host_bytes += result.nbytes
        return result

    def _launch_tick(self) -> None:
        state = self.state
        controls = self.controls
        wp.launch(
            integrate_tick,
            dim=state.car_count,
            inputs=[
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.boost,
                state.boosting_time,
                state.time_since_boosted,
                state.on_ground,
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
