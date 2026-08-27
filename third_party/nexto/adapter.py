"""Faithful CUDA adapter for the pinned public Nexto 1v1 policy.

This file is an adaptation of the observation, action-table, and kickoff
semantics in Rolv-Arild/Necto at commit
``2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca``.  The upstream work and this
adaptation are distributed under CC BY-NC-SA 4.0; see ``LICENSE`` and
``PROVENANCE.json`` in this directory.

The production path consumes existing CUDA tensors, constructs the exact
1v1 ``q``/``kv``/mask tuple on CUDA, executes the pinned TorchScript actor on
CUDA, and performs beta=1 lookup-table selection on CUDA.  Host conversion is
intentionally absent from the timed policy path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rivalsim.constants import DOUBLEJUMP_MAX_DELAY
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS

UPSTREAM_COMMIT = "2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca"
MODEL_SHA256 = "BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA"

IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
ACTIONS = slice(24, 32)

NEXTO_BOOST_LOCATIONS = np.asarray(
    (
        (0.0, -4240.0, 70.0),
        (-1792.0, -4184.0, 70.0),
        (1792.0, -4184.0, 70.0),
        (-3072.0, -4096.0, 73.0),
        (3072.0, -4096.0, 73.0),
        (-940.0, -3308.0, 70.0),
        (940.0, -3308.0, 70.0),
        (0.0, -2816.0, 70.0),
        (-3584.0, -2484.0, 70.0),
        (3584.0, -2484.0, 70.0),
        (-1788.0, -2300.0, 70.0),
        (1788.0, -2300.0, 70.0),
        (-2048.0, -1036.0, 70.0),
        (0.0, -1024.0, 70.0),
        (2048.0, -1036.0, 70.0),
        (-3584.0, 0.0, 73.0),
        (-1024.0, 0.0, 70.0),
        (1024.0, 0.0, 70.0),
        (3584.0, 0.0, 73.0),
        (-2048.0, 1036.0, 70.0),
        (0.0, 1024.0, 70.0),
        (2048.0, 1036.0, 70.0),
        (-1788.0, 2300.0, 70.0),
        (1788.0, 2300.0, 70.0),
        (-3584.0, 2484.0, 70.0),
        (3584.0, 2484.0, 70.0),
        (0.0, 2816.0, 70.0),
        (-940.0, 3310.0, 70.0),
        (940.0, 3308.0, 70.0),
        (-3072.0, 4096.0, 73.0),
        (3072.0, 4096.0, 73.0),
        (-1792.0, 4184.0, 70.0),
        (1792.0, 4184.0, 70.0),
        (0.0, 4240.0, 70.0),
    ),
    dtype=np.float32,
)


def nexto_pad_mapping() -> tuple[np.ndarray, np.ndarray]:
    """Return Nexto->Rival pad indices and coordinate residuals.

    The pinned source has one historical hard-coded coordinate at
    ``(-940, 3310, 70)`` while the authoritative Soccar location is
    ``(-940, 3308, 70)``.  Nearest unique coordinate matching preserves
    Nexto's exact entity coordinate and maps availability to that physical pad.
    """

    distances = np.linalg.norm(
        NEXTO_BOOST_LOCATIONS[:, None, :] - SOCCAR_PAD_POSITIONS[None, :, :],
        axis=-1,
    )
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    residual = distances[np.arange(34), mapping].astype(np.float32)
    if len(np.unique(mapping)) != 34 or float(residual.max()) > 2.0:
        raise RuntimeError("Nexto/RivalSim boost-pad coordinate mapping is not one-to-one")
    return mapping, residual


def _source_action_rows() -> list[list[int]]:
    actions: list[list[int]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, int(handbrake)]
                        )
    return actions


NEXTO_ACTION_COUNT = 90


def build_action_table(device: torch.device | str) -> torch.Tensor:
    table = torch.tensor(_source_action_rows(), dtype=torch.float32, device=device)
    if table.shape != (NEXTO_ACTION_COUNT, 8):
        raise RuntimeError(f"unexpected Nexto action table shape {tuple(table.shape)}")
    return table


def build_kickoff_sequence(device: torch.device | str) -> torch.Tensor:
    rows: list[list[float]] = []

    def extend(count: int, row: list[float]) -> None:
        rows.extend([row] * count)

    extend(11 * 4, [1, 0, 0, 0, 0, 0, 1, 0])
    extend(4 * 4, [1, -1, 0, -1, 0, 0, 1, 0])
    extend(2 * 4, [1, 0, 0, 0, 0, 1, 1, 0])
    extend(1 * 4, [1, 0, 0, 0, 0, 0, 1, 0])
    extend(1 * 4, [1, 0, -0.7, 0.8, 0, 1, 1, 0])
    extend(13 * 4, [1, 0, 1, 0, 0, 0, 1, 0])
    extend(10 * 4, [1, 0, 0.5, 0, 1, 0, 0, 0])
    return torch.tensor(rows, dtype=torch.float32, device=device)


KICKOFF_LENGTH = 168


def stable_tensor_hash(tensor: torch.Tensor) -> str:
    """Hash float32 row-major content; evidence-only and outside hot paths."""

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class NextoStateTensors:
    car_pos: torch.Tensor
    car_vel: torch.Tensor
    car_quat: torch.Tensor
    car_ang_vel: torch.Tensor
    car_boost: torch.Tensor
    car_demoed: torch.Tensor
    on_ground: torch.Tensor
    has_double_jumped: torch.Tensor
    has_flipped: torch.Tensor
    air_time_since_jump: torch.Tensor
    ball_pos: torch.Tensor
    ball_vel: torch.Tensor
    ball_ang_vel: torch.Tensor
    pad_cooldown: torch.Tensor

    @classmethod
    def from_bridge(cls, bridge: object) -> "NextoStateTensors":
        count = int(getattr(bridge, "num_envs"))
        views = getattr(bridge, "views")
        return cls(
            car_pos=views["car_pos"].reshape(count, 2, 3),
            car_vel=views["car_vel"].reshape(count, 2, 3),
            car_quat=views["car_quat"].reshape(count, 2, 4),
            car_ang_vel=views["car_ang_vel"].reshape(count, 2, 3),
            car_boost=views["boost"].reshape(count, 2),
            car_demoed=views["car_is_demoed"].reshape(count, 2),
            on_ground=views["on_ground"].reshape(count, 2),
            has_double_jumped=views["has_double_jumped"].reshape(count, 2),
            has_flipped=views["has_flipped"].reshape(count, 2),
            air_time_since_jump=views["air_time_since_jump"].reshape(count, 2),
            ball_pos=views["ball_pos"].reshape(count, 3),
            ball_vel=views["ball_vel"].reshape(count, 3),
            ball_ang_vel=views["ball_ang_vel"].reshape(count, 3),
            pad_cooldown=views["pad_cooldown"].reshape(count, 34),
        )


@dataclass(frozen=True, slots=True)
class NextoObservation:
    q: torch.Tensor
    kv: torch.Tensor
    mask: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q, self.kv, self.mask


@dataclass(frozen=True, slots=True)
class NextoDeviceConstants:
    pad_locations: torch.Tensor
    pad_index: torch.Tensor
    invert: torch.Tensor
    norm: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str) -> "NextoDeviceConstants":
        target = torch.device(device)
        pad_mapping, _ = nexto_pad_mapping()
        return cls(
            pad_locations=torch.as_tensor(
                NEXTO_BOOST_LOCATIONS.astype(np.float64), dtype=torch.float64, device=target
            ),
            pad_index=torch.as_tensor(pad_mapping, dtype=torch.long, device=target),
            invert=torch.tensor(
                [1.0] * 5 + [-1.0, -1.0, 1.0] * 5 + [1.0] * 4,
                dtype=torch.float64,
                device=target,
            ),
            norm=torch.tensor(
                [1.0] * 5 + [2300.0] * 6 + [1.0] * 6 + [5.5] * 3 + [1.0] * 4,
                dtype=torch.float64,
                device=target,
            ),
        )


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


def build_nexto_observation(
    state: NextoStateTensors,
    player_index: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    constants: NextoDeviceConstants | None = None,
) -> NextoObservation:
    """Build the pinned source's exact logical 1v1 observation on CUDA."""

    device = state.car_pos.device
    count = state.car_pos.shape[0]
    if player_index.shape != (count,) or player_index.device != device:
        raise ValueError("player_index must be one device-resident side index per world")
    if previous_action.shape != (count, 8) or previous_action.device != device:
        raise ValueError("previous_action must be a device-resident [world, 8] tensor")
    if state.car_pos.dtype != torch.float32 or previous_action.dtype != torch.float32:
        raise ValueError("Nexto state and action tensors must be float32")

    if constants is None:
        constants = NextoDeviceConstants.create(device)
    batch = torch.arange(count, device=device)
    self_index = player_index.to(torch.long)
    opponent_index = 1 - self_index
    physical_order = torch.stack((self_index, opponent_index), dim=1)

    # Pinned nexto_obs.py allocates NumPy's default float64 arrays and converts
    # them to torch.float32 only after all observation arithmetic.
    q = torch.zeros((count, 1, 32), dtype=torch.float64, device=device)
    kv = torch.zeros((count, 37, 24), dtype=torch.float64, device=device)
    mask = torch.zeros((count, 37), dtype=torch.float64, device=device)

    kv[:, 2, IS_BALL] = 1.0
    kv[:, 2, POS] = state.ball_pos.to(torch.float64)
    kv[:, 2, LIN_VEL] = state.ball_vel.to(torch.float64)
    kv[:, 2, ANG_VEL] = state.ball_ang_vel.to(torch.float64)

    pad_locations = constants.pad_locations
    pad_index = constants.pad_index
    kv[:, 3:, IS_BOOST] = 1.0
    kv[:, 3:, POS] = pad_locations
    kv[:, 3:, BOOST] = 0.12 + 0.88 * (pad_locations[:, 2] > 72.0).to(torch.float64)
    kv[:, 3:, DEMO] = (
        state.pad_cooldown.index_select(1, pad_index) == 0.0
    ).to(torch.float64)

    teams = torch.stack((self_index, opponent_index), dim=1).to(torch.float64)
    kv[:, :2, IS_MATE] = 1.0 - teams
    kv[:, :2, IS_OPP] = teams
    for entity in range(2):
        physical = physical_order[:, entity]
        pos = state.car_pos[batch, physical]
        vel = state.car_vel[batch, physical]
        quat = state.car_quat[batch, physical].to(torch.float64)
        angular = state.car_ang_vel[batch, physical].to(torch.float64)
        forward, up = _basis(quat)
        kv[:, entity, POS] = pos.to(torch.float64)
        kv[:, entity, LIN_VEL] = vel.to(torch.float64)
        kv[:, entity, FW] = forward
        kv[:, entity, UP] = up
        kv[:, entity, ANG_VEL] = angular
        kv[:, entity, BOOST] = state.car_boost[batch, physical].to(torch.float64) / 100.0
        kv[:, entity, DEMO] = state.car_demoed[batch, physical].to(torch.float64)
        kv[:, entity, ON_GROUND] = state.on_ground[batch, physical].to(torch.float64)
        kv[:, entity, HAS_FLIP] = (
            (state.has_flipped[batch, physical] == 0)
            & (state.has_double_jumped[batch, physical] == 0)
            & (state.air_time_since_jump[batch, physical] < float(DOUBLEJUMP_MAX_DELAY))
        ).to(torch.float64)
    kv[:, 0, IS_SELF] = 1.0

    orange = self_index == 1
    orange_entities = orange[:, None, None]
    kv *= torch.where(orange_entities, constants.invert, 1.0)
    mate = kv[:, :, IS_MATE].clone()
    opponent = kv[:, :, IS_OPP].clone()
    orange_rows = orange[:, None]
    kv[:, :, IS_MATE] = torch.where(orange_rows, opponent, mate)
    kv[:, :, IS_OPP] = torch.where(orange_rows, mate, opponent)

    kv /= constants.norm
    q[:, 0, :24] = kv[:, 0, :]
    q[:, 0, ACTIONS] = previous_action.to(torch.float64)

    kv[:, :, POS] -= q[:, :, POS]
    forward = q[:, :, FW]
    theta = torch.atan2(forward[..., 0], forward[..., 1]).unsqueeze(-1)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    xs = kv[:, :, POS.start : ANG_VEL.stop : 3]
    ys = kv[:, :, POS.start + 1 : ANG_VEL.stop : 3]
    nx = ct * xs - st * ys
    ny = st * xs + ct * ys
    kv[:, :, POS.start : ANG_VEL.stop : 3] = nx
    kv[:, :, POS.start + 1 : ANG_VEL.stop : 3] = ny
    return NextoObservation(q=q.to(torch.float32), kv=kv.to(torch.float32), mask=mask.to(torch.float32))


def _patch_pinned_actor_for_cuda(
    actor: torch.jit.ScriptModule, device: torch.device
) -> tuple[int, int]:
    """Retarget the traced action constant without changing weights or math.

    The 2021 artifact traced an explicit ``torch.device('cpu')`` around its
    embedded action table.  ``map_location`` already relocates that constant's
    storage, but the stale graph literal otherwise forces a device mismatch.
    Retargeting the single literal makes the original compiled graph executable
    on CUDA and is checked prospectively by CPU-vs-CUDA logit parity.
    """

    patched = 0
    graph = actor.net.output.graph
    for node in graph.nodes():
        if (
            node.kind() == "prim::Constant"
            and "value" in node.attributeNames()
            and node.kindOf("value") == "s"
            and node.s("value") == "cpu"
        ):
            node.s_("value", str(device))
            patched += 1
    if patched != 1:
        raise RuntimeError(f"expected one pinned TorchScript CPU literal, patched {patched}")
    # map_location also moves old traced zero-dimensional shape constants to
    # CUDA.  Each MultiheadAttention block then converts its CUDA head-count
    # scalar to a Python int, causing an otherwise unnecessary D2H copy.  Keep
    # those four immutable scalar metadata constants on CPU.  CUDA arithmetic
    # accepts zero-dimensional CPU scalars as kernel scalar arguments without a
    # host/device tensor transfer; the model weights, activations, attention,
    # logits, and embedded 90x8 action constant remain CUDA resident.
    scalar_constants_rehomed = 0
    for _name, block in actor.net.earl.blocks.named_children():
        for node in block.attention.graph.nodes():
            if (
                node.kind() == "prim::Constant"
                and "value" in node.attributeNames()
                and node.kindOf("value") == "t"
                and node.t("value").numel() == 1
            ):
                node.t_("value", node.t("value").cpu())
                scalar_constants_rehomed += 1
    if scalar_constants_rehomed != 4:
        raise RuntimeError(
            "expected four pinned attention scalar constants, rehomed "
            f"{scalar_constants_rehomed}"
        )
    return patched, scalar_constants_rehomed


class NextoPolicyAdapter:
    """Batched deterministic Nexto policy plus native cadence/kickoff state."""

    def __init__(
        self,
        num_worlds: int,
        *,
        device: torch.device | str = "cuda:0",
        model_path: str | Path | None = None,
    ):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the production Nexto adapter requires CUDA")
        if model_path is None:
            model_path = Path(__file__).with_name("nexto-model.pt")
        model_path = Path(model_path)
        if hashlib.sha256(model_path.read_bytes()).hexdigest().upper() != MODEL_SHA256:
            raise RuntimeError("pinned Nexto model SHA-256 mismatch")
        self.actor = torch.jit.load(str(model_path), map_location=self.device).eval()
        (
            self.compatibility_graph_literals_patched,
            self.compatibility_scalar_constants_rehomed,
        ) = _patch_pinned_actor_for_cuda(self.actor, self.device)
        self.action_table = build_action_table(self.device)
        self.kickoff_sequence = build_kickoff_sequence(self.device)
        self.constants = NextoDeviceConstants.create(self.device)
        if self.kickoff_sequence.shape != (KICKOFF_LENGTH, 8):
            raise RuntimeError("unexpected Nexto kickoff sequence shape")
        self.player_index = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)
        self.previous_action = torch.zeros(
            (self.num_worlds, 8), dtype=torch.float32, device=self.device
        )
        self.neural_counter = torch.zeros(
            self.num_worlds, dtype=torch.int64, device=self.device
        )
        self._cadence_tick = 0
        self.kickoff_index = torch.full(
            (self.num_worlds,), -1, dtype=torch.int64, device=self.device
        )
        self.inference_calls = 0
        self.observation_builds = 0
        self.timed_h2d_bytes = 0
        self.timed_d2h_bytes = 0

    def set_player_index(self, player_index: torch.Tensor) -> None:
        if player_index.shape != (self.num_worlds,) or player_index.device != self.device:
            raise ValueError("player_index must be a CUDA tensor with one side per world")
        if not torch.all((player_index == 0) | (player_index == 1)):
            raise ValueError("Nexto side indices must be 0 or 1")
        self.player_index.copy_(player_index)

    def notify_kickoff(self, reset_mask: torch.Tensor | None = None) -> None:
        if reset_mask is None:
            self.kickoff_index.zero_()
            return
        if reset_mask.shape != (self.num_worlds,) or reset_mask.device != self.device:
            raise ValueError("kickoff reset mask must be device resident")
        self.kickoff_index.copy_(
            torch.where(reset_mask, torch.zeros_like(self.kickoff_index), self.kickoff_index)
        )

    def logits(self, observation: NextoObservation) -> torch.Tensor:
        with torch.inference_mode():
            output, _weights = self.actor(observation.as_tuple())
        self.inference_calls += 1
        return output

    def neural_action(
        self,
        state: NextoStateTensors,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observation = build_nexto_observation(
            state,
            self.player_index,
            self.previous_action,
            constants=self.constants,
        )
        self.observation_builds += 1
        if active_mask is None:
            logits = self.logits(observation)
            indices = torch.argmax(logits, dim=-1)
            action = self.action_table.index_select(0, indices)
            self.previous_action.copy_(action)
            return action, indices
        selected = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        indices = torch.full(
            (self.num_worlds,), -1, dtype=torch.long, device=self.device
        )
        action = self.previous_action.clone()
        if selected.numel() > 0:
            selected_observation = NextoObservation(
                q=observation.q.index_select(0, selected),
                kv=observation.kv.index_select(0, selected),
                mask=observation.mask.index_select(0, selected),
            )
            logits = self.logits(selected_observation)
            selected_indices = torch.argmax(logits, dim=-1)
            selected_action = self.action_table.index_select(0, selected_indices)
            indices.index_copy_(0, selected, selected_indices)
            action.index_copy_(0, selected, selected_action)
            self.previous_action.index_copy_(0, selected, selected_action)
        return action, indices

    def activate(self, active_mask: torch.Tensor) -> None:
        """Initialize only worlds newly assigned to a Nexto episode."""

        if active_mask.shape != (self.num_worlds,) or active_mask.device != self.device:
            raise ValueError("Nexto activation mask must be device resident")
        self.previous_action.masked_fill_(active_mask[:, None], 0.0)
        self.neural_counter.copy_(
            torch.where(active_mask, torch.zeros_like(self.neural_counter), self.neural_counter)
        )
        self.kickoff_index.copy_(
            torch.where(active_mask, torch.zeros_like(self.kickoff_index), self.kickoff_index)
        )

    def tick_action(
        self,
        state: NextoStateTensors,
        kickoff_active: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return this tick's controls and advance the 15/120 Hz state.

        All worlds begin together and the stock bot never resets its neural
        cadence after a goal, so the ordinary 8-tick inference cadence remains
        batch-aligned.  Kickoff indices remain per-world because goals occur at
        different ticks.
        """

        if kickoff_active.shape != (self.num_worlds,) or kickoff_active.device != self.device:
            raise ValueError("kickoff_active must be device resident")
        all_active = active_mask is None
        if all_active:
            active_mask = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        if active_mask.shape != (self.num_worlds,) or active_mask.device != self.device:
            raise ValueError("Nexto active mask must be device resident")
        indices: torch.Tensor | None = None
        if all_active:
            if self._cadence_tick == 0:
                _action, indices = self.neural_action(state)
        else:
            compute = active_mask & (self.neural_counter == 0)
            if compute.any():
                _action, indices = self.neural_action(state, compute)

        inactive = active_mask & ~kickoff_active
        self.kickoff_index.copy_(
            torch.where(inactive, torch.full_like(self.kickoff_index, -1), self.kickoff_index)
        )
        newly_active = active_mask & kickoff_active & (self.kickoff_index < 0)
        self.kickoff_index.copy_(
            torch.where(newly_active, torch.zeros_like(self.kickoff_index), self.kickoff_index)
        )
        in_sequence = (
            active_mask
            & kickoff_active
            & (self.kickoff_index >= 0)
            & (self.kickoff_index < KICKOFF_LENGTH)
            & (state.ball_pos[:, 1] == 0.0)
        )
        sequence_action = self.kickoff_sequence.index_select(
            0, self.kickoff_index.clamp(0, KICKOFF_LENGTH - 1)
        )
        self.previous_action.copy_(
            torch.where(in_sequence[:, None], sequence_action, self.previous_action)
        )
        self.kickoff_index.copy_(
            torch.where(
                active_mask & kickoff_active,
                self.kickoff_index + 1,
                self.kickoff_index,
            )
        )
        self.neural_counter.copy_(
            torch.where(
                active_mask,
                torch.remainder(self.neural_counter + 1, 8),
                self.neural_counter,
            )
        )
        if all_active:
            self._cadence_tick = (self._cadence_tick + 1) % 8
        return self.previous_action, indices


__all__ = [
    "ACTIONS",
    "ANG_VEL",
    "BOOST",
    "DEMO",
    "FW",
    "HAS_FLIP",
    "IS_BALL",
    "IS_BOOST",
    "IS_MATE",
    "IS_OPP",
    "IS_SELF",
    "KICKOFF_LENGTH",
    "LIN_VEL",
    "MODEL_SHA256",
    "NEXTO_ACTION_COUNT",
    "NEXTO_BOOST_LOCATIONS",
    "ON_GROUND",
    "POS",
    "UP",
    "UPSTREAM_COMMIT",
    "NextoObservation",
    "NextoDeviceConstants",
    "NextoPolicyAdapter",
    "NextoStateTensors",
    "build_action_table",
    "build_kickoff_sequence",
    "build_nexto_observation",
    "nexto_pad_mapping",
    "stable_tensor_hash",
]
