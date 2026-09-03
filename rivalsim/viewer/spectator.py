"""Authoritative one-world full-match checkpoint spectator for RivalVis."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.constants import DOUBLEJUMP_MAX_DELAY
from rivalsim.kernels.boost_pad import BIG_PAD_COUNT, SOCCAR_PAD_POSITIONS
from rivalsim.rival2_contracts import (
    OBS_DIM,
    RIVAL2_REWARD_V2_VERSION,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from rivalsim.viewer.frame import (
    BallFrame,
    BoostPadFrame,
    CarFrame,
    TransformFrame,
    ViewerFrame,
)
from third_party.wisp75b.adapter import WispPolicyAdapter, WispStateTensors


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    sha256: str
    format: str
    policy_version: int | None
    iteration: int | None
    total_agent_samples: int | None
    reward_version: str | None
    episode_version: str | None
    observation_version: str
    action_version: str
    physics_hz: int
    policy_hz: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_collision_root(value: str | Path | None) -> Path:
    """Resolve CMFs from CLI, environment, or the standard sibling checkout."""

    candidates: list[Path] = []
    if value is not None:
        candidates.append(Path(value))
    environment = os.environ.get("RIVALSIM_COLLISION_DIR")
    if environment:
        candidates.append(Path(environment))
    repository = Path(__file__).resolve().parents[2]
    candidates.append(repository.parent / "RLBot-Rival" / "bot" / "collision_meshes")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "soccar").is_dir():
            return resolved
    rendered = ", ".join(str(item) for item in candidates) or "<none>"
    raise FileNotFoundError(
        "Soccar collision meshes were not found. Pass --collision-dir or set "
        f"RIVALSIM_COLLISION_DIR. Checked: {rendered}"
    )


def load_checkpoint_policy(
    path: str | Path, device: str | torch.device
) -> tuple[Rival2ActorCritic, Rival2PolicyConfig, CheckpointInfo]:
    """Load a normal resumable checkpoint or a plain Rival model state dict."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_format = "RIVAL2_STATE_DICT"
    metadata: Mapping[str, Any] = {}
    if isinstance(payload, Mapping) and payload.get("format") == "RIVAL2_CHECKPOINT_V1":
        checkpoint_format = "RIVAL2_CHECKPOINT_V1"
        config = Rival2PolicyConfig(**payload["policy_config"])
        state_dict = payload["model"]
        metadata = payload
    elif isinstance(payload, Mapping) and "model" in payload:
        raw_config = payload.get("policy_config", {})
        config = Rival2PolicyConfig(**raw_config)
        state_dict = payload["model"]
        metadata = payload
    elif isinstance(payload, Mapping) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        config = Rival2PolicyConfig()
        state_dict = payload
    else:
        raise ValueError(
            "unsupported Rival checkpoint; expected RIVAL2_CHECKPOINT_V1 or a model state dict"
        )
    model = Rival2ActorCritic(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    info = CheckpointInfo(
        path=checkpoint,
        sha256=digest,
        format=checkpoint_format,
        policy_version=(
            int(metadata["policy_version"])
            if metadata.get("policy_version") is not None
            else None
        ),
        iteration=(
            int(metadata["iteration"]) if metadata.get("iteration") is not None else None
        ),
        total_agent_samples=(
            int(metadata["total_agent_samples"])
            if metadata.get("total_agent_samples") is not None
            else None
        ),
        reward_version=(
            str(metadata["reward_version"])
            if metadata.get("reward_version") is not None
            else None
        ),
        episode_version=(
            str(metadata["episode_version"])
            if metadata.get("episode_version") is not None
            else None
        ),
        observation_version=str(metadata.get("observation_version", "RIVAL2_OBS_V1")),
        action_version=str(metadata.get("action_version", "RIVAL2_ACTION_V1")),
        physics_hz=int(metadata.get("physics_hz", 120)),
        policy_hz=int(metadata.get("policy_hz", 30)),
    )
    return model, config, info


class ViewerStateAdapter:
    """Translate authoritative RivalSim arrays into one renderer-neutral frame."""

    def __init__(self, env: Rival2FullMatchEnv):
        self.env = env
        self.ball_quaternion = wp.to_torch(env.world.state.ball_quat)
        self.hit_blue = wp.to_torch(env.world.car_ball.hit_this_tick)
        self.hit_orange = wp.to_torch(env.world.car_ball_b.hit_this_tick)

    @staticmethod
    def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    @staticmethod
    def _tuple4(values: np.ndarray) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    def capture(
        self,
        *,
        policy_decision: int,
        controls: torch.Tensor,
        rewards: torch.Tensor,
        last_touch: int | None,
        match_finished: bool,
        winner: int | None,
    ) -> ViewerFrame:
        views = self.env.bridge.views
        full = self.env.full_match_views
        torch.cuda.synchronize(self.env.device)
        car_position = views["car_pos"].reshape(1, 2, 3).cpu().numpy()[0]
        car_quaternion = views["car_quat"].reshape(1, 2, 4).cpu().numpy()[0]
        car_velocity = views["car_vel"].reshape(1, 2, 3).cpu().numpy()[0]
        car_angular = views["car_ang_vel"].reshape(1, 2, 3).cpu().numpy()[0]
        boost = views["boost"].reshape(1, 2).cpu().numpy()[0]
        on_ground = views["on_ground"].reshape(1, 2).cpu().numpy()[0]
        wheel_contact = views["wheel_contact"].reshape(1, 2, 4).cpu().numpy()[0]
        has_jumped = views["has_jumped"].reshape(1, 2).cpu().numpy()[0]
        is_jumping = views["is_jumping"].reshape(1, 2).cpu().numpy()[0]
        has_double = views["has_double_jumped"].reshape(1, 2).cpu().numpy()[0]
        has_flipped = views["has_flipped"].reshape(1, 2).cpu().numpy()[0]
        is_flipping = views["is_flipping"].reshape(1, 2).cpu().numpy()[0]
        air_since_jump = views["air_time_since_jump"].reshape(1, 2).cpu().numpy()[0]
        supersonic = views["is_supersonic"].reshape(1, 2).cpu().numpy()[0]
        demoed = views["car_is_demoed"].reshape(1, 2).cpu().numpy()[0]
        ball_position = views["ball_pos"].reshape(1, 3).cpu().numpy()[0]
        ball_velocity = views["ball_vel"].reshape(1, 3).cpu().numpy()[0]
        ball_angular = views["ball_ang_vel"].reshape(1, 3).cpu().numpy()[0]
        ball_quaternion = self.ball_quaternion.reshape(1, 4).cpu().numpy()[0]
        actions = controls.detach().reshape(2, 8).cpu().numpy()
        reward_values = rewards.detach().reshape(2).cpu().numpy()
        touches = np.asarray(
            (
                int(full["match_blue_touches"].item()),
                int(full["match_orange_touches"].item()),
            ),
            dtype=np.int32,
        )
        cars: list[CarFrame] = []
        for side in range(2):
            speed = float(np.linalg.norm(car_velocity[side]))
            distance = float(np.linalg.norm(ball_position - car_position[side]))
            cars.append(
                CarFrame(
                    transform=TransformFrame(
                        self._tuple3(car_position[side]),
                        self._tuple4(car_quaternion[side]),
                    ),
                    linear_velocity=self._tuple3(car_velocity[side]),
                    angular_velocity=self._tuple3(car_angular[side]),
                    boost=float(boost[side]),
                    speed=speed,
                    on_ground=bool(on_ground[side]),
                    wheel_contacts=int(np.count_nonzero(wheel_contact[side])),
                    has_jumped=bool(has_jumped[side]),
                    is_jumping=bool(is_jumping[side]),
                    has_double_jumped=bool(has_double[side]),
                    has_flipped=bool(has_flipped[side]),
                    is_flipping=bool(is_flipping[side]),
                    dodge_available=bool(
                        has_jumped[side]
                        and not has_double[side]
                        and not has_flipped[side]
                        and air_since_jump[side] <= DOUBLEJUMP_MAX_DELAY
                    ),
                    is_supersonic=bool(supersonic[side]),
                    is_demoed=bool(demoed[side]),
                    distance_to_ball=distance,
                    touches=int(touches[side]),
                    reward=float(reward_values[side]),
                    controls=tuple(float(value) for value in actions[side]),  # type: ignore[arg-type]
                )
            )
        cooldown = views["pad_cooldown"].reshape(1, 34).cpu().numpy()[0]
        pads = tuple(
            BoostPadFrame(
                position=self._tuple3(position),
                is_large=index < BIG_PAD_COUNT,
                active=bool(cooldown[index] <= 0.0),
            )
            for index, position in enumerate(SOCCAR_PAD_POSITIONS)
        )
        return ViewerFrame(
            physics_tick=self.env.world.tick_count,
            policy_decision=policy_decision,
            regulation_ticks_remaining=int(full["regulation_ticks_remaining"].item()),
            blue_score=int(full["blue_score"].item()),
            orange_score=int(full["orange_score"].item()),
            overtime=bool(full["overtime"].item()),
            kickoff_active=bool(full["kickoff_segment_active"].item()),
            match_finished=match_finished,
            winner=winner,
            last_touch=last_touch,
            ball=BallFrame(
                transform=TransformFrame(
                    self._tuple3(ball_position), self._tuple4(ball_quaternion)
                ),
                linear_velocity=self._tuple3(ball_velocity),
                angular_velocity=self._tuple3(ball_angular),
            ),
            cars=(cars[0], cars[1]),
            boost_pads=pads,
        )


class RivalVisSession:
    """One checkpoint-controlled full-match world, entirely outside training."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        collision_root: str | Path | None = None,
        device: str = "cuda:0",
        seed: int = 20260827,
        stochastic: bool = True,
        opponent: str = "self",
    ):
        if opponent not in {"self", "wisp"}:
            raise ValueError("RivalVis opponent must be 'self' or 'wisp'")
        self.device = torch.device(device)
        self.collision_root = resolve_collision_root(collision_root)
        self.model, self.policy_config, self.checkpoint_info = load_checkpoint_policy(
            checkpoint, self.device
        )
        self.seed = int(seed)
        self.stochastic = bool(stochastic)
        self.opponent = opponent
        self.blue_label = "RIVAL" if opponent == "wisp" else "BLUE"
        self.orange_label = "WISP" if opponent == "wisp" else "ORANGE"
        self._closed = False
        self._create_match(self.seed)

    def _create_match(self, seed: int) -> None:
        kickoff_selector = np.asarray([seed % 5], dtype=np.int32)
        self.env = Rival2FullMatchEnv(
            1,
            str(self.collision_root),
            device=str(self.device),
            seed=seed,
            reward_version=RIVAL2_REWARD_V2_VERSION,
            observation_version=self.checkpoint_info.observation_version,
            action_version=self.checkpoint_info.action_version,
            kickoff_selector=kickoff_selector,
            car_visitation_order="a_then_b",
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.wisp: WispPolicyAdapter | None = None
        self.wisp_state: WispStateTensors | None = None
        self.wisp_active: torch.Tensor | None = None
        self.physics_reset_mask = wp.to_torch(
            self.env.world.rival2.physics_reset_mask
        )
        if self.opponent == "wisp":
            self.wisp = WispPolicyAdapter(1, device=self.device)
            self.wisp.set_player_index(
                torch.ones(1, dtype=torch.long, device=self.device)
            )
            self.wisp_active = torch.ones(1, dtype=torch.bool, device=self.device)
            self.wisp.activate(self.wisp_active)
            self.wisp_state = WispStateTensors.from_bridge(self.env.bridge)
        self.adapter = ViewerStateAdapter(self.env)
        self.geometry = self.env.world.meshes.geometry
        self.current_action = torch.zeros((1, 2, 8), device=self.device)
        self.last_reward = torch.zeros((1, 2), device=self.device)
        self.last_touch: int | None = None
        self.interval_tick = 0
        self.policy_decision = 0
        self.match_finished = False
        self.winner: int | None = None
        self._decision_observation: torch.Tensor | None = None
        initial = self._capture()
        self.previous_frame = initial
        self.current_frame = initial

    @torch.no_grad()
    def _policy_action(self) -> torch.Tensor:
        observation = self.env.observation
        if self.opponent == "wisp":
            observation = observation[:, 0]
        actor, _value = self.model(observation.reshape(-1, OBS_DIM))
        if self.stochastic:
            action = sample_hybrid_action(
                actor,
                generator=self.generator,
                config=self.policy_config,
            ).action
        else:
            action = deterministic_hybrid_action(actor, self.policy_config)
        if self.opponent == "wisp":
            return action.reshape(1, 8)
        return action.reshape(1, 2, 8)

    def _wisp_action(self) -> torch.Tensor:
        if self.wisp is None or self.wisp_state is None or self.wisp_active is None:
            raise RuntimeError("Wisp opponent state is unavailable")
        score_diff = (
            self.env.full_match_views["orange_score"]
            - self.env.full_match_views["blue_score"]
        ).to(torch.float32)
        action, _indices = self.wisp.tick_action(
            self.wisp_state,
            active_mask=self.wisp_active,
            score_diff=score_diff,
        )
        return action

    def _capture(self) -> ViewerFrame:
        return self.adapter.capture(
            policy_decision=self.policy_decision,
            controls=self.current_action,
            rewards=self.last_reward,
            last_touch=self.last_touch,
            match_finished=self.match_finished,
            winner=self.winner,
        )

    def advance_physics_tick(self) -> ViewerFrame:
        if self._closed:
            raise RuntimeError("RivalVis session is closed")
        if self.match_finished:
            return self.current_frame
        self.env._activate_torch_stream()
        if self.interval_tick == 0:
            self._decision_observation = self.env.observation
            self.env.world.begin_decision()
            if self.opponent == "wisp":
                self.current_action[:, 0].copy_(self._policy_action())
            else:
                self.current_action = self.env.bridge.set_actions(
                    self._policy_action()
                ).clone()
        if self.opponent == "wisp":
            self.current_action[:, 1].copy_(self._wisp_action())
            self.current_action = self.env.bridge.set_actions(self.current_action).clone()
        self.env.world.step(1)
        self.interval_tick += 1
        if bool(self.adapter.hit_blue[0].item()):
            self.last_touch = 0
        if bool(self.adapter.hit_orange[0].item()):
            self.last_touch = 1
        if self.interval_tick == self.env.physics_ticks_per_decision:
            transition_observation = self.env.bridge.observation().clone()
            reward = self.env.bridge.views["rival2.reward"].reshape(1, 2).clone()
            if self._decision_observation is None:
                raise RuntimeError("missing RivalVis decision observation")
            reward.add_(
                self.env.bridge.approach_reward(
                    self._decision_observation, transition_observation
                )
            )
            self.last_reward = reward
            self.policy_decision += 1
            terminated = bool(
                self.env.bridge.views["rival2.terminated"].to(torch.bool).item()
            )
            if terminated:
                self.match_finished = True
                raw_winner = int(self.env.full_match_views["winner"].item())
                self.winner = raw_winner if raw_winner in (0, 1) else None
            else:
                physical_reset = self.physics_reset_mask.to(torch.bool).clone()
                self.env.world.apply_interval_resets()
                if self.wisp is not None and bool(physical_reset.any()):
                    self.wisp.activate(physical_reset)
                self.env.observation = self.env.bridge.observation()
                self.env.decision_count += 1
            self.interval_tick = 0
            self._decision_observation = None
        self.previous_frame = self.current_frame
        self.current_frame = self._capture()
        return self.current_frame

    def advance_policy_decision(self) -> ViewerFrame:
        target = self.env.physics_ticks_per_decision if self.interval_tick == 0 else (
            self.env.physics_ticks_per_decision - self.interval_tick
        )
        for _ in range(target):
            self.advance_physics_tick()
        return self.current_frame

    def restart(self, *, new_seed: bool = False) -> ViewerFrame:
        if new_seed:
            self.seed += 1
        self._create_match(self.seed)
        return self.current_frame

    def checkpoint_unchanged(self) -> bool:
        return _sha256(self.checkpoint_info.path) == self.checkpoint_info.sha256

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self._closed = True


class RivalVisReplay:
    """A complete authoritative match buffered before presentation starts."""

    def __init__(
        self,
        frames: tuple[ViewerFrame, ...],
        *,
        checkpoint_info: CheckpointInfo,
        collision_root: Path,
        device: str,
        seed: int,
        stochastic: bool,
        opponent: str,
        blue_label: str,
        orange_label: str,
        geometry: Any,
        build_seconds: float,
    ):
        if not frames:
            raise ValueError("RivalVis replay must contain at least one frame")
        self.frames = frames
        self.checkpoint_info = checkpoint_info
        self.collision_root = collision_root
        self.device = device
        self.seed = int(seed)
        self.stochastic = bool(stochastic)
        self.opponent = opponent
        self.blue_label = blue_label
        self.orange_label = orange_label
        self.geometry = geometry
        self.build_seconds = float(build_seconds)
        self.frame_index = 0

    @classmethod
    def record(
        cls,
        live: RivalVisSession,
        *,
        maximum_frames: int | None = None,
        progress_interval: int = 1200,
    ) -> RivalVisReplay:
        """Consume a live session into an immutable, consecutive frame replay."""

        if maximum_frames is not None and maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive when provided")
        started = time.perf_counter()
        frames = [live.current_frame]
        try:
            while not live.match_finished and (
                maximum_frames is None or len(frames) < maximum_frames
            ):
                frames.append(live.advance_physics_tick())
                if progress_interval > 0 and (len(frames) - 1) % progress_interval == 0:
                    elapsed = max(time.perf_counter() - started, 1.0e-9)
                    simulated_seconds = (len(frames) - 1) / 120.0
                    print(
                        "RivalVis buffering "
                        f"tick={len(frames) - 1} simulated={simulated_seconds:.1f}s "
                        f"rate={(len(frames) - 1) / elapsed:.1f} ticks/s",
                        flush=True,
                    )
            if not live.checkpoint_unchanged():
                raise RuntimeError(
                    "RivalVis checkpoint changed while the replay was being buffered"
                )
            elapsed = time.perf_counter() - started
            replay = cls(
                tuple(frames),
                checkpoint_info=live.checkpoint_info,
                collision_root=Path(live.collision_root),
                device=str(live.device),
                seed=live.seed,
                stochastic=live.stochastic,
                opponent=live.opponent,
                blue_label=live.blue_label,
                orange_label=live.orange_label,
                geometry=live.geometry,
                build_seconds=elapsed,
            )
        finally:
            live.close()
        print(
            f"RivalVis buffered {len(replay.frames)} consecutive frames "
            f"in {replay.build_seconds:.1f}s",
            flush=True,
        )
        return replay

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        collision_root: str | Path | None = None,
        device: str = "cuda:0",
        seed: int = 20260827,
        stochastic: bool = True,
        opponent: str = "self",
    ) -> RivalVisReplay:
        live = RivalVisSession(
            checkpoint,
            collision_root=collision_root,
            device=device,
            seed=seed,
            stochastic=stochastic,
            opponent=opponent,
        )
        return cls.record(live)

    @property
    def previous_frame(self) -> ViewerFrame:
        return self.frames[max(0, self.frame_index - 1)]

    @property
    def current_frame(self) -> ViewerFrame:
        return self.frames[self.frame_index]

    @property
    def match_finished(self) -> bool:
        return self.frame_index >= len(self.frames) - 1

    @property
    def winner(self) -> int | None:
        return self.current_frame.winner

    def advance_physics_tick(self) -> ViewerFrame:
        if not self.match_finished:
            self.frame_index += 1
        return self.current_frame

    def advance_policy_decision(self) -> ViewerFrame:
        decision = self.current_frame.policy_decision
        while (
            not self.match_finished
            and self.current_frame.policy_decision == decision
        ):
            self.frame_index += 1
        return self.current_frame

    def restart(self, *, new_seed: bool = False) -> ViewerFrame:
        seed = self.seed + int(new_seed)
        replacement = type(self).from_checkpoint(
            self.checkpoint_info.path,
            collision_root=self.collision_root,
            device=self.device,
            seed=seed,
            stochastic=self.stochastic,
            opponent=self.opponent,
        )
        self.__dict__.update(replacement.__dict__)
        return self.current_frame

    def checkpoint_unchanged(self) -> bool:
        return _sha256(self.checkpoint_info.path) == self.checkpoint_info.sha256

    def close(self) -> None:
        return None


__all__ = [
    "CheckpointInfo",
    "RivalVisReplay",
    "RivalVisSession",
    "ViewerStateAdapter",
    "load_checkpoint_policy",
    "resolve_collision_root",
]
