"""Authoritative one-world full-match checkpoint spectator for RivalVis."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.constants import DOUBLEJUMP_MAX_DELAY
from rivalsim.kernels.boost_pad import BIG_PAD_COUNT, SOCCAR_PAD_POSITIONS
from rivalsim.kernels.rival2 import PHYSICS_TICKS_PER_DECISION
from rivalsim.rival2_contracts import OBS_DIM, RIVAL2_REWARD_V2_VERSION
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
    ):
        self.device = torch.device(device)
        self.collision_root = resolve_collision_root(collision_root)
        self.model, self.policy_config, self.checkpoint_info = load_checkpoint_policy(
            checkpoint, self.device
        )
        self.seed = int(seed)
        self.stochastic = bool(stochastic)
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
            kickoff_selector=kickoff_selector,
            car_visitation_order="a_then_b",
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.adapter = ViewerStateAdapter(self.env)
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
        actor, _value = self.model(self.env.observation.reshape(-1, OBS_DIM))
        if self.stochastic:
            action = sample_hybrid_action(
                actor,
                generator=self.generator,
                config=self.policy_config,
            ).action
        else:
            action = deterministic_hybrid_action(actor, self.policy_config)
        return action.reshape(1, 2, 8)

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
            self.current_action = self.env.bridge.set_actions(self._policy_action()).clone()
        self.env.world.step(1)
        self.interval_tick += 1
        if bool(self.adapter.hit_blue[0].item()):
            self.last_touch = 0
        if bool(self.adapter.hit_orange[0].item()):
            self.last_touch = 1
        if self.interval_tick == PHYSICS_TICKS_PER_DECISION:
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
                self.env.world.apply_interval_resets()
                self.env.observation = self.env.bridge.observation()
                self.env.decision_count += 1
            self.interval_tick = 0
            self._decision_observation = None
        self.previous_frame = self.current_frame
        self.current_frame = self._capture()
        return self.current_frame

    def advance_policy_decision(self) -> ViewerFrame:
        target = PHYSICS_TICKS_PER_DECISION if self.interval_tick == 0 else (
            PHYSICS_TICKS_PER_DECISION - self.interval_tick
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


__all__ = [
    "CheckpointInfo",
    "RivalVisSession",
    "ViewerStateAdapter",
    "load_checkpoint_policy",
    "resolve_collision_root",
]
