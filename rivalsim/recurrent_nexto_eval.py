"""Deterministic 120 Hz recurrent Rival versus Nexto evaluation runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.kernels.rival2 import EPISODE_LIMIT_TICKS, REWARD_MODE_GAMEPLAY
from rivalsim.nexto_short_eval import (
    PHYSICS_HZ,
    RIVAL_CADENCE_TICKS,
    ShortEvalTelemetry,
    ShortEvalTiming,
)
from rivalsim.open_play import OpenPlayTelemetry
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_recurrent_policy import (
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
    deterministic_recurrent_action,
)
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors


class RecurrentNextoEpisodeRunner:
    """One bounded episode per world with aligned recurrent reset semantics."""

    def __init__(
        self,
        num_worlds: int,
        collision_root: str,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_checkpoint_format: str,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
        evaluation_seed: int,
        device: str = "cuda:0",
    ):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        self.lifecycle_cadence_ticks = RIVAL_CADENCE_TICKS
        layout = np.asarray(starting_layout, dtype=np.int32).reshape(self.num_worlds)
        side = np.asarray(rival_side, dtype=np.int32).reshape(self.num_worlds)
        if np.any((layout < 0) | (layout >= 5)):
            raise ValueError("starting layouts must be in [0,5)")
        if np.any((side < 0) | (side > 1)):
            raise ValueError("Rival side must be Blue=0 or Orange=1")
        self.starting_layout = layout
        self.rival_side_host = side
        self.world = Rival2WorldSim(
            self.num_worlds,
            collision_root,
            device=device,
            seed=evaluation_seed,
            kickoff_selector=layout,
            car_lifecycle_seed=evaluation_seed,
            reward_mode=REWARD_MODE_GAMEPLAY,
        )
        self.warp_stream = wp.get_stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        self._activate_stream()
        self.bridge = Rival2TensorBridge(self.world)

        checkpoint_path = Path(checkpoint_path)
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
        if checkpoint_sha != expected_checkpoint_sha256.upper():
            raise RuntimeError("recurrent checkpoint SHA-256 mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != expected_checkpoint_format:
            raise RuntimeError("unsupported recurrent Stage-1 checkpoint format")
        config = Rival2RecurrentPolicyConfig(**payload["policy_config"])
        if payload.get("policy_config_sha256") != config.content_hash:
            raise RuntimeError("recurrent policy contract hash mismatch")
        self.rival_policy = Rival2RecurrentActorCritic(config).to(self.device)
        self.rival_policy.load_state_dict(payload["model"], strict=True)
        self.rival_policy.eval().requires_grad_(False)
        self.checkpoint_identity = {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "format": payload["format"],
            "selected_step": int(payload["selected_step"]),
            "policy_config": asdict(config),
            "policy_config_sha256": config.content_hash,
            "evaluation_only_stage1_load": True,
        }
        del payload

        self.rival_side = torch.as_tensor(side, dtype=torch.long, device=self.device)
        self.nexto_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.nexto = NextoPolicyAdapter(self.num_worlds, device=self.device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.rival_observation = self.bridge.observation()
        self.rival_hidden = self.rival_policy.initial_hidden(self.num_worlds, device=self.device)
        self.rival_action = torch.zeros(
            (self.num_worlds, 8), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros((self.num_worlds, 2, 8), dtype=torch.float32, device=self.device)
        self.telemetry = ShortEvalTelemetry(self.world)
        self.telemetry.attach()
        self.open_play_telemetry = OpenPlayTelemetry(self.world)
        self.open_play_telemetry.attach(self.world)
        self.host_tick = 0
        self.recurrent_reset_count = 0
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.world.capture_graph(block_ticks=1)

    def _activate_stream(self) -> None:
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)

    @torch.inference_mode()
    def initial_action_probe(self) -> np.ndarray:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        hidden = self.rival_policy.initial_hidden(self.num_worlds, device=self.device)
        actor, _value, _next_hidden = self.rival_policy(observation, hidden)
        return deterministic_recurrent_action(actor).detach().cpu().numpy().astype(np.float32)

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            actor, _value, self.rival_hidden = self.rival_policy(observation, self.rival_hidden)
            self.rival_action.copy_(deterministic_recurrent_action(actor))

    def tick(self) -> None:
        self._activate_stream()
        self._update_rival_action()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        kickoff_active = self.bridge.views["rival2.kickoff_indicator"] != 0
        nexto_action, _indices = self.nexto.tick_action(self.nexto_state, kickoff_active)
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            reset_mask = self.bridge.views["rival2.reset_mask"].to(torch.bool)
            self.nexto.notify_kickoff(reset_mask)
            self.world.apply_interval_resets()
            if bool(torch.any(reset_mask)):
                # ``_update_rival_action`` intentionally creates the recurrent
                # state under inference mode.  Reset that inference tensor
                # under the same mode rather than attempting an ordinary
                # autograd-tracked in-place write at an episode boundary.
                with torch.inference_mode():
                    self.rival_hidden[:, reset_mask] = 0.0
                self.recurrent_reset_count += int(reset_mask.sum().item())
        self.rival_observation = self.bridge.observation()

    def run(self) -> ShortEvalTiming:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(EPISODE_LIMIT_TICKS):
            self.tick()
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        return ShortEvalTiming(
            physics_ticks_requested=EPISODE_LIMIT_TICKS,
            seconds=seconds,
            world_ticks_per_second=self.num_worlds * EPISODE_LIMIT_TICKS / seconds,
        )

    def export(self) -> dict[str, Any]:
        return {
            "raw": self.telemetry.numpy(),
            "open_play_raw": self.open_play_telemetry.numpy(),
            "checkpoint_identity": self.checkpoint_identity,
            "starting_layout": self.starting_layout.copy(),
            "rival_side": self.rival_side_host.copy(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "world_host_to_device_bytes_after_initialization": int(self.world.host_to_device_bytes),
            "world_device_to_host_bytes_after_initialization": int(self.world.device_to_host_bytes),
            "nexto_inference_calls": int(self.nexto.inference_calls),
            "nexto_observation_builds": int(self.nexto.observation_builds),
            "recurrent_reset_count": self.recurrent_reset_count,
            "rival_policy_hz": PHYSICS_HZ,
            "rival_cadence_ticks": 1,
            "lifecycle_cadence_ticks": self.lifecycle_cadence_ticks,
            "initial_hidden": "zero_at_native_playable_kickoff",
        }


__all__ = ["RecurrentNextoEpisodeRunner"]
