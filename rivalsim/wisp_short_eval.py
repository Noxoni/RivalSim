"""Paired short-lifecycle Rival 2.0 versus pinned Wisp evaluation runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp

from rivalsim.kernels.rival2 import EPISODE_LIMIT_TICKS, REWARD_MODE_GAMEPLAY
from rivalsim.nexto_short_eval import (
    DEFAULT_DASH_EVENT_CAPACITY,
    RIVAL_CADENCE_TICKS,
    ShortEvalTelemetry,
    ShortEvalTiming,
)
from rivalsim.rival2_contracts import (
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from third_party.wisp75b.adapter import WispPolicyAdapter, WispStateTensors

SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS = (
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
)


class WispShortEpisodeRunner:
    """Rival at 30 Hz against deterministic Wisp's 8/7 tick controller."""

    def __init__(
        self,
        num_worlds: int,
        collision_root: str,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
        stochastic_rival: bool,
        evaluation_seed: int,
        device: str = "cuda:0",
        dash_event_capacity: int = DEFAULT_DASH_EVENT_CAPACITY,
    ):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        layout = np.asarray(starting_layout, dtype=np.int32).reshape(self.num_worlds)
        side = np.asarray(rival_side, dtype=np.int32).reshape(self.num_worlds)
        if np.any((layout < 0) | (layout >= 5)):
            raise ValueError("starting layouts must be in [0, 5)")
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
            raise RuntimeError("Rival checkpoint SHA-256 mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_reward = payload.get("reward_version")
        if checkpoint_reward not in SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS:
            raise RuntimeError("Wisp evaluator requires a Gameplay V1/V2/V3 checkpoint")
        if payload.get("episode_version") != RIVAL2_EPISODE_VERSION:
            raise RuntimeError("checkpoint episode identity is not RIVAL2_EPISODE_V1")
        policy_config = Rival2PolicyConfig(**payload["policy_config"])
        if policy_config.content_hash != payload["policy_config_hash"]:
            raise RuntimeError("Rival checkpoint policy configuration mismatch")
        if payload.get("contract_hashes") != contract_hashes_for_reward(
            checkpoint_reward, RIVAL2_EPISODE_VERSION
        ):
            raise RuntimeError("Rival checkpoint contract mismatch")
        self.rival_policy = Rival2ActorCritic(policy_config).to(self.device)
        self.rival_policy.load_state_dict(payload["model"])
        self.rival_policy.eval()
        self.checkpoint_identity = {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "iteration": int(payload["iteration"]),
            "policy_version": int(payload["policy_version"]),
            "total_agent_samples": int(payload["total_agent_samples"]),
            "policy_config": asdict(policy_config),
            "policy_config_hash": policy_config.content_hash,
            "reward_version": checkpoint_reward,
            "episode_version": payload["episode_version"],
            "contract_hashes": payload["contract_hashes"],
        }
        del payload

        self.stochastic_rival = bool(stochastic_rival)
        self.rival_generator = torch.Generator(device=self.device).manual_seed(int(evaluation_seed))
        self.rival_side = torch.as_tensor(side, dtype=torch.long, device=self.device)
        self.wisp_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.wisp = WispPolicyAdapter(self.num_worlds, device=self.device)
        self.wisp.set_player_index(self.wisp_side)
        self.wisp.activate(torch.ones(self.num_worlds, dtype=torch.bool, device=self.device))
        self.wisp_state = WispStateTensors.from_bridge(self.bridge)
        self.rival_observation = self.bridge.observation()
        self.rival_action = torch.zeros((self.num_worlds, 8), device=self.device)
        self.actions = torch.zeros((self.num_worlds, 2, 8), device=self.device)
        self.all_active = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        self.telemetry = ShortEvalTelemetry(self.world, event_capacity=dash_event_capacity)
        self.telemetry.attach()
        self.host_tick = 0
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.world.capture_graph(block_ticks=1)

    def _activate_stream(self) -> None:
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            actor, _value = self.rival_policy(observation)
            if self.stochastic_rival:
                self.rival_action.copy_(
                    sample_hybrid_action(actor, generator=self.rival_generator).action
                )
            else:
                self.rival_action.copy_(deterministic_hybrid_action(actor))

    def tick(self) -> None:
        self._activate_stream()
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            self._update_rival_action()
            self.world.begin_decision()
        wisp_action, _indices = self.wisp.tick_action(self.wisp_state, active_mask=self.all_active)
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.wisp_side] = wisp_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.host_tick += 1
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            self.world.apply_interval_resets()
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

    def export(self) -> dict[str, object]:
        return {
            "raw": self.telemetry.numpy(),
            "checkpoint_identity": self.checkpoint_identity,
            "starting_layout": self.starting_layout.copy(),
            "rival_side": self.rival_side_host.copy(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "world_host_to_device_bytes_after_initialization": int(self.world.host_to_device_bytes),
            "world_device_to_host_bytes_after_initialization": int(self.world.device_to_host_bytes),
            "wisp_inference_calls": int(self.wisp.inference_calls),
        }


__all__ = ["WispShortEpisodeRunner"]
