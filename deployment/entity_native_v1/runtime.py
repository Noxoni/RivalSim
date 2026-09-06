"""Qualified native RLBot packet runtime for the unchanged joint90 entity actor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Load the standalone inference/scheduler module without importing RivalSim's
# simulator package __init__ or requiring CUDA/Warp in the RLBot environment.
sys.path.insert(0, str(ROOT / "rivalsim"))
sys.path.insert(0, str(HERE))
from entity_native_bridge import RecurrentPacketScheduler
from packet_observation import Rival2LiveAdapter, _phase_name

VERSION = "RIVAL2_ENTITY_NATIVE_PACKET_RUNTIME_V1"
OBS_HASH = "10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF"
QUALITY = {"direct": 0, "derived": 1, "approximate": 2, "unavailable": 3}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


class EntityPacketRuntime:
    def __init__(self, manifest, field_info, *, model=None):
        if manifest["format"] != VERSION:
            raise RuntimeError("Wrong native packet runtime format")
        if manifest["contracts"] != dict(physics_hz=120, policy_hz=30, hold_ticks=4,
                observation_dim=182, observation_schema_sha256=OBS_HASH,
                action_version="RIVAL2_ACTION_JOINT90_30HZ_V1"):
            raise RuntimeError("Packet cadence/observation/action contract mismatch")
        self.manifest = manifest
        if model is None:
            path = (ROOT / manifest["artifact"]["path"]).resolve()
            path.relative_to(ROOT)
            if sha(path) != manifest["artifact"]["sha256"]:
                raise RuntimeError("Export SHA mismatch")
            model = torch.jit.load(str(path), map_location="cpu").eval()
        self.adapter = Rival2LiveAdapter(manifest, field_info)
        # Throwaway recurrent state: warm-up never becomes initial game history.
        begin = time.perf_counter()
        with torch.inference_mode():
            for _ in range(20):
                action, hidden, logits = model(torch.zeros(1,182), torch.zeros(1,1,256))
                if not all(bool(torch.isfinite(x).all()) for x in (action, hidden, logits)):
                    raise RuntimeError("Nonfinite warm-up output")
        self.warmup_ms = (time.perf_counter()-begin)*1000
        self.scheduler = RecurrentPacketScheduler(model, 256)
        self.last_phase = "Inactive"
        self.phase_epoch = 0
        self.score = None
        self.binding = None
        self.last_packet_frame = None
        self.last_decision = None
        self.base_quality = np.asarray([QUALITY[f["classification"]] for f in manifest["fields"]], dtype=np.uint8)
        if self.base_quality.shape != (182,) or [f["index"] for f in manifest["fields"]] != list(range(182)):
            raise RuntimeError("Incomplete observation quality metadata")
        self.interval_missed = 0
        self.episode_history_degraded = False
        self.pauses = 0

    def step(self, packet, own_player_id):
        self.last_decision = None
        phase = _phase_name(packet)
        frame = int(packet.match_info.frame_num)
        if frame < 0:
            raise RuntimeError("Invalid physics frame")
        players = list(packet.players)
        own = [p for p in players if int(p.player_id) == int(own_player_id)]
        if len(own) != 1 or len(players) != 2 or len(packet.teams) != 2:
            raise RuntimeError("Need uniquely bound local player in standard1v1")
        team = int(own[0].team)
        indices = self.adapter._team_indices(packet)
        binding = tuple(int(players[int(i)].player_id) for i in indices)
        score = tuple(int(t.score) for t in packet.teams)
        rebound = binding != self.binding
        scored = self.score is not None and score != self.score
        countdown = phase == "Countdown" and self.last_phase != "Countdown"
        # A stale packet in an unchanged player/score stream cannot reset memory.
        if (self.last_packet_frame is not None and frame < self.last_packet_frame
                and not rebound and not scored and not countdown):
            self.scheduler.stats["out_of_order"] += 1
            return self.scheduler.action.numpy().copy() if phase in {"Active", "Kickoff"} else np.zeros(8, np.float32)
        if rebound or scored or countdown:
            self.phase_epoch += 1
        self.binding, self.score = binding, score
        if phase == "Paused":
            if self.last_phase != "Paused":
                self.pauses += 1
            self.scheduler.pause(frame)
            self.last_packet_frame, self.last_phase = frame, phase
            return np.zeros(8, np.float32)
        if phase not in {"Active", "Kickoff"}:
            self.scheduler.step(frame=frame, identity=self.phase_epoch, active=False,
                                observation=lambda: None, lifecycle=lambda d,r: None)
            self.last_packet_frame, self.last_phase = frame, phase
            return np.zeros(8, np.float32)
        if len(packet.balls) != 1:
            raise RuntimeError("Need one ball")
        if self.last_phase == "Paused" and self.scheduler.last_frame is not None:
            # Ignore paused frame-number advance; at most the current actionable
            # endpoint tick resumes the clock. Does not invent intermediate states.
            self.scheduler.pause(max(self.scheduler.last_frame, frame-1))
        self.last_packet_frame, self.last_phase = frame, phase
        prior_decisions = self.scheduler.stats["decisions"]

        def lifecycle(delta, reset):
            if reset:
                self.adapter.reset(packet)
                self.interval_missed = 0
                self.episode_history_degraded = phase != "Kickoff"
            else:
                self.adapter.advance(packet, delta)
                if delta > 1:
                    self.interval_missed += delta-1
                    self.episode_history_degraded = True

        captured = {}
        def observation():
            obs = self.adapter.observation(packet)[0, team].copy()
            quality = self.base_quality.copy()
            if self.episode_history_degraded:
                # Mark previous emitted-vs-consumed history uncertain after gaps;
                # never replace the actual182input values with these masks.
                quality[167:175] = np.maximum(quality[167:175], 2)
            captured.update(observation=obs, quality=quality)
            return torch.from_numpy(obs[None])

        action = self.scheduler.step(frame=frame, identity=self.phase_epoch, active=True,
                                     observation=observation, lifecycle=lifecycle).numpy().copy()
        if self.scheduler.stats["decisions"] != prior_decisions:
            self.last_decision = dict(frame=frame, phase=phase, score=list(score), team=team,
                own_player_id=int(own_player_id), epoch=self.phase_epoch,
                observation=captured["observation"], quality=captured["quality"],
                action=action.copy(), missed_ticks_in_interval=self.interval_missed,
                history_degraded=self.episode_history_degraded)
            self.interval_missed = 0
            self.adapter.memory.previous_action[0, team] = action
            self.adapter.memory.clear_interval_events()
        return action

    def summary(self):
        return dict(format=VERSION, source=self.manifest["source"],
                    scheduler=dict(self.scheduler.stats), warmup_ms=self.warmup_ms,
                    pauses=self.pauses, last_phase=self.last_phase,
                    quality_counts={k:int((self.base_quality==v).sum()) for k,v in QUALITY.items()},
                    packet_domain_exact=False, optimizer_steps=0)
