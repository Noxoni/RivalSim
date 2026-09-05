"""Bounded, deterministic post-pilot match comparison; never an optimizer.

Uses existing FullMatchRunner lifecycle/telemetry. The narrow subclass changes
only checkpoint construction and recurrent action/reset scheduling. No legacy
checkpoint is loaded as a compatibility dummy, and no candidate is selected
using this evaluation. Run only after the entity pilot has stopped at +100.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import warp as wp

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_joint_control import (
    CHECKPOINTS,
    COLLISION,
    EXTERNAL,
    RESULTS,
    verify,
)
from benchmarks.run_rival2_ssl_exploration_comparison import PARENT, PARENT_SHA
from rivalsim.fresh_ground_30hz import policy_config
from rivalsim.full_match import (
    REGULATION_TICKS,
    FullMatchRunner,
    FullMatchState,
    FullMatchTelemetry,
)
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic
from rivalsim.rival2_unified_policy import deterministic_unified_action
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

SEED = 2026090573
OVERTIME_CAP_TICKS = 120 * 120
PROTOCOL = RESULTS / "full_match_protocol.json"
OUTPUT = RESULTS / "full_match_comparison.json"
SOURCES = (
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "rivalsim/full_match.py",
    "tests/test_ssl_entity_full_match.py",
)


def assignments():
    # Five standard layouts, each on both sides, identical for both policies.
    return np.tile(np.arange(5, dtype=np.int32), 2), np.repeat(np.arange(2, dtype=np.int32), 5)


def spec():
    layouts, sides = assignments()
    return dict(
        version="RIVAL2_ENTITY_POST100_FULL_MATCH_COMPARISON_V1",
        checkpoints=["immutable_hybrid_parent_597", "entity_fixed_plus100"],
        candidate_selection=False,
        seed=SEED,
        starting_layouts=layouts.tolist(),
        rival_sides=sides.tolist(),
        matches_per_policy=10,
        regulation_ticks=REGULATION_TICKS,
        overtime_cap_ticks=OVERTIME_CAP_TICKS,
        unresolved_overtime="report unresolved; never count as a win",
        physics_hz=120,
        rival_policy_hz=30,
        nexto_policy_hz=15,
        action_sampling=False,
        hidden="zero initially and on every native kickoff reset; continuous otherwise",
        resets="Existing full-match goal/tied-regulation kickoff only; no no-touch resets",
        reward_optimization=False,
        diagnostic_match_reward="Existing match runtime reward buffers unused by policy/optimizer",
        active_training_concurrency=False,
        interpretation="Small deterministic development comparison, not SSL proof or ranked play",
    )


def reset_hidden(hidden, mask):
    return hidden.masked_fill(mask[None, :, None], 0)


class CandidateMatchRunner(FullMatchRunner):
    def __init__(self, checkpoint, checkpoint_sha, *, entity):
        # Reuse the accepted world, match state, telemetry and inherited export/
        # timing methods. Avoid the legacy base constructor's hybrid-only loader.
        assert sha(checkpoint) == checkpoint_sha
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.entity = entity
        self.rival_policy = (
            EntityJointControlActorCritic()
            if entity
            else IndependentCriticActorCritic(policy_config())
        )
        self.rival_policy.load_state_dict(payload["model"], strict=True)
        self.model_hash_before = tensor_hash(self.rival_policy.state_dict())
        assert all(bool(torch.isfinite(v).all()) for v in payload["model"].values())
        self.checkpoint_identity = dict(
            path=str(checkpoint.relative_to(ROOT)),
            sha256=checkpoint_sha,
            format=payload["format"],
            policy_hz=30,
            physics_hz=120,
            action="joint90_argmax" if entity else "hybrid_mean_button_threshold",
        )
        del payload
        layout, side = assignments()
        self.num_worlds, self.device = len(side), torch.device("cuda:0")
        self.world = Rival2WorldSim(
            self.num_worlds,
            COLLISION,
            device="cuda:0",
            seed=SEED,
            kickoff_selector=layout,
            car_lifecycle_seed=SEED,
        )
        self.warp_stream = wp.get_stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        self._activate_stream()
        self.rival_policy = self.rival_policy.to(self.device).eval()
        self.hidden = self.rival_policy.initial_hidden(self.num_worlds)
        self.bridge = Rival2TensorBridge(self.world)
        self.match = FullMatchState(self.num_worlds, self.world.device, layout, side)
        self.match_views = self.match.torch_views()
        self.telemetry = FullMatchTelemetry(self.match)
        self.telemetry.attach(self.world)
        self.rival_side = self.match_views["rival_side"].long()
        self.nexto_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.nexto = NextoPolicyAdapter(self.num_worlds, device=self.device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.rival_observation = self.bridge.observation()
        self.rival_action = torch.zeros((self.num_worlds, 8), device=self.device)
        self.actions = torch.zeros((self.num_worlds, 2, 8), device=self.device)
        self.rival_cadence_ticks = self.lifecycle_cadence_ticks = 4
        self.host_tick = 0
        self.hidden_reset_count = torch.zeros(self.num_worlds, device=self.device)
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.world.capture_graph(block_ticks=1)

    @torch.inference_mode()
    def _update_rival_action(self):
        observation = self.rival_observation[self.batch_index, self.rival_side]
        actor, hidden = self.rival_policy.forward_actor(observation, self.hidden)
        action = (
            self.rival_policy.deterministic(actor)
            if self.entity
            else deterministic_unified_action(actor)
        )
        if not bool(torch.isfinite(action).all() & torch.isfinite(hidden).all()):
            raise RuntimeError("nonfinite full-match policy output")
        self.rival_action.copy_(action)
        self.hidden = hidden

    def tick(self):
        self._activate_stream()
        if self.host_tick % 4 == 0:
            self._update_rival_action()
            self.world.begin_decision()
        controls, _ = self.nexto.tick_action(
            self.nexto_state,
            self.match_views["kickoff_active"] != 0,
            active_mask=self.match_views["done"] == 0,
        )
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = controls
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.match_views["rival_scheduler_tick"].add_(1).remainder_(4)
        self.match_views["nexto_scheduler_tick"].add_(1).remainder_(8)
        self.host_tick += 1
        if self.host_tick % 4 == 0:
            # Exactly the inherited match-owned physical reset boundary, with
            # an additional recurrent reset. Never reset on short-episode age.
            wp.copy(self.world.rival2.reset_mask, self.match.pending_reset)
            reset = self.match_views["pending_reset"] != 0
            self.hidden = reset_hidden(self.hidden, reset)
            self.hidden_reset_count += reset
            self.nexto.notify_kickoff(reset)
            self.world.apply_interval_resets()
            self.telemetry.after_resets(self.world, self.world.rival2.reset_mask)
            self.rival_observation = self.bridge.observation()


def summarize(raw):
    side = raw["match.rival_side"].astype(np.int64)
    rows = np.arange(len(side))
    done = raw["match.done"] != 0
    winner = raw["match.winner"]
    scores = np.stack((raw["match.blue_score"], raw["match.orange_score"]), 1)
    result = dict(
        wins=int((done & (winner == side)).sum()),
        losses=int((done & (winner != side)).sum()),
        unresolved=int((~done).sum()),
        goals_for=int(scores[rows, side].sum()),
        goals_against=int(scores[rows, 1 - side].sum()),
        touches=int(raw["touch_count"][rows, side].sum()),
        matches_without_rival_touch=int((raw["touch_count"][rows, side] == 0).sum()),
        kickoff_first_touches=int(raw["kickoff_first_touch_count"][rows, side].sum()),
        native_goal_resets=True,
        no_touch_truncations=0,
        no_touch_semantics="Zero by protocol; use matches_without_rival_touch, not reset count",
    )
    result["touches_per_minute"] = result["touches"] / (
        float(raw["match.total_ticks"].sum()) / 7200
    )
    return result


def prepare():
    if PROTOCOL.exists():
        raise RuntimeError("post-pilot protocol already frozen")
    tests = RESULTS / "full_match_unit_tests.xml"
    suites = ET.parse(tests).getroot().findall("testsuite")
    assert suites and all(
        int(s.get("failures", "0")) == 0 and int(s.get("errors", "0")) == 0 for s in suites
    )
    write_json(
        PROTOCOL,
        dict(spec=spec(), sources={p: sha(ROOT / p) for p in SOURCES}, tests_sha256=sha(tests)),
    )


def run():
    verify(published=True)
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["spec"] == spec()
    for name, digest in protocol["sources"].items():
        assert sha(ROOT / name) == digest, name
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        remote = subprocess.check_output(["git", "show", f"origin/main:{name}"], cwd=ROOT)
        assert remote.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["accepted_updates"] == 100 and state["status"] == "completed", state
    # The training worker holds this exact lease until it has fully exited.
    # Refuse concurrent GPU use even if it just wrote a terminal status file.
    import msvcrt

    lease = (EXTERNAL / "campaign.lock").open("r+b")
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    if OUTPUT.exists():
        raise RuntimeError("comparison already completed; never silently rerun")
    evaluation = json.loads((RESULTS / "evaluation_100.json").read_text())
    paths = (
        (PARENT, PARENT_SHA, False),
        (
            CHECKPOINTS / "plus_100.pt",
            evaluation["checkpoint"]["sha256"],
            True,
        ),
    )
    results = []
    for checkpoint, digest, entity in paths:
        arm_output = RESULTS / (
            "full_match_entity100.json" if entity else "full_match_parent597.json"
        )
        if arm_output.exists():
            result = json.loads(arm_output.read_text())
            assert result["protocol_sha256"] == sha(PROTOCOL)
            assert result["checkpoint"]["sha256"] == digest == sha(checkpoint)
            assert result["model_unchanged"] and result["checkpoint_unchanged"]
            results.append(result)
            continue
        runner = CandidateMatchRunner(checkpoint, digest, entity=entity)
        timing = runner.run_ticks(REGULATION_TICKS)
        elapsed = timing.seconds
        for _ in range(OVERTIME_CAP_TICKS // 600):
            if bool(runner.phase_status()["done"].all()):
                break
            elapsed += runner.run_ticks(600).seconds
        export = runner.export()
        raw = export.pop("raw")
        assert not bool(raw["goal_overflow"].any()), "goal telemetry overflow"
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        assert sha(checkpoint) == digest
        result = dict(
            protocol_sha256=sha(PROTOCOL),
            checkpoint=export["checkpoint"],
            summary=summarize(raw),
            raw={k: v.tolist() for k, v in raw.items()},
            wall_seconds=elapsed,
            hidden_resets=runner.hidden_reset_count.cpu().tolist(),
            model_unchanged=True,
            checkpoint_unchanged=True,
            optimizer_steps=0,
        )
        results.append(result)
        write_json(arm_output, result)
        print(json.dumps(dict(checkpoint=str(checkpoint), summary=result["summary"])), flush=True)
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUTPUT, dict(utc=utc(), protocol_sha256=sha(PROTOCOL), policies=results))
    lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else run()
