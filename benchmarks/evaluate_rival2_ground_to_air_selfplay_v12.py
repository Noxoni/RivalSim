"""Evaluate the selected V12 aerial option inside the frozen V23 policy.

This is a read-only deployment check.  V23 supplies ordinary actions and the
selected V12 aerial scorer is latched only by its frozen observation-only
router.  The trace uses authoritative 120 Hz contacts and goals; it does not
classify named mechanics or modify a reward.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import analyze_rival2_v23_physical_telemetry as physical  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as base  # noqa: E402
from rivalsim.full_match import REGULATION_TICKS  # noqa: E402
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    ROUTE_NAMES,
    AerialOptionRouterConfig,
    AerialOptionSelfPlayRouter,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_DEPLOYMENT_EVALUATION"
RESULTS = ROOT / "results/rival2/ground_to_air_selfplay_v12"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
SELECTED = (
    ROOT
    / "checkpoints/rival2/ground_to_air_selfplay_v12"
    / "rival2_ground_to_air_selfplay_v12_u0060.pt"
)
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
SELECTED_SHA256 = "0A80DD35040D5FE354240D4E4E4F4B2CD50EB342CC95985647D3B0947DB092B2"
SEED = 2026090206


class V12CompositePhysicalTelemetryRunner(physical.PhysicalTelemetryRunner):
    """Side-specialized V23 with one shared selected aerial option."""

    orange_checkpoint = ORANGE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        payload = torch.load(SELECTED, map_location="cpu", weights_only=False)
        if payload.get("format") != "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_CHECKPOINT":
            raise RuntimeError("unexpected selected V12 checkpoint format")
        if int(payload.get("iteration", -1)) != 60:
            raise RuntimeError("selected V12 checkpoint is not update 60")
        provenance = payload.get("provenance", {})
        if provenance.get("blue_v23", {}).get("sha256") != BLUE_SHA256:
            raise RuntimeError("selected V12 Blue provenance mismatch")
        if provenance.get("orange_v23", {}).get("sha256") != ORANGE_SHA256:
            raise RuntimeError("selected V12 Orange provenance mismatch")
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != self.checkpoint_identity["blue"]["policy_config"]:
            raise RuntimeError("selected aerial option architecture differs from V23")
        self.option_policy = Rival2ActorCritic(config).to(self.device)
        self.option_policy.load_state_dict(payload["model"], strict=True)
        self.option_policy.eval().requires_grad_(False)
        router_config = AerialOptionRouterConfig(**payload["router_config"])
        reward_config = AerialSelfPlayRewardConfig(**payload["aerial_reward_config"])
        self.aerial_router = AerialOptionSelfPlayRouter(
            self.num_worlds,
            device=self.device,
            router_config=router_config,
            reward_config=reward_config,
        )
        shape = (self.trace_ticks, self.num_worlds)
        for name in (
            "option_active",
            "option_activated",
            "option_route",
            "option_contact",
            "option_entry_contact",
            "option_second_contact",
            "option_productive_contact",
            "option_goal",
            "option_ground_failure",
        ):
            self.trace[name] = torch.empty(shape, dtype=torch.int16, device=self.device)
        self._option_before = torch.zeros(
            (self.num_worlds, config.obs_dim), dtype=torch.float32, device=self.device
        )
        self._option_active = torch.zeros(
            self.num_worlds, dtype=torch.bool, device=self.device
        )
        self._option_activated = self._option_active.clone()
        self._option_route = torch.full(
            (self.num_worlds,), -1, dtype=torch.int64, device=self.device
        )

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation)
            orange_actor, _ = self.orange_policy(observation)
            base_actor = torch.where(
                (self.rival_side == 1).unsqueeze(1), orange_actor, blue_actor
            )
            base_action = deterministic_hybrid_action(base_actor)
            option_actor, _ = self.option_policy(observation)
            option_action = deterministic_hybrid_action(option_actor)
            selection = self.aerial_router.select(
                observation,
                kickoff_active=self.match_views["kickoff_active"] != 0,
                match_done=self.match_views["done"] != 0,
            )
            self.rival_action.copy_(
                torch.where(selection.active[:, None], option_action, base_action)
            )
            self._option_before.copy_(observation)
            self._option_active.copy_(selection.active)
            self._option_activated.copy_(selection.activated)
            self._option_route.copy_(selection.route)

    def _record_post_physics(
        self,
        ball_velocity_before: torch.Tensor,
        match_active_before: torch.Tensor,
    ) -> None:
        index = self.trace_index
        after = self.bridge.observation()[self.batch_index, self.rival_side]
        goal_for = (
            (self._goal_scored != 0)
            & (self._scoring_team.to(torch.int64) == self.rival_side)
        )
        outcome = self.aerial_router.observe(
            self._option_before,
            after,
            active_before=self._option_active,
            goal_for_lane=goal_for,
        )
        super()._record_post_physics(ball_velocity_before, match_active_before)
        values = {
            "option_active": self._option_active,
            "option_activated": self._option_activated,
            "option_route": self._option_route,
            "option_contact": outcome.contact,
            "option_entry_contact": outcome.entry_airborne_contact,
            "option_second_contact": outcome.second_airborne_contact,
            "option_productive_contact": outcome.productive_goalward_contact,
            "option_goal": outcome.goal_within_contact_budget,
            "option_ground_failure": outcome.ball_ground_failure,
        }
        for name, value in values.items():
            self.trace[name][index].copy_(value.to(torch.int16))


def _option_summary(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    active_match = trace["match_active"] != 0
    active = (trace["option_active"] != 0) & active_match
    activated = (trace["option_activated"] != 0) & active_match
    route = trace["option_route"]
    contacts = (trace["option_contact"] != 0) & active_match
    entry = (trace["option_entry_contact"] != 0) & active_match
    second = (trace["option_second_contact"] != 0) & active_match
    productive = (trace["option_productive_contact"] != 0) & active_match
    goals = (trace["option_goal"] != 0) & active_match
    ground_failures = (trace["option_ground_failure"] != 0) & active_match
    per_route: dict[str, Any] = {}
    for route_id, name in enumerate(ROUTE_NAMES):
        route_active = active & (route == route_id)
        per_route[name] = {
            "active_ticks": int(route_active.sum()),
            "activations": int((activated & (route == route_id)).sum()),
            "contacts": int((contacts & route_active).sum()),
            "entry_airborne_contacts": int((entry & route_active).sum()),
            "second_airborne_contacts": int((second & route_active).sum()),
            "productive_goalward_contacts": int((productive & route_active).sum()),
            "goals_within_six_contacts": int((goals & route_active).sum()),
        }
    return {
        "activations": int(activated.sum()),
        "active_ticks": int(active.sum()),
        "contacts": int(contacts.sum()),
        "entry_airborne_contacts": int(entry.sum()),
        "second_airborne_contacts": int(second.sum()),
        "productive_goalward_contacts": int(productive.sum()),
        "goals_within_six_contacts": int(goals.sum()),
        "ball_ground_failures": int(ground_failures.sum()),
        "per_route": per_route,
    }


def run(args: argparse.Namespace) -> int:
    identities = {
        "blue_v23": base.sha256_file(BLUE),
        "orange_v23": base.sha256_file(ORANGE),
        "selected_v12_u0060": base.sha256_file(SELECTED),
    }
    expected = {
        "blue_v23": BLUE_SHA256,
        "orange_v23": ORANGE_SHA256,
        "selected_v12_u0060": SELECTED_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"deployment identity mismatch: {identities}")
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = V12CompositePhysicalTelemetryRunner(
        10,
        str(Path(args.collision_root).resolve()),
        BLUE,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=SEED,
        trace_ticks=REGULATION_TICKS + physical.MAXIMUM_OVERTIME_TICKS,
    )
    print("V12 deployment: running deterministic 10-match Nexto matrix", flush=True)
    timing = runner.run_ticks(REGULATION_TICKS)
    seconds = timing.seconds
    status = runner.phase_status()
    overtime_ticks = 0
    while np.any(status["done"] == 0) and overtime_ticks < physical.MAXIMUM_OVERTIME_TICKS:
        ticks = min(
            physical.OVERTIME_POLL_TICKS,
            physical.MAXIMUM_OVERTIME_TICKS - overtime_ticks,
        )
        extra = runner.run_ticks(ticks)
        seconds += extra.seconds
        overtime_ticks += ticks
        status = runner.phase_status()
    if np.any(status["done"] == 0):
        raise RuntimeError("V12 deployment evaluation exceeded overtime bound")
    trace = runner.trace_numpy()
    raw = runner.export()["raw"]
    touches_by_world = raw["touch_count"][np.arange(10), rival_side]
    overall = physical._analyze_subset(trace, np.arange(10), rival_side)
    option = _option_summary(trace)
    report = {
        "format": VERSION,
        "diagnostic_only": True,
        "policy_mutation": False,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "checkpoint": identities,
        "selection": {
            "accepted_update": 60,
            "reason": (
                "highest controlled goal-within-six fraction with preserved second-touch "
                "fraction and lowest ball-ground failure across updates 30/60/90/120"
            ),
        },
        "evaluation": {
            "opponent": "pinned Nexto",
            "action_mode": "deterministic",
            "seed": SEED,
            "matrix": "five standard kickoff layouts x both Rival sides",
            "physics_hz": physical.PHYSICS_HZ,
            "physics_ticks": int(runner.trace_index),
            "overtime_ticks": overtime_ticks,
            "seconds": seconds,
            "score": {
                "wins": int((status["winner"] == rival_side).sum()),
                "losses": int((status["winner"] != rival_side).sum()),
                "rival_goals": int(
                    np.where(
                        rival_side == 0, status["blue_score"], status["orange_score"]
                    ).sum()
                ),
                "nexto_goals": int(
                    np.where(
                        rival_side == 0, status["orange_score"], status["blue_score"]
                    ).sum()
                ),
            },
            "no_touch_worlds": int((touches_by_world == 0).sum()),
            "touches_by_world": touches_by_world.astype(int).tolist(),
            "finite": bool(
                np.isfinite(trace["action"]).all()
                and np.isfinite(trace["ball_x"]).all()
                and np.isfinite(trace["car_x"]).all()
            ),
        },
        "overall": overall,
        "option": option,
        "router_telemetry": runner.aerial_router.telemetry(),
        "baseline_v23": {
            "score": {"wins": 8, "losses": 2, "rival_goals": 159, "nexto_goals": 111},
            "touches": 687,
            "touches_per_minute": 13.74,
            "high_aerial_touches": 0,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "natural_nexto_evaluation_u0060.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
