"""Evaluate the controlled ground-to-air scorer as a gated V23 option.

The protected V23 side-specialized policies remain responsible for every
ordinary action.  A separately validated canonical option policy is consulted
only while the observation-only ground-to-air latch is active.  This stage has
no optimizer and makes no reward or simulator changes.
"""

from __future__ import annotations

import argparse
import json
import math
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
from rivalsim.rival2_ground_to_air_hybrid import (  # noqa: E402
    NaturalGroundToAirController,
    NaturalGroundToAirGateConfig,
)
from rivalsim.rival2_ground_to_air_option import (  # noqa: E402
    GroundToAirConfig,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_HYBRID_V1"
AUTHORITY = ROOT / "results/rival2/ground_to_air_hybrid_v1/authority.json"
AUTHORITY_SHA256 = "E6BA74060E8FA7ED84779F8175A757B831FD2DC6AB75F6248D65A674708138ED"
RESULTS = ROOT / "results/rival2/ground_to_air_hybrid_v1"
BUNDLE = ROOT / "checkpoints/rival2/ground_to_air_hybrid_v1"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"


def _identity_paths(authority: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["controlled_option"]["checkpoint"],
        authority["controlled_option"]["result"],
    )


def option_config(authority: dict[str, Any]) -> GroundToAirConfig:
    fields = asdict(GroundToAirConfig())
    supplied = {name: authority["option_config"][name] for name in fields}
    return GroundToAirConfig(**supplied)


def gate_config(authority: dict[str, Any]) -> NaturalGroundToAirGateConfig:
    fields = asdict(NaturalGroundToAirGateConfig())
    supplied = {name: authority["gate"][name] for name in fields}
    return NaturalGroundToAirGateConfig(**supplied)


def load_authority() -> dict[str, Any]:
    if base.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("ground-to-air hybrid authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected ground-to-air hybrid authority format")
    for identity in _identity_paths(authority):
        path = ROOT / identity["path"]
        if base.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"bound ground-to-air hybrid input changed: {path}")
    option = option_config(authority)
    gate = gate_config(authority)
    if asdict(option) != authority["option_config"]:
        raise RuntimeError("runtime ground-to-air option config differs from authority")
    if asdict(gate) != authority["gate"]:
        raise RuntimeError("runtime ground-to-air gate differs from authority")
    integrity = authority["integrity"]
    if any(
        int(integrity[name]) != 0
        for name in ("optimizer_steps_allowed", "ppo_steps_allowed", "reward_changes_allowed")
    ):
        raise RuntimeError("ground-to-air hybrid authority permits mutation")
    return authority


class GroundToAirHybridPhysicalTelemetryRunner(physical.PhysicalTelemetryRunner):
    """V23 deployment with a separately frozen, state-gated aerial policy."""

    orange_checkpoint = ORANGE

    def __init__(
        self,
        *args: Any,
        option: GroundToAirConfig,
        gate: NaturalGroundToAirGateConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        payload = torch.load(OPTION, map_location="cpu", weights_only=False)
        expected = self.checkpoint_identity["blue"]
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != expected["policy_config"]:
            raise RuntimeError("ground-to-air option architecture differs from V23")
        if payload.get("contract_hashes") != expected["contract_hashes"]:
            raise RuntimeError("ground-to-air option contract differs from V23")
        if int(payload.get("policy_hz", 0)) != self.rival_policy_hz:
            raise RuntimeError("ground-to-air option cadence differs from V23")
        self.option_policy = Rival2ActorCritic(config).to(self.device)
        self.option_policy.load_state_dict(payload["model"], strict=True)
        self.option_policy.eval()
        self.ground_to_air_controller = NaturalGroundToAirController(
            self.num_worlds,
            device=self.device,
            option_config=option,
            gate_config=gate,
        )
        shape = (self.trace_ticks, self.num_worlds)
        for name in (
            "option_active",
            "option_activated",
            "option_eligible",
            "option_pop_started",
            "option_pop_primitive",
            "option_learned_control",
        ):
            self.trace[name] = torch.empty(shape, dtype=torch.int16, device=self.device)
        zero = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._option_active = zero.clone()
        self._option_activated = zero.clone()
        self._option_eligible = zero.clone()
        self._option_pop_started = zero.clone()
        self._option_pop_primitive = zero.clone()
        self._option_learned_control = zero.clone()

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation)
            orange_actor, _ = self.orange_policy(observation)
            base_actor = torch.where((self.rival_side == 1).unsqueeze(1), orange_actor, blue_actor)
            base_action = deterministic_hybrid_action(base_actor)
            option_actor, _ = self.option_policy(observation)
            option_action = deterministic_hybrid_action(option_actor)
            step, natural_eligibility = self.ground_to_air_controller.step(
                option_action,
                observation,
                kickoff_active=self.match_views["kickoff_active"] != 0,
                match_done=self.match_views["done"] != 0,
            )
            self.rival_action.copy_(torch.where(step.active[:, None], step.action, base_action))
            self._option_active.copy_(step.active)
            self._option_activated.copy_(step.activated)
            self._option_eligible.copy_(natural_eligibility.eligible)
            self._option_pop_started.copy_(step.pop_started)
            self._option_pop_primitive.copy_(step.pop_primitive & step.active)
            self._option_learned_control.copy_(step.learned_control & step.active)

    def _record_post_physics(
        self,
        ball_velocity_before: torch.Tensor,
        match_active_before: torch.Tensor,
    ) -> None:
        index = self.trace_index
        super()._record_post_physics(ball_velocity_before, match_active_before)
        values = {
            "option_active": self._option_active,
            "option_activated": self._option_activated,
            "option_eligible": self._option_eligible,
            "option_pop_started": self._option_pop_started,
            "option_pop_primitive": self._option_pop_primitive,
            "option_learned_control": self._option_learned_control,
        }
        for name, value in values.items():
            self.trace[name][index].copy_(value.to(torch.int16))


def _rising(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    result[1:] &= ~mask[:-1]
    return result


def option_trace_summary(trace: dict[str, np.ndarray], rival_side: np.ndarray) -> dict[str, Any]:
    active_match = trace["match_active"] != 0
    active = (trace["option_active"] != 0) & active_match
    activated = (trace["option_activated"] != 0) & active_match
    pop_started = (trace["option_pop_started"] != 0) & active_match
    rival_touch = _rising((trace["rival_hit_raw"] != 0) & active_match)
    option_touch = rival_touch & active
    airborne = (trace["on_ground"] == 0) & active_match
    elevated_touch = option_touch & airborne & (trace["ball_z"] >= 250.0)
    high_touch = elevated_touch & (trace["car_z"] >= 300.0) & (trace["ball_z"] >= 300.0)
    rival_goal = (
        (trace["goal_scored"] != 0) & active_match & (trace["scoring_team"] == rival_side[None, :])
    )
    option_goal = rival_goal & active

    activation_contact_counts: list[int] = []
    over_six = 0
    for world in range(active.shape[1]):
        starts = np.flatnonzero(activated[:, world])
        for start_index, start in enumerate(starts):
            later_start = (
                starts[start_index + 1] if start_index + 1 < len(starts) else active.shape[0]
            )
            inactive_after = np.flatnonzero(~active[start:later_start, world])
            end = start + int(inactive_after[0]) if inactive_after.size else later_start
            count = int(option_touch[start:end, world].sum())
            activation_contact_counts.append(count)
            over_six += int(count > 6)

    def subset(rows: np.ndarray) -> dict[str, Any]:
        return {
            "worlds": int(rows.sum()),
            "activations": int(activated[:, rows].sum()),
            "pop_starts": int(pop_started[:, rows].sum()),
            "option_touches": int(option_touch[:, rows].sum()),
            "elevated_option_touches": int(elevated_touch[:, rows].sum()),
            "high_option_touches": int(high_touch[:, rows].sum()),
            "option_goals": int(option_goal[:, rows].sum()),
        }

    counts = np.asarray(activation_contact_counts, dtype=np.int64)
    return {
        "activations": int(activated.sum()),
        "active_ticks": int(active.sum()),
        "eligible_ticks": int(((trace["option_eligible"] != 0) & active_match).sum()),
        "pop_starts": int(pop_started.sum()),
        "pop_primitive_ticks": int(((trace["option_pop_primitive"] != 0) & active_match).sum()),
        "learned_control_ticks": int(((trace["option_learned_control"] != 0) & active_match).sum()),
        "option_touches": int(option_touch.sum()),
        "elevated_option_touches": int(elevated_touch.sum()),
        "high_option_touches": int(high_touch.sum()),
        "option_goals": int(option_goal.sum()),
        "activation_contact_counts": counts.astype(int).tolist(),
        "maximum_contacts_per_activation": int(counts.max()) if counts.size else 0,
        "activations_over_six_contacts": int(over_six),
        "per_side": {
            "blue": subset(rival_side == 0),
            "orange": subset(rival_side == 1),
        },
    }


def promotion_verdict(
    report: dict[str, Any], authority: dict[str, Any]
) -> tuple[bool, dict[str, bool]]:
    gate = authority["validation"]["promotion_gate"]
    baseline = authority["validation"]["baseline"]
    score = report["evaluation"]["score"]
    overall = report["overall"]
    option = report["option"]
    checks = {
        "wins": score["wins"] >= int(gate["minimum_wins"]),
        "losses": score["losses"] <= int(gate["maximum_losses"]),
        "goal_differential": score["rival_goals"] - score["nexto_goals"]
        >= int(gate["minimum_goal_differential"]),
        "touches": overall["touches"]["total"]
        >= math.ceil(float(baseline["touches"]) * float(gate["minimum_touch_fraction"])),
        "touch_rate": overall["touches"]["per_minute"]
        >= float(baseline["touches_per_minute"]) * float(gate["minimum_touch_rate_fraction"]),
        "no_touch_worlds": report["evaluation"]["no_touch_worlds"]
        <= int(gate["maximum_no_touch_worlds"]),
        "activations": option["activations"] >= int(gate["minimum_activations"]),
        "both_sides_activated": all(
            option["per_side"][side]["activations"] >= int(gate["minimum_activations_per_side"])
            for side in ("blue", "orange")
        ),
        "pop_starts": option["pop_starts"] >= int(gate["minimum_pop_starts"]),
        "elevated_option_touches": option["elevated_option_touches"]
        >= int(gate["minimum_elevated_option_touches"]),
        "high_option_touches": option["high_option_touches"]
        >= int(gate["minimum_high_option_touches"]),
        "option_goals": option["option_goals"] >= int(gate["minimum_option_goals"]),
        "contact_budget": option["activations_over_six_contacts"]
        <= int(gate["maximum_activations_over_six_contacts"]),
        "finite": bool(report["evaluation"]["finite_actions_and_observations"]),
    }
    return all(checks.values()), checks


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    option = option_config(authority)
    gate = gate_config(authority)
    protected_before = {
        "blue": base.sha256_file(BLUE),
        "orange": base.sha256_file(ORANGE),
        "option": base.sha256_file(OPTION),
    }
    preflight = {
        "format": f"{VERSION}_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": AUTHORITY_SHA256,
        "option_config": asdict(option),
        "gate": asdict(gate),
        "protected_before": protected_before,
        "policy_parameter_mutation": False,
        "optimizer_steps": 0,
        "ppo_steps": 0,
        "reward_changes": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = GroundToAirHybridPhysicalTelemetryRunner(
        10,
        str(Path(args.collision_root).resolve()),
        BLUE,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=int(authority["validation"]["seed"]),
        trace_ticks=REGULATION_TICKS + physical.MAXIMUM_OVERTIME_TICKS,
        option=option,
        gate=gate,
    )
    print("ground-to-air hybrid: running frozen 10-match Nexto matrix", flush=True)
    timing = runner.run_ticks(REGULATION_TICKS)
    timing_seconds = timing.seconds
    status = runner.phase_status()
    overtime_ticks = 0
    while np.any(status["done"] == 0) and overtime_ticks < physical.MAXIMUM_OVERTIME_TICKS:
        ticks = min(
            physical.OVERTIME_POLL_TICKS,
            physical.MAXIMUM_OVERTIME_TICKS - overtime_ticks,
        )
        extra = runner.run_ticks(ticks)
        timing_seconds += extra.seconds
        overtime_ticks += ticks
        status = runner.phase_status()
    if np.any(status["done"] == 0):
        raise RuntimeError("ground-to-air hybrid evaluation exceeded overtime bound")

    trace = runner.trace_numpy()
    raw = runner.export()["raw"]
    touches_by_world = raw["touch_count"][np.arange(10), rival_side]
    traced_touches = _rising((trace["rival_hit_raw"] != 0) & (trace["match_active"] != 0))
    if int(touches_by_world.sum()) != int(traced_touches.sum()):
        raise RuntimeError("ground-to-air hybrid touch trace differs from telemetry")
    overall = physical._analyze_subset(trace, np.arange(10), rival_side)
    option = option_trace_summary(trace, rival_side)
    report = {
        "format": f"{VERSION}_NATURAL_EVALUATION",
        "authority_sha256": AUTHORITY_SHA256,
        "policy_parameter_mutation": False,
        "optimizer_steps": 0,
        "ppo_steps": 0,
        "reward_changes": 0,
        "evaluation": {
            "opponent": "pinned Nexto",
            "matrix": authority["validation"]["matrix"],
            "seed": int(authority["validation"]["seed"]),
            "physics_hz": physical.PHYSICS_HZ,
            "physics_ticks": int(runner.trace_index),
            "overtime_ticks": overtime_ticks,
            "seconds": timing_seconds,
            "score": {
                "wins": int((status["winner"] == rival_side).sum()),
                "losses": int((status["winner"] != rival_side).sum()),
                "rival_goals": int(
                    np.where(rival_side == 0, status["blue_score"], status["orange_score"]).sum()
                ),
                "nexto_goals": int(
                    np.where(rival_side == 0, status["orange_score"], status["blue_score"]).sum()
                ),
            },
            "no_touch_worlds": int((touches_by_world == 0).sum()),
            "touches_by_world": touches_by_world.astype(int).tolist(),
            "finite_actions_and_observations": bool(
                np.isfinite(trace["action"]).all()
                and np.isfinite(trace["ball_x"]).all()
                and np.isfinite(trace["car_x"]).all()
            ),
            "touch_trace_matches_authoritative_telemetry": True,
        },
        "overall": overall,
        "rival_as_blue": physical._analyze_subset(
            trace, np.flatnonzero(rival_side == 0), rival_side
        ),
        "rival_as_orange": physical._analyze_subset(
            trace, np.flatnonzero(rival_side == 1), rival_side
        ),
        "option": option,
        "controller_telemetry": runner.ground_to_air_controller.telemetry(),
        "protected_before": protected_before,
        "protected_after": {
            "blue": base.sha256_file(BLUE),
            "orange": base.sha256_file(ORANGE),
            "option": base.sha256_file(OPTION),
        },
    }
    passed, checks = promotion_verdict(report, authority)
    report["promotion_checks"] = checks
    report["verdict"] = "PASS" if passed else "FAIL"
    base.write_json(RESULTS / "natural_evaluation.json", report)

    bundle: dict[str, Any] | None = None
    if passed:
        BUNDLE.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format": f"{VERSION}_BUNDLE",
            "authority": {
                "path": AUTHORITY.relative_to(ROOT).as_posix(),
                "sha256": AUTHORITY_SHA256,
            },
            "protected_competitive_base": authority["protected_competitive_base"],
            "controlled_option": authority["controlled_option"],
            "gate": authority["gate"],
            "natural_evaluation": {
                "path": "results/rival2/ground_to_air_hybrid_v1/natural_evaluation.json",
                "sha256": base.sha256_file(RESULTS / "natural_evaluation.json"),
            },
            "policy_parameter_mutation": False,
            "optimizer_steps": 0,
            "ppo_steps": 0,
        }
        base.write_json(BUNDLE / "bundle.json", bundle)
    result = {
        "format": f"{VERSION}_RESULT",
        "verdict": "PASS" if passed else "FAIL",
        "authority_sha256": AUTHORITY_SHA256,
        "promotion_checks": checks,
        "bundle": bundle,
        "score": report["evaluation"]["score"],
        "touches": overall["touches"],
        "scoring": overall["scoring"],
        "option": option,
        "protected_unchanged": report["protected_before"] == report["protected_after"],
        "stop_reason": "promotion_gate_passed" if passed else "promotion_gate_failed",
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
