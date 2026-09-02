"""Prospective natural-play validation of Rival's trajectory-planned aerial option."""

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
from rivalsim.rival2_aerial_intercept_hybrid import (  # noqa: E402
    HYBRID_VERSION,
    AerialInterceptGateConfig,
    AerialInterceptHybridController,
)
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402

AUTHORITY = ROOT / "results/rival2/aerial_intercept_hybrid_v1/authority.json"
AUTHORITY_SHA256 = "8E481135217A6E93BD0EFE9BE1E0EFF3D80E4C494E0552989DE6838F20ACDFE3"
RESULTS = ROOT / "results/rival2/aerial_intercept_hybrid_v1"
BUNDLE = ROOT / "checkpoints/rival2/aerial_intercept_hybrid_v1"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"


def gate_config(authority: dict[str, Any]) -> AerialInterceptGateConfig:
    keys = asdict(AerialInterceptGateConfig())
    return AerialInterceptGateConfig(
        **{name: authority["gate"][name] for name in keys}
    )


def load_authority() -> dict[str, Any]:
    if base.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial intercept hybrid authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{HYBRID_VERSION}_AUTHORITY":
        raise RuntimeError("unexpected aerial intercept hybrid authority format")
    identities = (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["controller"],
        authority["controller"]["teacher"],
        authority["controller"]["calibration"],
        authority["diagnostic_predecessor"]["dagger_v1_result"],
        authority["validation"]["baseline"],
    )
    for identity in identities:
        path = ROOT / identity["path"]
        if base.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"bound hybrid input changed: {path}")
    if authority["integrity"]["optimizer_steps_allowed"] != 0:
        raise RuntimeError("hybrid authority unexpectedly permits optimizer steps")
    config = gate_config(authority)
    if any(authority["gate"].get(name) != value for name, value in asdict(config).items()):
        raise RuntimeError("hybrid gate authority does not match runtime config")
    return authority


class HybridPhysicalTelemetryRunner(physical.PhysicalTelemetryRunner):
    """V23 side-specialized policy with the frozen stateful aerial option."""

    orange_checkpoint = ORANGE

    def __init__(self, *args: Any, gate: AerialInterceptGateConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aerial_controller = AerialInterceptHybridController(
            self.num_worlds, device=self.device, config=gate
        )
        shape = (self.trace_ticks, self.num_worlds)
        for name in (
            "option_active",
            "option_activated",
            "option_primitive",
            "option_eligible",
        ):
            self.trace[name] = torch.empty(
                shape, dtype=torch.int16, device=self.device
            )
        zero = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._option_active = zero.clone()
        self._option_activated = zero.clone()
        self._option_primitive = zero.clone()
        self._option_eligible = zero.clone()

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation)
            orange_actor, _ = self.orange_policy(observation)
            actor = torch.where(
                (self.rival_side == 1).unsqueeze(1), orange_actor, blue_actor
            )
            base_action = deterministic_hybrid_action(actor)
            step = self.aerial_controller.step(
                base_action,
                observation,
                kickoff_active=self.match_views["kickoff_active"] != 0,
                match_done=self.match_views["done"] != 0,
            )
            self.rival_action.copy_(step.action)
            self._option_active.copy_(step.active)
            self._option_activated.copy_(step.activated)
            self._option_primitive.copy_(step.primitive & step.active)
            self._option_eligible.copy_(step.eligibility.eligible)

    def _record_post_physics(
        self,
        ball_velocity_before: torch.Tensor,
        match_active_before: torch.Tensor,
    ) -> None:
        index = self.trace_index
        super()._record_post_physics(ball_velocity_before, match_active_before)
        self.trace["option_active"][index].copy_(self._option_active.to(torch.int16))
        self.trace["option_activated"][index].copy_(
            self._option_activated.to(torch.int16)
        )
        self.trace["option_primitive"][index].copy_(
            self._option_primitive.to(torch.int16)
        )
        self.trace["option_eligible"][index].copy_(
            self._option_eligible.to(torch.int16)
        )


def option_trace_summary(
    trace: dict[str, np.ndarray], rival_side: np.ndarray
) -> dict[str, Any]:
    match_active = trace["match_active"] != 0
    option_active = (trace["option_active"] != 0) & match_active
    activated = (trace["option_activated"] != 0) & match_active
    touch = physical._rising((trace["rival_hit_raw"] != 0) & match_active)
    airborne = (trace["on_ground"] == 0) & match_active
    high = (
        touch
        & airborne
        & (trace["car_z"] >= 300.0)
        & (trace["ball_z"] >= 300.0)
    )
    option_touch = touch & option_active
    option_high = high & option_active
    per_side: dict[str, Any] = {}
    for side, name in ((0, "blue"), (1, "orange")):
        rows = rival_side == side
        per_side[name] = {
            "worlds": int(rows.sum()),
            "activations": int(activated[:, rows].sum()),
            "active_ticks": int(option_active[:, rows].sum()),
            "option_touches": int(option_touch[:, rows].sum()),
            "option_high_aerial_contacts": int(option_high[:, rows].sum()),
        }
    return {
        "activations": int(activated.sum()),
        "active_ticks": int(option_active.sum()),
        "primitive_ticks": int(
            ((trace["option_primitive"] != 0) & match_active).sum()
        ),
        "eligible_ticks": int(
            ((trace["option_eligible"] != 0) & match_active).sum()
        ),
        "option_touches": int(option_touch.sum()),
        "option_high_aerial_contacts": int(option_high.sum()),
        "per_side": per_side,
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
        >= math.ceil(
            float(baseline["touches"]) * float(gate["minimum_touch_fraction_of_v23"])
        ),
        "touch_rate": overall["touches"]["per_minute"]
        >= float(baseline["touches_per_minute"])
        * float(gate["minimum_touch_rate_fraction_of_v23"]),
        "high_aerial_contacts": overall["touches"]["high_aerial_proxy"]
        >= int(gate["minimum_high_aerial_contacts"]),
        "high_aerial_goals": overall["scoring"]["goals_from_high_aerial_proxy"]
        >= int(gate["minimum_high_aerial_goals"]),
        "no_touch_worlds": report["evaluation"]["no_touch_worlds"]
        <= int(gate["maximum_additional_no_touch_worlds"]),
        "finite": bool(report["evaluation"]["finite_actions_and_observations"]),
        "both_sides_activated": all(
            option["per_side"][name]["activations"] > 0 for name in ("blue", "orange")
        ),
    }
    return all(checks.values()), checks


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    gate = gate_config(authority)
    protected_before = {
        "blue": base.sha256_file(BLUE),
        "orange": base.sha256_file(ORANGE),
    }
    preflight = {
        "format": f"{HYBRID_VERSION}_PREFLIGHT",
        "authority_sha256": AUTHORITY_SHA256,
        "authority_committed_before_evaluation": True,
        "protected_competitive_base": protected_before,
        "gate": asdict(gate),
        "optimizer_steps": 0,
        "ppo_steps": 0,
        "reward_changes": 0,
        "verdict": "PASS",
    }
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = HybridPhysicalTelemetryRunner(
        10,
        str(Path(args.collision_root).resolve()),
        BLUE,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=int(authority["validation"]["seed"]),
        trace_ticks=REGULATION_TICKS + physical.MAXIMUM_OVERTIME_TICKS,
        gate=gate,
    )
    print("aerial hybrid: running frozen 10-match Nexto matrix", flush=True)
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
        raise RuntimeError("hybrid evaluation exceeded frozen overtime bound")

    trace = runner.trace_numpy()
    raw = runner.export()["raw"]
    touches_by_world = raw["touch_count"][np.arange(10), rival_side]
    traced_touches = physical._rising(
        (trace["rival_hit_raw"] != 0) & (trace["match_active"] != 0)
    )
    if int(touches_by_world.sum()) != int(traced_touches.sum()):
        raise RuntimeError("hybrid touch trace does not match authoritative telemetry")
    overall = physical._analyze_subset(trace, np.arange(10), rival_side)
    report = {
        "format": f"{HYBRID_VERSION}_NATURAL_EVALUATION",
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
        "option": option_trace_summary(trace, rival_side),
        "controller_telemetry": runner.aerial_controller.telemetry(),
        "protected_competitive_base_before": protected_before,
        "protected_competitive_base_after": {
            "blue": base.sha256_file(BLUE),
            "orange": base.sha256_file(ORANGE),
        },
    }
    passed, checks = promotion_verdict(report, authority)
    report["promotion_checks"] = checks
    report["verdict"] = "PASS" if passed else "FAIL"
    base.write_json(RESULTS / "natural_evaluation.json", report)

    bundle: dict[str, Any] | None = None
    if passed:
        bundle = {
            "format": f"{HYBRID_VERSION}_BUNDLE",
            "authority": {
                "path": AUTHORITY.relative_to(ROOT).as_posix(),
                "sha256": AUTHORITY_SHA256,
            },
            "controller": authority["controller"],
            "gate": authority["gate"],
            "blue": authority["protected_competitive_base"]["blue"],
            "orange": authority["protected_competitive_base"]["orange"],
            "natural_evaluation": {
                "path": "results/rival2/aerial_intercept_hybrid_v1/natural_evaluation.json",
                "sha256": base.sha256_file(RESULTS / "natural_evaluation.json"),
            },
            "policy_parameter_mutation": False,
            "optimizer_steps": 0,
            "ppo_steps": 0,
        }
        base.write_json(BUNDLE / "bundle.json", bundle)
    result = {
        "format": f"{HYBRID_VERSION}_RESULT",
        "verdict": "PASS" if passed else "FAIL",
        "authority_sha256": AUTHORITY_SHA256,
        "promotion_checks": checks,
        "bundle": bundle,
        "score": report["evaluation"]["score"],
        "touches": overall["touches"],
        "scoring": overall["scoring"],
        "option": report["option"],
        "protected_competitive_base_unchanged": report[
            "protected_competitive_base_before"
        ]
        == report["protected_competitive_base_after"],
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
