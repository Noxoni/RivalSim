"""Physically validate the single-artifact Rival official controller.

The evaluator runs the exact deterministic five-kickoff-layout x both-side
Nexto matrix used to accept V23.  The router and every specialist are
inference-only.  No model parameter, reward, or optimizer state is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.analyze_rival2_v23_physical_telemetry import (  # noqa: E402
    MAXIMUM_OVERTIME_TICKS,
    OVERTIME_POLL_TICKS,
    PHYSICS_HZ,
    _analyze_subset,
    _PhysicalTelemetryMixin,
    _rising,
)
from benchmarks.run_rival2_codex_autonomous_v1 import sha256_file  # noqa: E402
from rivalsim.full_match import REGULATION_TICKS, FullMatchRunner  # noqa: E402
from rivalsim.rival2_official_bundle_v1 import (  # noqa: E402
    MODE_NAMES,
    OFFICIAL_BUNDLE_V1_FORMAT,
    load_official_checkpoint,
)

OFFICIAL = ROOT / "checkpoints/rival2/official_v1/rival2_official_v1.pt"
OFFICIAL_SHA256 = "20D03ECFAD8680D9F5464AEBA7C45B3FF86B3FD7FFDA50BE5160F3A4BF1EBC19"
BASE_BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
BASELINE = (
    ROOT
    / "results/rival2/capability_curriculum_v1/v23_comparable_physical_telemetry.json"
)
DEFAULT_OUTPUT = ROOT / "results/rival2/official_v1/physical_validation.json"


class OfficialPhysicalTelemetryRunner(_PhysicalTelemetryMixin, FullMatchRunner):
    """Run the official multi-policy artifact in the accepted match harness."""

    official_checkpoint = OFFICIAL
    official_sha256 = OFFICIAL_SHA256

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        observed = sha256_file(self.official_checkpoint)
        if observed != self.official_sha256:
            raise RuntimeError(
                f"official checkpoint identity changed: {observed} != {self.official_sha256}"
            )
        payload, controller = load_official_checkpoint(
            self.official_checkpoint,
            self.num_worlds,
            device=self.device,
        )
        if payload.get("format") != OFFICIAL_BUNDLE_V1_FORMAT:
            raise RuntimeError("official checkpoint format mismatch")
        if int(payload.get("physics_hz", 0)) != PHYSICS_HZ:
            raise RuntimeError("official checkpoint physics cadence mismatch")
        if int(payload.get("policy_hz", 0)) != PHYSICS_HZ:
            raise RuntimeError("official checkpoint policy cadence mismatch")
        self.official_payload = payload
        self.official_controller = controller
        self._latest_router_mode = torch.zeros(
            self.num_worlds, dtype=torch.int64, device=self.device
        )
        shape = (self.trace_ticks, self.num_worlds)
        self.trace["router_mode"] = torch.empty(
            shape, dtype=torch.int16, device=self.device
        )

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        action, selection = self.official_controller.action(
            observation,
            self.rival_side,
            kickoff_active=self.match_views["kickoff_active"] != 0,
            match_done=self.match_views["done"] != 0,
        )
        self.rival_action.copy_(action)
        self._latest_router_mode.copy_(selection.mode)

    def _record_post_physics(
        self,
        ball_velocity_before: torch.Tensor,
        match_active_before: torch.Tensor,
    ) -> None:
        index = self.trace_index
        super()._record_post_physics(ball_velocity_before, match_active_before)
        self.trace["router_mode"][index].copy_(self._latest_router_mode.to(torch.int16))


def _mode_telemetry(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    active = trace["match_active"] != 0
    modes = trace["router_mode"].astype(np.int64)
    action = trace["action"]
    result: dict[str, Any] = {}
    for mode, name in enumerate(MODE_NAMES):
        selected = active & (modes == mode)
        rising = _rising(selected)
        values = action[selected]
        result[name] = {
            "activations": int(rising.sum()),
            "active_ticks": int(selected.sum()),
            "active_seconds": float(selected.sum() / PHYSICS_HZ),
            "active_fraction": float(selected.sum() / max(int(active.sum()), 1)),
            "mean_action": (
                values.mean(axis=0, dtype=np.float64).astype(float).tolist()
                if values.size
                else None
            ),
        }
    return result


def _acceptance(
    score: dict[str, int], overall: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    baseline_touches = int(baseline["overall"]["touches"]["total"])
    touches = int(overall["touches"]["total"])
    checks = {
        "wins_at_least_6_of_10": int(score["wins"]) >= 6,
        "nonnegative_goal_differential": int(score["rival_goals"])
        >= int(score["nexto_goals"]),
        "touches_at_least_75_percent_of_v23": touches >= int(
            np.floor(0.75 * baseline_touches)
        ),
        "finite_summary": bool(
            np.isfinite(
                [
                    float(score["rival_goals"]),
                    float(score["nexto_goals"]),
                    float(overall["ground"]["horizontal_speed_uu_per_second"]["mean"]),
                    float(overall["touches"]["per_minute"]),
                ]
            ).all()
        ),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reference": {
            "checkpoint": baseline["checkpoint"],
            "score": baseline["evaluation"]["score"],
            "touches": baseline_touches,
        },
    }


def run(args: argparse.Namespace) -> int:
    official = Path(args.checkpoint).resolve()
    expected = str(args.checkpoint_sha256).upper()
    observed = sha256_file(official)
    if observed != expected:
        raise RuntimeError(f"official checkpoint SHA-256 mismatch: {observed} != {expected}")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    OfficialPhysicalTelemetryRunner.official_checkpoint = official
    OfficialPhysicalTelemetryRunner.official_sha256 = expected
    runner = OfficialPhysicalTelemetryRunner(
        10,
        str(Path(args.collision_root).resolve()),
        BASE_BLUE,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=int(args.seed),
        trace_ticks=REGULATION_TICKS + MAXIMUM_OVERTIME_TICKS,
    )
    print("official validation: running 10 deterministic regulation matches", flush=True)
    timing = runner.run_ticks(REGULATION_TICKS)
    timing_seconds = timing.seconds
    status = runner.phase_status()
    overtime_ticks_run = 0
    while np.any(status["done"] == 0) and overtime_ticks_run < MAXIMUM_OVERTIME_TICKS:
        ticks = min(OVERTIME_POLL_TICKS, MAXIMUM_OVERTIME_TICKS - overtime_ticks_run)
        print(
            "official validation: advancing "
            f"{int((status['done'] == 0).sum())} tied match(es) through overtime",
            flush=True,
        )
        overtime_timing = runner.run_ticks(ticks)
        timing_seconds += overtime_timing.seconds
        overtime_ticks_run += ticks
        status = runner.phase_status()
    if np.any(status["done"] == 0):
        raise RuntimeError("official validation exceeded the five-minute overtime bound")

    trace = runner.trace_numpy()
    exported = runner.export()
    raw = exported["raw"]
    observed_touches = int(raw["touch_count"][np.arange(10), rival_side].sum())
    traced_touches = int(
        _rising((trace["rival_hit_raw"] != 0) & (trace["match_active"] != 0)).sum()
    )
    if observed_touches != traced_touches:
        raise RuntimeError(f"touch trace mismatch: {traced_touches} != {observed_touches}")

    score = {
        "wins": int((status["winner"] == rival_side).sum()),
        "losses": int((status["winner"] != rival_side).sum()),
        "rival_goals": int(
            np.where(rival_side == 0, status["blue_score"], status["orange_score"]).sum()
        ),
        "nexto_goals": int(
            np.where(rival_side == 0, status["orange_score"], status["blue_score"]).sum()
        ),
    }
    blue_worlds = np.flatnonzero(rival_side == 0)
    orange_worlds = np.flatnonzero(rival_side == 1)
    overall = _analyze_subset(trace, np.arange(10), rival_side)
    report = {
        "format": "RIVAL2_OFFICIAL_CAPABILITY_PHYSICAL_VALIDATION_V1",
        "candidate": {
            "path": official.relative_to(ROOT).as_posix(),
            "sha256": observed,
            "format": OFFICIAL_BUNDLE_V1_FORMAT,
        },
        "evaluation": {
            "opponent": "pinned Nexto",
            "action_mode": "deterministic",
            "seed": int(args.seed),
            "matrix": "five standard kickoff layouts x both Rival sides",
            "physics_hz": PHYSICS_HZ,
            "physics_ticks": int(runner.trace_index),
            "regulation_ticks": REGULATION_TICKS,
            "overtime_ticks_run": overtime_ticks_run,
            "seconds": timing_seconds,
            "score": score,
            "touch_trace_matches_authoritative_telemetry": True,
        },
        "router": {
            "summary": runner.official_controller.router.telemetry(),
            "trace": _mode_telemetry(trace),
        },
        "overall": overall,
        "rival_as_blue": _analyze_subset(trace, blue_worlds, rival_side),
        "rival_as_orange": _analyze_subset(trace, orange_worlds, rival_side),
    }
    report["acceptance"] = _acceptance(score, overall, baseline)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["acceptance"]["verdict"] == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--seed", type=int, default=2_026_090_206)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL)
    parser.add_argument("--checkpoint-sha256", default=OFFICIAL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
