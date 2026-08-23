"""Measure the required CPU/GPU/RocketSim v0.1 parity corpus."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.parity import (
    axis_sign_check,
    ball_rocketsim_errors,
    ball_same_equation_errors,
    evaluate_errors,
    rocketsim_errors,
    same_equation_errors,
)
from rivalsim.parity_tolerances import ROCKETSIM_TOLERANCES, SAME_EQUATION_TOLERANCES
from rivalsim.reference.cpu_simple import CpuSimulator
from rivalsim.reference.rocketsim_oracle import RocketSimOracle, binding_metadata
from rivalsim.scenarios import HORIZONS, parity_scenarios, source_backed_jump_checks
from rivalsim.simulator import RivalSim


def run_parity(device: str = "cuda:0") -> dict[str, Any]:
    wp.init()
    scenario_results: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, float]]] = {
        "same_equation": {},
        "rocketsim": {},
    }
    all_same_pass = bool(SAME_EQUATION_TOLERANCES)
    all_oracle_pass = bool(ROCKETSIM_TOLERANCES)
    axis_sign_pass = True

    for scenario in parity_scenarios():
        cpu = CpuSimulator(scenario.initial)
        gpu = RivalSim(1, device=device, randomize=False)
        gpu.reset(scenario.initial)
        oracle = RocketSimOracle.for_scenario(scenario)
        records: list[dict[str, Any]] = []

        for tick, controls in enumerate(scenario.controls, start=1):
            cpu.set_controls(controls)
            gpu.set_controls(controls)
            oracle.set_controls(controls)
            cpu.step()
            gpu.step()
            oracle.step()

            if tick not in HORIZONS:
                continue
            cpu_state = cpu.snapshot()
            gpu_state = gpu.snapshot()
            oracle_frame = oracle.frame()
            same = same_equation_errors(gpu_state, cpu_state)
            live = rocketsim_errors(gpu_state, oracle_frame)
            if scenario.compare_ball:
                same.update(ball_same_equation_errors(gpu_state, cpu_state))
                live.update(ball_rocketsim_errors(gpu_state, oracle_frame))

            same_tolerance = SAME_EQUATION_TOLERANCES.get(tick, {})
            live_tolerance = ROCKETSIM_TOLERANCES.get(tick, {})
            same_pass, same_failures = evaluate_errors(same, same_tolerance)
            live_pass, live_failures = evaluate_errors(live, live_tolerance)
            if not SAME_EQUATION_TOLERANCES:
                same_pass = False
            if not ROCKETSIM_TOLERANCES:
                live_pass = False
            sign_pass = axis_sign_check(scenario.name, gpu_state, oracle_frame)
            all_same_pass &= same_pass
            all_oracle_pass &= live_pass
            axis_sign_pass &= sign_pass
            _accumulate(aggregate["same_equation"], tick, same)
            _accumulate(aggregate["rocketsim"], tick, live)
            records.append(
                {
                    "tick": tick,
                    "same_equation": _rounded(same),
                    "rocketsim": _rounded(live),
                    "same_equation_pass": same_pass,
                    "same_equation_failures": same_failures,
                    "rocketsim_pass": live_pass,
                    "rocketsim_failures": live_failures,
                    "axis_sign_pass": sign_pass,
                }
            )

        scenario_results.append(
            {
                "name": scenario.name,
                "family": scenario.family,
                "note": scenario.note,
                "compare_ball": scenario.compare_ball,
                "horizons": records,
            }
        )

    measurement_only = not SAME_EQUATION_TOLERANCES or not ROCKETSIM_TOLERANCES
    return {
        "schema_version": 1,
        "milestone": "v0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "measurement_only" if measurement_only else "gate_evaluation",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "warp": wp.__version__,
            "device": str(wp.get_device(device)),
        },
        "rocketsim": binding_metadata(),
        "horizons_ticks": list(HORIZONS),
        "source_backed_first_jump": source_backed_jump_checks(),
        "tolerances": {
            "same_equation": SAME_EQUATION_TOLERANCES,
            "rocketsim": ROCKETSIM_TOLERANCES,
        },
        "summary": {
            "scenario_count": len(scenario_results),
            "same_equation_pass": all_same_pass,
            "rocketsim_pass": all_oracle_pass,
            "axis_sign_pass": axis_sign_pass,
            "basic_parity_pass": all_same_pass and all_oracle_pass and axis_sign_pass,
            "aggregate_maxima_by_horizon": _rounded_nested(aggregate),
        },
        "scenarios": scenario_results,
        "limitations": [
            (
                "First-jump ground contact and wheel sticky force are source-backed; live "
                "trajectories begin immediately after the impulse in THE_VOID."
            ),
            (
                "RocketSim's Python binding does not expose time_since_boosted or is_boosting, "
                "so those fields use CPU/GPU same-equation checks only."
            ),
            (
                "THE_VOID puts an exactly motionless ball to sleep; the corpus uses a 0.001 "
                "uu/s seed velocity for non-ball scenarios."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_parity(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def _accumulate(
    target: dict[str, dict[str, float]], tick: int, errors: dict[str, float | int]
) -> None:
    horizon = target.setdefault(str(tick), {})
    for metric, value in errors.items():
        horizon[metric] = max(horizon.get(metric, 0.0), float(value))


def _rounded(value: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: int(item) if key.endswith("_mismatch") else float(f"{float(item):.9g}")
        for key, item in value.items()
    }


def _rounded_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rounded_nested(item) for key, item in value.items()}
    if isinstance(value, float):
        return float(f"{value:.9g}")
    return value


if __name__ == "__main__":
    main()
