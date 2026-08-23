from __future__ import annotations

import pytest

from rivalsim.parity import axis_sign_check, evaluate_errors, rocketsim_errors
from rivalsim.parity_tolerances import ROCKETSIM_TOLERANCES
from rivalsim.reference.rocketsim_oracle import RocketSimOracle
from rivalsim.scenarios import parity_scenarios
from rivalsim.simulator import RivalSim


@pytest.mark.parametrize(
    "scenario_name",
    ["stationary_gravity_drop", "boost_from_rest", "combined_air_torque", "forward_dodge"],
)
def test_selected_rocketsim_parity(scenario_name: str) -> None:
    scenario = next(item for item in parity_scenarios() if item.name == scenario_name)
    gpu = RivalSim(1, randomize=False)
    gpu.reset(scenario.initial)
    oracle = RocketSimOracle.for_scenario(scenario)
    for tick, controls in enumerate(scenario.controls, start=1):
        gpu.set_controls(controls)
        oracle.set_controls(controls)
        gpu.step()
        oracle.step()
        if tick not in (1, 30, 120):
            continue
        state = gpu.snapshot()
        frame = oracle.frame()
        errors = rocketsim_errors(state, frame)
        passed, failures = evaluate_errors(errors, ROCKETSIM_TOLERANCES[tick])
        assert passed, failures
        assert axis_sign_check(scenario.name, state, frame)
