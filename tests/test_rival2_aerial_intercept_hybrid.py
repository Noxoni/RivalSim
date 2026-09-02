from __future__ import annotations

import torch

from benchmarks.run_rival2_aerial_intercept_hybrid_v1 import promotion_verdict
from rivalsim.rival2_aerial_intercept_hybrid import (
    AerialInterceptGateConfig,
    AerialInterceptHybridController,
    aerial_intercept_eligibility,
)
from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import POSITION_SCALE


def _observation(rows: int = 2) -> torch.Tensor:
    observation = torch.zeros((rows, 182), dtype=torch.float32)
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    observation[:, FIELD["self.on_ground"]] = 1.0
    observation[:, FIELD["self.boost"]] = 1.0
    observation[:, FIELD["ball.position.z"]] = 500.0 / POSITION_SCALE[2]
    observation[:, FIELD["relative.ball_position.y"]] = 400.0 / POSITION_SCALE[1]
    observation[:, FIELD["relative.ball_position.z"]] = 483.0 / POSITION_SCALE[2]
    observation[:, FIELD["ball.linear_velocity.z"]] = 0.25
    return observation


def test_gate_uses_only_visible_physical_state() -> None:
    observation = _observation()
    result = aerial_intercept_eligibility(observation, AerialInterceptGateConfig())
    assert result.eligible.tolist() == [True, True]
    observation[1, FIELD["self.boost"]] = 0.1
    observation[0, FIELD["self.forward.y"]] = -1.0
    result = aerial_intercept_eligibility(observation, AerialInterceptGateConfig())
    assert result.eligible.tolist() == [False, False]


def test_hybrid_latches_primitive_then_teacher_and_releases_on_touch() -> None:
    observation = _observation(1)
    controller = AerialInterceptHybridController(1, device="cpu")
    base = torch.full((1, 8), -0.25)
    lifecycle = torch.zeros(1, dtype=torch.bool)
    first = controller.step(
        base, observation, kickoff_active=lifecycle, match_done=lifecycle
    )
    assert first.activated.item()
    assert first.active.item()
    assert first.primitive.item()
    assert first.action[0, 0].item() == 1.0
    assert first.action[0, 5].item() == 1.0
    for _ in range(29):
        observation[:, FIELD["self.on_ground"]] = 0.0
        step = controller.step(
            base, observation, kickoff_active=lifecycle, match_done=lifecycle
        )
    assert not step.primitive.item()
    assert step.action[0, 1].item() == 0.0
    observation[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    released = controller.step(
        base, observation, kickoff_active=lifecycle, match_done=lifecycle
    )
    assert released.released_touch.item()
    assert not released.active.item()
    assert torch.equal(released.action, base)


def test_kickoff_reset_clears_latch_without_reactivation() -> None:
    observation = _observation(1)
    controller = AerialInterceptHybridController(1, device="cpu")
    base = torch.zeros((1, 8))
    false = torch.zeros(1, dtype=torch.bool)
    controller.step(base, observation, kickoff_active=false, match_done=false)
    true = torch.ones(1, dtype=torch.bool)
    result = controller.step(base, observation, kickoff_active=true, match_done=false)
    assert result.released_reset.item()
    assert not result.active.item()
    assert not result.activated.item()
    assert torch.equal(result.action, base)


def test_airborne_intercept_bypasses_ground_launch_primitive() -> None:
    observation = _observation(1)
    observation[:, FIELD["self.on_ground"]] = 0.0
    observation[:, FIELD["self.position.z"]] = 120.0 / POSITION_SCALE[2]
    config = AerialInterceptGateConfig(
        allow_ground_launch=False,
        allow_airborne_intercept=True,
    )
    controller = AerialInterceptHybridController(1, device="cpu", config=config)
    lifecycle = torch.zeros(1, dtype=torch.bool)
    result = controller.step(
        torch.zeros((1, 8)),
        observation,
        kickoff_active=lifecycle,
        match_done=lifecycle,
    )
    assert result.activated.item()
    assert result.airborne_activated.item()
    assert not result.ground_activated.item()
    assert not result.primitive.item()
    assert torch.equal(result.action, result.eligibility.plan.action)


def test_nonfinite_observation_fails_closed() -> None:
    observation = _observation(1)
    observation[0, 0] = float("nan")
    try:
        aerial_intercept_eligibility(observation, AerialInterceptGateConfig())
    except ValueError as exc:
        assert "nonfinite" in str(exc)
    else:
        raise AssertionError("nonfinite observation did not fail closed")


def test_campaign_promotion_requires_physical_aerial_results() -> None:
    authority = {
        "validation": {
            "baseline": {"touches": 100, "touches_per_minute": 10.0},
            "promotion_gate": {
                "minimum_wins": 6,
                "maximum_losses": 4,
                "minimum_goal_differential": 0,
                "minimum_touch_fraction_of_v23": 0.8,
                "minimum_touch_rate_fraction_of_v23": 0.8,
                "minimum_high_aerial_contacts": 5,
                "minimum_high_aerial_goals": 1,
                "maximum_additional_no_touch_worlds": 0,
            },
        }
    }
    report = {
        "evaluation": {
            "score": {"wins": 6, "losses": 4, "rival_goals": 12, "nexto_goals": 10},
            "no_touch_worlds": 0,
            "finite_actions_and_observations": True,
        },
        "overall": {
            "touches": {"total": 80, "per_minute": 8.0, "high_aerial_proxy": 5},
            "scoring": {"goals_from_high_aerial_proxy": 1},
        },
        "option": {
            "per_side": {"blue": {"activations": 1}, "orange": {"activations": 1}}
        },
    }
    passed, checks = promotion_verdict(report, authority)
    assert passed and all(checks.values())
    report["overall"]["touches"]["high_aerial_proxy"] = 4
    passed, checks = promotion_verdict(report, authority)
    assert not passed
    assert not checks["high_aerial_contacts"]
