from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_official_bundle_v1 import (
    FIELD,
    MODE_AERIAL,
    MODE_BASE,
    MODE_DEMO,
    MODE_RECOVERY,
    OfficialCapabilityRouterConfigV1,
    OfficialCapabilityRouterV1,
)


def blank(lanes: int = 1) -> torch.Tensor:
    value = torch.zeros((lanes, OBS_DIM), dtype=torch.float32)
    value[:, FIELD["self.on_ground"]] = 1.0
    value[:, FIELD["self.up.z"]] = 1.0
    value[:, FIELD["self.forward.y"]] = 1.0
    value[:, FIELD["self.boost"]] = 1.0
    return value


def select(router: OfficialCapabilityRouterV1, observation: torch.Tensor) -> int:
    result = router.select(
        observation,
        kickoff_active=torch.zeros(observation.shape[0], dtype=torch.bool),
        match_done=torch.zeros(observation.shape[0], dtype=torch.bool),
    )
    return int(result.mode[0])


def test_router_uses_base_for_ordinary_ground_play() -> None:
    router = OfficialCapabilityRouterV1(1, device="cpu")
    assert select(router, blank()) == MODE_BASE


def test_router_prioritizes_airborne_aerial_window() -> None:
    observation = blank()
    observation[:, FIELD["self.on_ground"]] = 0.0
    observation[:, FIELD["self.position.z"]] = 200.0 / 2044.0
    observation[:, FIELD["ball.position.y"]] = 1800.0 / 5120.0
    observation[:, FIELD["ball.position.z"]] = 500.0 / 2044.0
    observation[:, FIELD["relative.ball_position.y"]] = 300.0 / 5120.0
    observation[:, FIELD["relative.ball_position.z"]] = 300.0 / 2044.0
    observation[:, FIELD["relative.opponent_position.y"]] = 2500.0 / 5120.0
    config = replace(OfficialCapabilityRouterConfigV1(), automatic_aerial_enabled=True)
    assert (
        select(
            OfficialCapabilityRouterV1(1, device="cpu", config=config), observation
        )
        == MODE_AERIAL
    )


def test_router_uses_recovery_only_away_from_ball() -> None:
    observation = blank()
    observation[:, FIELD["self.on_ground"]] = 0.0
    observation[:, FIELD["self.position.z"]] = 100.0 / 2044.0
    observation[:, FIELD["self.linear_velocity.y"]] = 1000.0 / 2300.0
    observation[:, FIELD["self.linear_velocity.z"]] = -200.0 / 2300.0
    observation[:, FIELD["relative.ball_position.y"]] = 1200.0 / 5120.0
    config = replace(
        OfficialCapabilityRouterConfigV1(), automatic_recovery_enabled=True
    )
    assert (
        select(
            OfficialCapabilityRouterV1(1, device="cpu", config=config), observation
        )
        == MODE_RECOVERY
    )


def test_automatic_recovery_is_fail_closed_after_physical_regression() -> None:
    observation = blank()
    observation[:, FIELD["self.on_ground"]] = 0.0
    observation[:, FIELD["self.position.z"]] = 100.0 / 2044.0
    observation[:, FIELD["self.linear_velocity.y"]] = 1000.0 / 2300.0
    observation[:, FIELD["self.linear_velocity.z"]] = -200.0 / 2300.0
    observation[:, FIELD["relative.ball_position.y"]] = 1200.0 / 5120.0
    assert select(OfficialCapabilityRouterV1(1, device="cpu"), observation) == MODE_BASE


def test_router_uses_demo_only_in_aligned_offensive_supersonic_state() -> None:
    observation = blank()
    observation[:, FIELD["self.is_supersonic"]] = 1.0
    observation[:, FIELD["self.position.y"]] = 1000.0 / 5120.0
    observation[:, FIELD["ball.position.y"]] = 2400.0 / 5120.0
    observation[:, FIELD["relative.ball_position.y"]] = 1400.0 / 5120.0
    observation[:, FIELD["relative.opponent_position.x"]] = 100.0 / 4096.0
    observation[:, FIELD["relative.opponent_position.y"]] = 700.0 / 5120.0
    config = replace(
        OfficialCapabilityRouterConfigV1(), automatic_offensive_demo_enabled=True
    )
    assert (
        select(
            OfficialCapabilityRouterV1(1, device="cpu", config=config), observation
        )
        == MODE_DEMO
    )


def test_kickoff_resets_active_specialist() -> None:
    config = replace(
        OfficialCapabilityRouterConfigV1(),
        automatic_offensive_demo_enabled=True,
        specialist_cooldown_ticks=1,
    )
    router = OfficialCapabilityRouterV1(1, device="cpu", config=config)
    observation = blank()
    observation[:, FIELD["self.is_supersonic"]] = 1.0
    observation[:, FIELD["self.position.y"]] = 1000.0 / 5120.0
    observation[:, FIELD["ball.position.y"]] = 2400.0 / 5120.0
    observation[:, FIELD["relative.ball_position.y"]] = 1400.0 / 5120.0
    observation[:, FIELD["relative.opponent_position.y"]] = 700.0 / 5120.0
    assert select(router, observation) == MODE_DEMO
    result = router.select(
        observation,
        kickoff_active=torch.ones(1, dtype=torch.bool),
        match_done=torch.zeros(1, dtype=torch.bool),
    )
    assert int(result.mode[0]) == MODE_BASE


def test_router_rejects_bad_shape() -> None:
    router = OfficialCapabilityRouterV1(1, device="cpu")
    with pytest.raises(ValueError, match="observation shape"):
        router.select(
            torch.zeros((1, OBS_DIM - 1)),
            kickoff_active=torch.zeros(1, dtype=torch.bool),
            match_done=torch.zeros(1, dtype=torch.bool),
        )
