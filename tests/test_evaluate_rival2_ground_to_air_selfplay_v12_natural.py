from __future__ import annotations

import torch

from benchmarks.evaluate_rival2_ground_to_air_selfplay_v12_natural import (
    AERIAL_OPTION_CHECKPOINT_FORMATS,
    HANDOFF_FEATURE_NAMES,
    RouteFeatureMoments,
    handoff_features,
)
from rivalsim.rival2_aerial_option import FIELD


def test_integrated_v17_checkpoint_format_is_read_only_compatible() -> None:
    assert (
        "RIVAL2_GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_CHECKPOINT"
        in AERIAL_OPTION_CHECKPOINT_FORMATS
    )


def test_handoff_features_include_physical_state_and_exact_action() -> None:
    observation = torch.zeros((2, 182), dtype=torch.float32)
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    observation[:, FIELD["ball.position.y"]] = 0.25
    observation[:, FIELD["ball.position.z"]] = 0.5
    observation[:, FIELD["relative.ball_position.y"]] = 0.1
    observation[:, FIELD["self.boost"]] = torch.tensor((0.25, 0.75))
    action = torch.tensor(
        ((1.0, -0.5, 0.25, 0.0, -0.25, 1.0, 1.0, 0.0),) * 2,
        dtype=torch.float32,
    )

    features = handoff_features(observation, action)
    lookup = {name: index for index, name in enumerate(HANDOFF_FEATURE_NAMES)}

    assert features.shape == (2, len(HANDOFF_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    assert torch.equal(features[:, lookup["boost_fraction"]], torch.tensor((0.25, 0.75)))
    assert torch.equal(features[:, lookup["action_throttle"]], torch.ones(2))
    assert torch.equal(features[:, lookup["action_steer"]], torch.full((2,), -0.5))


def test_route_feature_moments_are_masked_and_route_local() -> None:
    feature_count = len(HANDOFF_FEATURE_NAMES)
    values = torch.zeros((3, feature_count), dtype=torch.float32)
    values[0, 0] = 2.0
    values[1, 0] = 6.0
    values[2, 0] = 100.0
    accumulator = RouteFeatureMoments(device=torch.device("cpu"))
    accumulator.add(
        values,
        mask=torch.tensor((True, True, False)),
        route=torch.tensor((2, 2, 1), dtype=torch.int64),
    )

    result = accumulator.export()

    assert result["rising_double_jump"]["count"] == 2
    first = result["rising_double_jump"]["features"][HANDOFF_FEATURE_NAMES[0]]
    assert first["mean"] == 4.0
    assert first["standard_deviation"] == 2.0
    assert result["soft_incoming_chip"]["count"] == 0
