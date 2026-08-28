from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from rivalsim.human_demo.bc_observation_bridge import FieldQuality
from rivalsim.human_demo.missing_feature_distillation import file_sha256
from rivalsim.human_demo.observation_adapter_v2 import (
    FREEPLAY_NUISANCE_INDICES,
    AdapterProfile,
    HumanDemoObservationAdapterV2,
    ObservationAdapterConfig,
    adapter_objective,
    apply_native_pad_overlay,
    canonical_pad_index,
    expected_quality,
    native_pad_overlay,
)
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES, ORANGE_PAD_REMAP
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_SHA256 = "227AFE90C5678E299851C30D14F9CA914C1B05D679BA2D67440248DED30F08A1"


def test_observation_adapter_v2_frozen_config_hash() -> None:
    assert (
        file_sha256(ROOT / "results/rival2/human_demo_observation_adapter_v2/frozen_config.json")
        == FROZEN_CONFIG_SHA256
    )


def test_full_authoritative_path_is_parameter_independent_exact_bypass() -> None:
    torch.manual_seed(17)
    adapter = HumanDemoObservationAdapterV2()
    policy = Rival2ActorCritic()
    policy.eval()
    value = torch.randn(11, OBS_DIM)

    before = adapter(value, None, profile=AdapterProfile.FULL)
    actor_before, critic_before = policy(before)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_(mean=50.0, std=20.0)
    after = adapter(value, None, profile=AdapterProfile.FULL)
    actor_after, critic_after = policy(after)

    assert before is value
    assert after is value
    torch.testing.assert_close(actor_before, actor_after, atol=0, rtol=0)
    torch.testing.assert_close(critic_before, critic_after, atol=0, rtol=0)


def test_exact_and_derived_fields_are_hard_copied() -> None:
    torch.manual_seed(19)
    adapter = HumanDemoObservationAdapterV2()
    value = torch.randn(7, OBS_DIM)
    quality = expected_quality(AdapterProfile.GAMEPLAY, device=value.device)
    assert quality is not None
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.uniform_(-100.0, 100.0)

    repaired = adapter(value, quality, profile=AdapterProfile.GAMEPLAY)
    exact = quality >= int(FieldQuality.EXACT_DERIVED)

    torch.testing.assert_close(repaired[:, exact], value[:, exact], atol=0, rtol=0)
    assert torch.isfinite(repaired).all()


def test_freeplay_never_fabricates_an_opponent() -> None:
    adapter = HumanDemoObservationAdapterV2()
    value = torch.randn(5, OBS_DIM)
    quality = expected_quality(AdapterProfile.FREEPLAY, device=value.device)
    assert quality is not None
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.uniform_(-10.0, 10.0)

    repaired = adapter(value, quality, profile=AdapterProfile.FREEPLAY)

    torch.testing.assert_close(
        repaired[:, list(FREEPLAY_NUISANCE_INDICES)],
        torch.zeros(5, len(FREEPLAY_NUISANCE_INDICES)),
        atol=0,
        rtol=0,
    )


def test_quality_promotion_is_rejected() -> None:
    adapter = HumanDemoObservationAdapterV2()
    value = torch.zeros(2, OBS_DIM)
    quality = expected_quality(AdapterProfile.GAMEPLAY, device=value.device)
    assert quality is not None
    quality = quality.clone()
    unavailable = torch.nonzero(quality == int(FieldQuality.UNAVAILABLE))[0]
    quality[unavailable] = int(FieldQuality.EXACT_DIRECT)

    with np.testing.assert_raises_regex(ValueError, "promotes"):
        adapter(value, quality, profile=AdapterProfile.GAMEPLAY)


def test_objective_updates_only_adapter_and_keeps_policy_frozen() -> None:
    torch.manual_seed(23)
    adapter = HumanDemoObservationAdapterV2(
        ObservationAdapterConfig(hidden_dim=32, hidden_layers=1)
    )
    policy = Rival2ActorCritic(Rival2PolicyConfig(hidden_dim=32, hidden_layers=1))
    policy.eval()
    policy.requires_grad_(False)
    policy_before = copy.deepcopy(policy.state_dict())
    full = torch.randn(16, OBS_DIM)
    gameplay_quality = expected_quality(AdapterProfile.GAMEPLAY, device=full.device)
    freeplay_quality = expected_quality(AdapterProfile.FREEPLAY, device=full.device)
    assert gameplay_quality is not None and freeplay_quality is not None
    gameplay = full.masked_fill(gameplay_quality == 0, 0.0)
    freeplay = full.masked_fill(freeplay_quality == 0, 0.0)

    result = adapter_objective(
        adapter,
        policy,
        full,
        gameplay,
        gameplay_quality,
        freeplay,
        freeplay_quality,
        policy_config=policy.config,
        gameplay_actor_weight=1.0,
        gameplay_reconstruction_weight=1.0,
        freeplay_reconstruction_weight=0.5,
        approximate_residual_weight=0.1,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, policy_before[name], atol=0, rtol=0)


def test_adapter_initialization_and_rebuild_are_deterministic() -> None:
    torch.manual_seed(29)
    left = HumanDemoObservationAdapterV2()
    torch.manual_seed(29)
    right = HumanDemoObservationAdapterV2()

    for left_value, right_value in zip(
        left.state_dict().values(), right.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(left_value, right_value, atol=0, rtol=0)


def _pad_frame(*, team: int, position: tuple[float, float, float]) -> dict[str, object]:
    return {
        "cars": [
            {
                "team": team,
                "flags": {"is_local_human": True},
            }
        ],
        "boost_pads": [
            {
                "stable_id": "pickup:1234",
                "position": position,
                "respawn_delay": 4.0,
                "cooldown_remaining": 2.0,
                "cooldown_quality": 1,
            }
        ],
    }


def test_native_pad_position_maps_by_geometry_and_preserves_unknown_pads() -> None:
    field = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}
    physical, error = canonical_pad_index((0.0, -1024.0, 64.08))
    assert physical == 17
    assert error == 0.0
    overlay = native_pad_overlay(_pad_frame(team=0, position=(0.0, -1024.0, 64.08)))

    assert overlay.mapped_physical_indices == (17,)
    assert overlay.values[field["boost_pad.17.active"]] == 0.0
    assert overlay.values[field["boost_pad.17.cooldown"]] == 0.5
    assert overlay.supported.sum() == 2
    assert not overlay.supported[field["boost_pad.16.active"]]
    offset_physical, offset_error = canonical_pad_index((1788.0, -2302.0, 64.08))
    assert offset_physical == 15
    assert offset_error == 2.0


def test_native_pad_overlay_respects_orange_canonical_remap() -> None:
    field = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}
    physical = 17
    agent_index = ORANGE_PAD_REMAP.index(physical)
    overlay = native_pad_overlay(_pad_frame(team=1, position=(0.0, -1024.0, 64.08)))

    assert overlay.supported[field[f"boost_pad.{agent_index}.active"]]
    assert overlay.supported[field[f"boost_pad.{agent_index}.cooldown"]]


def test_native_pad_overlay_overrides_only_supported_fields() -> None:
    source = torch.randn(2, OBS_DIM)
    values = torch.zeros_like(source)
    supported = torch.zeros_like(source, dtype=torch.bool)
    supported[:, 10] = True
    values[:, 10] = 0.25

    result = apply_native_pad_overlay(source, values, supported)

    torch.testing.assert_close(result[:, 10], torch.full((2,), 0.25), atol=0, rtol=0)
    torch.testing.assert_close(result[:, :10], source[:, :10], atol=0, rtol=0)
    torch.testing.assert_close(result[:, 11:], source[:, 11:], atol=0, rtol=0)
