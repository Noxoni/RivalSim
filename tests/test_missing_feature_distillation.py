from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rivalsim.human_demo.bc_observation_bridge import (
    DegradationProfile,
    degradation_quality_mask,
    degrade_simulator_observations,
    hybrid_actor_channel_kl,
)
from rivalsim.human_demo.missing_feature_distillation import (
    build_whole_world_split,
    degrade_observations_torch,
    file_sha256,
    world_observation_batch,
)
from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_policy import Rival2PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_SHA256 = "5F20CE9FDE854A99405D53864FB1FB72F9B28FA4EC882F8D4C675DF627A16955"


def test_frozen_pre_step_distillation_config_hash() -> None:
    path = ROOT / "results/rival2/missing_feature_distillation_v1/frozen_config.json"

    assert file_sha256(path) == FROZEN_CONFIG_SHA256


def test_whole_world_split_is_deterministic_disjoint_and_complete() -> None:
    left = build_whole_world_split(
        worlds=32,
        train_worlds=24,
        validation_worlds=4,
        test_worlds=4,
        seed=17,
    )
    right = build_whole_world_split(
        worlds=32,
        train_worlds=24,
        validation_worlds=4,
        test_worlds=4,
        seed=17,
    )

    np.testing.assert_array_equal(left.train, right.train)
    np.testing.assert_array_equal(left.validation, right.validation)
    np.testing.assert_array_equal(left.test, right.test)
    joined = np.concatenate((left.train, left.validation, left.test))
    assert np.unique(joined).size == 32
    assert left.manifest["whole_world_disjoint"]
    assert left.manifest["split_manifest_sha256"] == right.manifest[
        "split_manifest_sha256"
    ]


def test_torch_degradation_matches_committed_numpy_bridge_profiles() -> None:
    source = np.random.default_rng(7).normal(size=(5, OBS_DIM)).astype(np.float32)
    tensor = torch.from_numpy(source.copy())

    for profile in (DegradationProfile.GAMEPLAY, DegradationProfile.FREEPLAY):
        expected, expected_quality = degrade_simulator_observations(
            source,
            profile=profile,
        )
        quality = torch.from_numpy(
            np.asarray(degradation_quality_mask(profile)).copy()
        )
        actual = degrade_observations_torch(tensor, quality)

        np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(quality.numpy(), expected_quality[0])


def test_world_batch_selects_complete_trajectories_before_flattening() -> None:
    observations = torch.arange(3 * 4 * 2 * OBS_DIM, dtype=torch.float32).reshape(
        3, 4, 2, OBS_DIM
    )

    selected = world_observation_batch(observations, np.asarray([3, 1]))
    expected = observations[:, [3, 1]].permute(1, 0, 2, 3).reshape(-1, OBS_DIM)

    torch.testing.assert_close(selected, expected)
    assert selected.shape == (12, OBS_DIM)


def test_exact_hybrid_kl_is_zero_for_identical_actor_outputs() -> None:
    actor = torch.randn(19, 13)

    channel = hybrid_actor_channel_kl(
        actor,
        actor.clone(),
        policy_config=Rival2PolicyConfig(),
    )

    torch.testing.assert_close(channel, torch.zeros_like(channel), atol=1e-7, rtol=0)
