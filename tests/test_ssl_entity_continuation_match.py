import copy

import pytest

from benchmarks.evaluate_rival2_ssl_entity_continuation_match import (
    PARENT_SHA,
    VERSION,
    authority,
    content_hash,
    original_spec,
    spec,
    validate_payload,
)


def test_fixed_200_reuses_exact_100_match_method():
    protocol = spec()
    assert protocol["candidate_accepted_updates"] == 200
    assert protocol["match_method"] == {
        k: v for k, v in original_spec().items() if k not in ("version", "checkpoints")
    }
    method = protocol["match_method"]
    assert method["matches_per_policy"] == 10
    assert method["regulation_ticks"] == 36000
    assert method["starting_layouts"] == [0, 1, 2, 3, 4] * 2
    assert method["rival_sides"] == [0] * 5 + [1] * 5
    assert not method["action_sampling"] and not method["reward_optimization"]
    assert not method["candidate_selection"] and not method["active_training_concurrency"]


def test_wrong_offset_or_lineage_is_rejected():
    payload = dict(
        format=VERSION + "_CHECKPOINT",
        accepted_updates=200,
        continuation_parent_sha256=PARENT_SHA,
        continuation_authority_sha256=content_hash(authority()),
    )
    validate_payload(payload)
    for key, value in (
        ("format", "old_hybrid"),
        ("accepted_updates", 199),
        ("continuation_parent_sha256", "another_parent"),
        ("continuation_authority_sha256", "changed_authority"),
    ):
        changed = copy.deepcopy(payload)
        changed[key] = value
        with pytest.raises(AssertionError):
            validate_payload(changed)
