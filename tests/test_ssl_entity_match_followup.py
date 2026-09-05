import pytest

from benchmarks.evaluate_rival2_ssl_entity_match_followup import (
    PARENT_SHA,
    VERSION,
    authority,
    content_hash,
    match_spec,
    method,
    validate_payload,
)


def test_same_fixed_match_protocol_no_selection_or_training():
    spec = method(250, 200)
    assert spec["match_method"] == {
        k: v for k, v in match_spec().items() if k not in ("version", "checkpoints")
    }
    assert not spec["rerun_baseline"] and not spec["automatic_training_change"]
    for target, baseline in ((249, 200), (200, 200), (200, 250), (300, 199)):
        with pytest.raises(AssertionError):
            method(target, baseline)


def test_exact_target_lineage_not_newest_or_best():
    payload = dict(
        format=VERSION + "_CHECKPOINT",
        accepted_updates=250,
        continuation_parent_sha256=PARENT_SHA,
        continuation_authority_sha256=content_hash(authority()),
    )
    validate_payload(payload, 250)
    for key, value in (
        ("accepted_updates", 251),
        ("format", "wrong"),
        ("continuation_parent_sha256", "wrong"),
        ("continuation_authority_sha256", "wrong"),
    ):
        changed = dict(payload, **{key: value})
        with pytest.raises(AssertionError):
            validate_payload(changed, 250)
