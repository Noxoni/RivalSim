from __future__ import annotations

import gzip
import json
from pathlib import Path

from benchmarks.build_v022_deep_trace_cache import (
    _candidate_pairs,
    _explicit_case_selection,
    _pilot_failure_selection,
)


def test_candidate_pairs_cover_union_without_duplicates() -> None:
    payload = b"\n".join(
        (
            b'{"record":"state","tick":1}',
            b'{"record":"bvh_traversal","tick":1,"world_body_index":10,"faces":[4,3,4]}',
            b'{"record":"bvh_traversal","tick":2,"world_body_index":11,"faces":[2]}',
        )
    )
    assert _candidate_pairs(payload) == ((10, 3), (10, 4), (11, 2))


def test_pilot_failure_selection_is_ordered_and_hashed(tmp_path: Path) -> None:
    run = {"selection_sha256": "selection", "corpus_sha256": "corpus"}
    aggregate = {"counts": {"failed_cases": 2}}
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (tmp_path / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    chunks = (
        (
            "chunk-00000-000000-000002.json.gz",
            [
                {"case_id": "A", "pass": False},
                {"case_id": "B", "pass": True},
            ],
        ),
        (
            "chunk-00001-000002-000003.json.gz",
            [{"case_id": "C", "pass": False}],
        ),
    )
    for name, records in chunks:
        with gzip.open(tmp_path / name, "wt", encoding="utf-8") as stream:
            json.dump({"records": records}, stream)

    source, failed = _pilot_failure_selection(tmp_path)
    assert failed == ("A", "C")
    assert source["selection_sha256"] == "selection"
    assert source["corpus_sha256"] == "corpus"
    assert source["reported_failed_case_count"] == 2
    assert len(source["chunks"]) == 2
    assert all(item["sha256"] for item in source["chunks"])


def test_explicit_case_selection_is_ordered_and_authority_bound() -> None:
    identity = {
        "identity_inputs": {
            "corpus": {
                "corpus_sha256": "corpus",
                "generator_source_sha256": "source",
                "generator_config_sha256": "config",
                "seed": 42,
            }
        }
    }
    source, selected = _explicit_case_selection(["B", "A"], identity)
    assert selected == ("B", "A")
    assert source["case_ids"] == ["B", "A"]
    assert source["corpus_sha256"] == "corpus"
    assert source["generator_source_sha256"] == "source"
    assert source["generator_config_sha256"] == "config"
    assert source["seed"] == 42
    assert source["case_ids_sha256"]
