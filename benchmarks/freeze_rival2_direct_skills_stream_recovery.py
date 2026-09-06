"""Publish a zero-learning operational amendment; preserve the original package."""

import json
import subprocess
import xml.etree.ElementTree as ET

from benchmarks.run_rival2_direct_skills_v1 import (
    EXTERNAL,
    RESULTS,
    ROOT,
    SOURCES,
    text_sha,
    verify,
    write_json,
)


def freeze():
    old = json.loads((RESULTS / "package.json").read_text())
    assert "operational_recovery" not in old
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["accepted_updates"] == 0 and state["status"] == "failed"
    expected = subprocess.check_output(
        ["git", "show", "a1342f34:results/rival2/direct_skills_v1/package.json"], cwd=ROOT
    )
    assert json.loads(expected) == old
    for path, digest in old["sources"].items():
        if path != "benchmarks/run_rival2_direct_skills_v1.py":
            assert text_sha(ROOT / path) == digest, path
    for report in ("stream_recovery_tests.xml", "audit_tests.xml"):
        suites = ET.parse(RESULTS / report).getroot().findall("testsuite")
        assert suites and all(
            int(s.get("failures", 0)) == int(s.get("errors", 0)) == int(s.get("skipped", 0)) == 0
            for s in suites
        )
    write_json(RESULTS / "package_before_stream_recovery.json", old)
    write_json(RESULTS / "initial_runtime_failure.json", state["failure"])
    sources = {p: text_sha(ROOT / p) for p in SOURCES}
    package = dict(old)
    package["sources"] = sources
    package["operational_recovery"] = dict(
        original_commit="a1342f34a047b8bd489a31005f8a0a17cd46034c",
        accepted_updates_before_fix=0,
        authority_unchanged=True,
        original_source_changes=["benchmarks/run_rival2_direct_skills_v1.py"],
        semantics="Owned match CUDA stream and expanded evidence hash verification only",
        resume_checkpoint=state["latest_checkpoint"],
        do_not_repeat_completed_skill_baseline=True,
    )
    package["evidence"] = dict(old["evidence"])
    for name in (
        "package_before_stream_recovery.json",
        "initial_runtime_failure.json",
        "STREAM_RECOVERY.md",
        "stream_recovery_tests.xml",
        "audit_tests.xml",
        "integrity_000000.json",
    ):
        package["evidence"][name] = text_sha(RESULTS / name)
    write_json(RESULTS / "package.json", package)
    verify(published=False)


if __name__ == "__main__":
    freeze()
