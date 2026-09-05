"""Fixed +200 full-match follow-up, reusing +100 without rerunning it.

No optimizer, new detector, reward, physics or controller implementation.
The learner must have stopped at an accepted boundary and released its lease.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.evaluate_rival2_ssl_entity_full_match import (
    OVERTIME_CAP_TICKS,
    REGULATION_TICKS,
    CandidateMatchRunner,
    summarize,
)
from benchmarks.evaluate_rival2_ssl_entity_full_match import (
    spec as original_spec,
)
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_continuation import (
    CHECKPOINTS,
    EXTERNAL,
    PARENT_SHA,
    PILOT,
    RESULTS,
    VERSION,
    authority,
    content_hash,
    gpu_lease,
    text_sha,
    verify,
)

UPDATE = 200
PROTOCOL = RESULTS / "full_match_000200_protocol.json"
OUTPUT = RESULTS / "full_match_000200.json"
BASELINE = PILOT / "full_match_entity100.json"
SOURCES = (
    "benchmarks/evaluate_rival2_ssl_entity_continuation_match.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "rivalsim/full_match.py",
    "tests/test_ssl_entity_continuation_match.py",
)


def spec():
    return dict(
        version="RIVAL2_ENTITY_FIXED200_MATCH_FOLLOWUP_V1",
        match_method={
            k: v for k, v in original_spec().items() if k not in ("version", "checkpoints")
        },
        baseline="Reuse completed fixed+100 raw results; never rerun/select another baseline",
        baseline_checkpoint_sha256=PARENT_SHA,
        candidate_accepted_updates=UPDATE,
        candidate_path=(CHECKPOINTS / "plus_000200.pt").relative_to(ROOT).as_posix(),
        continuation_authority_sha256=content_hash(authority()),
        candidate_sha_binding="Exact SHA in evaluation200; fixed offset, not selection",
        pause="Brief accepted-boundary pause after200 scenario evaluation; same-Adam resume",
        purpose="Natural fixed-Nexto gameplay and possession evidence, not an SSL acceptance claim",
    )


def validate_payload(payload):
    assert payload["format"] == VERSION + "_CHECKPOINT"
    assert payload["accepted_updates"] == UPDATE
    assert payload["continuation_parent_sha256"] == PARENT_SHA
    assert payload["continuation_authority_sha256"] == content_hash(authority())


def prepare():
    verify()
    assert not PROTOCOL.exists(), "Protocol already frozen"
    assert not (RESULTS / "evaluation_000200.json").exists(), "Freeze before target evaluation"
    baseline = json.loads(BASELINE.read_text())
    tests = RESULTS / "full_match_000200_tests.xml"
    suites = ET.parse(tests).getroot().findall("testsuite")
    assert suites and all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
    assert baseline["checkpoint"]["sha256"] == PARENT_SHA
    assert baseline["model_unchanged"] and baseline["checkpoint_unchanged"]
    assert baseline["optimizer_steps"] == 0
    write_json(
        PROTOCOL,
        dict(
            spec=spec(),
            sources={p: text_sha(ROOT / p) for p in SOURCES},
            baseline_text_sha256=text_sha(BASELINE),
            tests_text_sha256=text_sha(tests),
        ),
    )


def run():
    verify()
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["spec"] == spec()
    for name, digest in protocol["sources"].items():
        assert text_sha(ROOT / name) == digest, name
    assert text_sha(BASELINE) == protocol["baseline_text_sha256"]
    tests = RESULTS / "full_match_000200_tests.xml"
    assert text_sha(tests) == protocol["tests_text_sha256"]
    for path in (
        *SOURCES,
        PROTOCOL.relative_to(ROOT).as_posix(),
        tests.relative_to(ROOT).as_posix(),
    ):
        remote = subprocess.check_output(["git", "show", "origin/main:" + path], cwd=ROOT)
        assert remote.replace(b"\r\n", b"\n") == (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["status"] == "stopped_at_accepted_boundary" and state["accepted_updates"] >= UPDATE
    assert (EXTERNAL / "STOP").exists(), "Explicit evaluation pause required"
    assert not OUTPUT.exists(), "Full match already complete; never silently rerun"
    with gpu_lease():
        evaluation = json.loads((RESULTS / "evaluation_000200.json").read_text())
        checkpoint = CHECKPOINTS / "plus_000200.pt"
        digest = evaluation["checkpoint"]["sha256"]
        assert sha(checkpoint) == digest
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_payload(payload)
        del payload
        runner = CandidateMatchRunner(checkpoint, digest, entity=True)
        elapsed = runner.run_ticks(REGULATION_TICKS).seconds
        for _ in range(OVERTIME_CAP_TICKS // 600):
            if bool(runner.phase_status()["done"].all()):
                break
            elapsed += runner.run_ticks(600).seconds
        raw = runner.export()["raw"]
        assert not bool(raw["goal_overflow"].any()), "goal telemetry overflow"
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        assert sha(checkpoint) == digest
        result = dict(
            utc=utc(),
            protocol_text_sha256=text_sha(PROTOCOL),
            checkpoint=runner.checkpoint_identity,
            summary=summarize(raw),
            raw={k: v.tolist() for k, v in raw.items()},
            wall_seconds=elapsed,
            hidden_resets=runner.hidden_reset_count.cpu().tolist(),
            baseline_source=BASELINE.relative_to(ROOT).as_posix(),
            baseline_text_sha256=text_sha(BASELINE),
            baseline_summary=json.loads(BASELINE.read_text())["summary"],
            model_unchanged=True,
            checkpoint_unchanged=True,
            optimizer_steps=0,
            training_pause_checkpoint=state["latest_checkpoint"],
        )
        write_json(OUTPUT, result)
        print(json.dumps({k: v for k, v in result.items() if k != "raw"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    prepare() if args.mode == "prepare" else run()
