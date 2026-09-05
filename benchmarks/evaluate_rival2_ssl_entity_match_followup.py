"""Published fixed-boundary development match follow-up; no learning or selection.

Reuses the already validated match runner, physics, Nexto, counters and protocol.
The target is a named scheduled checkpoint, not the rolling checkpoint used to
resume training after a brief accepted-boundary pause.
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
    spec as match_spec,
)
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_continuation import (
    CHECKPOINTS,
    EXTERNAL,
    PARENT_SHA,
    RESULTS,
    VERSION,
    authority,
    content_hash,
    gpu_lease,
    text_sha,
    verify,
)

SOURCES = (
    "benchmarks/evaluate_rival2_ssl_entity_match_followup.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "rivalsim/full_match.py",
    "tests/test_ssl_entity_match_followup.py",
)
TESTS = RESULTS / "match_followup_tests.xml"


def method(target, baseline):
    assert target > baseline >= 200 and target % 50 == baseline % 50 == 0
    return dict(
        version="RIVAL2_ENTITY_SCHEDULED_MATCH_FOLLOWUP_V1",
        target=target,
        baseline=baseline,
        match_method={k: v for k, v in match_spec().items() if k not in ("version", "checkpoints")},
        selection="Named scheduled checkpoint, not best checkpoint selection",
        use="Development follow-up after short evaluation; not an untouched acceptance test",
        pause="Latest accepted boundary at/after target; preserve its exact Adam state for resume",
        rerun_baseline=False,
        automatic_training_change=False,
    )


def validate_payload(payload, target):
    assert payload["format"] == VERSION + "_CHECKPOINT"
    assert payload["accepted_updates"] == target
    assert payload["continuation_parent_sha256"] == PARENT_SHA
    assert payload["continuation_authority_sha256"] == content_hash(authority())


def prepare(target, baseline):
    verify()
    protocol_path = RESULTS / f"full_match_{target:06d}_protocol.json"
    assert not protocol_path.exists(), "Prospective protocol already exists"
    assert not (RESULTS / f"full_match_{target:06d}.json").exists(), "Already evaluated"
    prior = RESULTS / f"full_match_{baseline:06d}.json"
    evaluation = RESULTS / f"evaluation_{target:06d}.json"
    source = json.loads(prior.read_text())
    assert source["model_unchanged"] and source["checkpoint_unchanged"]
    assert source["optimizer_steps"] == 0
    prior_protocol = json.loads((RESULTS / f"full_match_{baseline:06d}_protocol.json").read_text())
    prior_method = prior_protocol.get("spec", prior_protocol.get("method"))["match_method"]
    assert prior_method == method(target, baseline)["match_method"]
    checkpoint = CHECKPOINTS / f"plus_{target:06d}.pt"
    digest = json.loads(evaluation.read_text())["checkpoint"]["sha256"]
    assert sha(checkpoint) == digest
    validate_payload(torch.load(checkpoint, map_location="cpu", weights_only=False), target)
    suites = ET.parse(TESTS).getroot().findall("testsuite")
    assert suites and all(int(s.get("errors", 0)) == int(s.get("failures", 0)) == 0 for s in suites)
    write_json(
        protocol_path,
        dict(
            method=method(target, baseline),
            checkpoint_sha256=digest,
            continuation_authority_sha256=content_hash(authority()),
            baseline_text_sha256=text_sha(prior),
            evaluation_text_sha256=text_sha(evaluation),
            sources={p: text_sha(ROOT / p) for p in SOURCES},
            tests_text_sha256=text_sha(TESTS),
        ),
    )


def run(target, baseline):
    verify()
    protocol_path = RESULTS / f"full_match_{target:06d}_protocol.json"
    output = RESULTS / f"full_match_{target:06d}.json"
    prior = RESULTS / f"full_match_{baseline:06d}.json"
    evaluation = RESULTS / f"evaluation_{target:06d}.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["method"] == method(target, baseline)
    assert protocol["continuation_authority_sha256"] == content_hash(authority())
    assert text_sha(prior) == protocol["baseline_text_sha256"]
    assert text_sha(evaluation) == protocol["evaluation_text_sha256"]
    assert text_sha(TESTS) == protocol["tests_text_sha256"]
    for name, digest in protocol["sources"].items():
        assert text_sha(ROOT / name) == digest
    remote_sha = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True
    ).split()[0]
    assert (
        remote_sha
        == subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    )
    for path in (
        *SOURCES,
        protocol_path.relative_to(ROOT).as_posix(),
        TESTS.relative_to(ROOT).as_posix(),
    ):
        remote = subprocess.check_output(["git", "show", "origin/main:" + path], cwd=ROOT)
        assert remote.replace(b"\r\n", b"\n") == (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["status"] == "stopped_at_accepted_boundary" and state["accepted_updates"] >= target
    marker = (EXTERNAL / "STOP").read_text()
    assert marker.startswith(f"AGENT_OWNED_MATCH_FOLLOWUP_{target:06d}\n")
    resume = state["latest_checkpoint"]
    assert sha(Path(resume["path"])) == resume["sha256"]
    assert not output.exists(), "Already evaluated; do not silently rerun"
    with gpu_lease():
        checkpoint = CHECKPOINTS / f"plus_{target:06d}.pt"
        digest = protocol["checkpoint_sha256"]
        assert sha(checkpoint) == digest
        validate_payload(torch.load(checkpoint, map_location="cpu", weights_only=False), target)
        runner = CandidateMatchRunner(checkpoint, digest, entity=True)
        elapsed = runner.run_ticks(REGULATION_TICKS).seconds
        for _ in range(OVERTIME_CAP_TICKS // 600):
            if bool(runner.phase_status()["done"].all()):
                break
            elapsed += runner.run_ticks(600).seconds
        raw = runner.export()["raw"]
        assert not bool(raw["goal_overflow"].any())
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        assert sha(checkpoint) == digest
        assert sha(Path(resume["path"])) == resume["sha256"]
        result = dict(
            utc=utc(),
            protocol_text_sha256=text_sha(protocol_path),
            checkpoint=runner.checkpoint_identity,
            summary=summarize(raw),
            raw={k: v.tolist() for k, v in raw.items()},
            wall_seconds=elapsed,
            hidden_resets=runner.hidden_reset_count.cpu().tolist(),
            baseline_source=prior.relative_to(ROOT).as_posix(),
            baseline_text_sha256=text_sha(prior),
            baseline_summary=json.loads(prior.read_text())["summary"],
            model_unchanged=True,
            checkpoint_unchanged=True,
            optimizer_steps=0,
            training_pause_checkpoint=resume,
            training_pause_checkpoint_unchanged=True,
        )
        write_json(output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "raw"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--baseline", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    prepare(args.target, args.baseline) if args.mode == "prepare" else run(
        args.target, args.baseline
    )
