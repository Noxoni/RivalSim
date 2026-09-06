"""Corrected-controller development matches; no learner or optimizer is created."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, summarize
from benchmarks.report_rival2_ssl_entity_match_followup import reduce
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from rivalsim.fresh_ground_30hz import content_hash
from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter, VERSION as CONTROLLER

VERSION = "RIVAL2_DIRECT_SKILLS_NATIVE_NEXTO_EVALUATION_V1"
OUT = ROOT / "results/rival2/direct_skills_native_nexto_v1"
MODE, NEXTO_SEED = "native_v5", 2026090693
CANDIDATES = {
    "reference600": dict(path="checkpoints/rival2/direct_skills_v1/plus_000600.pt",
        sha256="8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2", accepted_updates=600),
    "finishing650": dict(path="checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt",
        sha256="939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D", accepted_updates=650),
}
SOURCES = (
    "benchmarks/evaluate_direct_skills_native_nexto_v1.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "benchmarks/direct_skills_eval_stream.py",
    "benchmarks/report_rival2_ssl_entity_match_followup.py",
    "third_party/nexto/native_v5.py", "third_party/nexto/adapter.py",
    "rivalsim/full_match.py", "rivalsim/rival2_env.py", "rivalsim/kernels/rival2.py",
    "rivalsim/ssl_entity_policy.py", "rivalsim/ssl_joint_control_policy.py",
    "results/rival2/nexto_native_controller_v1/gpu_results.json",
    "tests/test_direct_skills_native_nexto_evaluation.py",
)


def text_sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def specification():
    return dict(version=VERSION, nexto_controller=CONTROLLER, mode=MODE, nexto_seed=NEXTO_SEED,
        candidates=CANDIDATES, matches=10, physics_hz=120, rival_policy_hz=30,
        regulation_ticks=36000, overtime_cap_ticks=14400, physics_seed=2026090573,
        starts="Existing five standard layouts, both Rival sides; real goal resets, no no-touch resets",
        rival_controls="Deterministic raw joint90 argmax, no scripted prefix or task ID",
        selection="Lexicographic: wins, aggregate goal difference, fewer concedes; exact ties prefer finishing650. Only these two preserved checkpoints; no optimization during selection.",
        purpose="Development baseline and parent choice for corrected-opponent training, not held-out acceptance or native Rocket League/SSL proof",
        legacy="No comparison against old opponent as evidence of learning. Both candidates get exactly the same corrected evaluator and seed.",
        optimizer_steps=0, reward_changes=False, architecture_changes=False)


def select_parent(results):
    if set(results) != set(CANDIDATES): raise ValueError("Both frozen candidates required")
    for row in results.values():
        if row["summary"]["unresolved"] or row["summary"]["wins"] + row["summary"]["losses"] != 10:
            raise ValueError("Only complete matches may select the development parent")
    def key(name):
        s = results[name]["summary"]
        return s["wins"], s["goals_for"] - s["goals_against"], -s["goals_against"], name == "finishing650"
    return max(CANDIDATES, key=key)


class NativeNextoMatchRunner(CandidateMatchRunner):
    def __init__(self, path, expected):
        super().__init__(path, expected, entity=True)
        self.nexto = NextoNativeV5PolicyAdapter(self.num_worlds, device=self.device,
                                               sampling_mode=MODE, seed=NEXTO_SEED)
        self.nexto.set_player_index(self.nexto_side)


def evaluate(path, expected, output, context):
    """Caller holds the GPU lease; this owns its stream and never overwrites results."""
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved["checkpoint"]["sha256"] == expected == sha(path)
        assert saved["evaluation_spec_sha256"] == content_hash(specification())
        assert saved["context"] == context
        assert all(reduce(output)["integrity"].values())
        return saved
    receipt = output.with_suffix(".started.json")
    if receipt.exists(): raise RuntimeError("Interrupted evaluation needs explicit operational audit: " + str(receipt))
    write_json(receipt, dict(utc=utc(), checkpoint=str(path), sha256=expected, context=context,
                            evaluation_spec_sha256=content_hash(specification())))
    with owned_match_stream():
        runner = NativeNextoMatchRunner(path, expected)
        nexto_before = tensor_hash(runner.nexto.actor.state_dict())
        elapsed = 0.0
        for _ in range(15):
            elapsed += runner.run_ticks(2400).seconds
            print("EVAL_PROGRESS " + json.dumps(dict(file=output.name, ticks=runner.host_tick,
                wall_seconds=elapsed)), flush=True)
        for _ in range(24):
            if bool(runner.phase_status()["done"].all()): break
            elapsed += runner.run_ticks(600).seconds
        raw = runner.export()["raw"]
        assert not raw["goal_overflow"].any()
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        assert tensor_hash(runner.nexto.actor.state_dict()) == nexto_before
        assert sha(path) == expected
        result = dict(utc=utc(), context=context, evaluation_spec_sha256=content_hash(specification()),
            checkpoint=dict(path=path.relative_to(ROOT).as_posix(), sha256=expected),
            nexto_controller=CONTROLLER, nexto_sampling_mode=MODE, nexto_seed=NEXTO_SEED,
            summary=summarize(raw), raw={k: v.tolist() for k,v in raw.items()}, wall_seconds=elapsed,
            hidden_resets=runner.hidden_reset_count.cpu().tolist(), optimizer_steps=0,
            model_unchanged=True, checkpoint_unchanged=True, nexto_unchanged=True,
            interpretation="Corrected-controller simulator development match, not native game or SSL proof")
        write_json(output, result)
        del runner; gc.collect(); torch.cuda.empty_cache()
    integrity = reduce(output)
    assert all(integrity["integrity"].values())
    write_json(output.with_suffix(".integrity.json"), integrity)
    print("EVAL_COMPLETE " + json.dumps(dict(file=output.name, summary=result["summary"])), flush=True)
    return result


def prepare():
    assert not (OUT / "baseline_protocol.json").exists()
    for item in CANDIDATES.values(): assert sha(ROOT / item["path"]) == item["sha256"]
    write_json(OUT / "baseline_protocol.json", dict(spec=specification(),
        sources={p: text_sha(ROOT / p) for p in SOURCES}))


def verify():
    protocol = json.loads((OUT / "baseline_protocol.json").read_text())
    assert protocol["spec"] == specification()
    for p,h in protocol["sources"].items(): assert text_sha(ROOT / p) == h, p
    for p in (*SOURCES, "results/rival2/direct_skills_native_nexto_v1/baseline_protocol.json"):
        stored = subprocess.check_output(["git", "show", "origin/main:" + p], cwd=ROOT)
        assert stored.replace(b"\r\n", b"\n") == (ROOT / p).read_bytes().replace(b"\r\n", b"\n"), p
    return protocol


def baseline():
    protocol = verify()
    assert not (OUT / "selection.json").exists(), "Baseline/selection already complete"
    with gpu_lease():
        results = {name: evaluate(ROOT / item["path"], item["sha256"], OUT / (name + ".json"),
            dict(kind="frozen_parent_selection", candidate=name, protocol_sha256=content_hash(protocol)))
            for name, item in CANDIDATES.items()}
    winner = select_parent(results)
    write_json(OUT / "selection.json", dict(selected_name=winner, checkpoint=CANDIDATES[winner],
        protocol_sha256=content_hash(protocol), rule=specification()["selection"], optimizer_steps=0,
        results={n: dict(summary=r["summary"], file_sha256=sha(OUT / (n+".json"))) for n,r in results.items()}))
    print("SELECTED " + winner, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "baseline"))
    torch.set_num_threads(2)
    (prepare if parser.parse_args().mode == "prepare" else baseline)()
