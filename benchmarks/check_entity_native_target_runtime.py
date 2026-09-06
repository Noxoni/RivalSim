"""Run with external RLBot's Python, without importing RivalSim or writing there."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/entity_native_bridge_v1"


def run():
    results = []
    torch.set_num_threads(1)
    for label in ("reference600", "finishing650"):
        manifest = json.loads((OUT / f"{label}.json").read_text())
        artifact = ROOT / manifest["artifact"]["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest().upper() == manifest["artifact"]["sha256"]
        corpus = ROOT / manifest["corpus"]["path"]
        assert hashlib.sha256(corpus.read_bytes()).hexdigest().upper() == manifest["corpus"]["sha256"]
        actor = torch.jit.load(str(artifact), map_location="cpu").eval()
        with np.load(corpus, allow_pickle=False) as archive:
            samples = {k:archive[k] for k in ("frame", "reset_before", "observation", "source_action")}
        hidden = torch.zeros(1,1,256)
        times = []
        differences = 0
        decisions = 0
        with torch.inference_mode():
            for world in range(10):
                hidden = torch.zeros_like(hidden)
                for i in range(len(samples["frame"])):
                    if samples["reset_before"][i,world]:
                        hidden = torch.zeros_like(hidden)
                    obs = torch.from_numpy(samples["observation"][i,world:world+1])
                    begin = time.perf_counter()
                    action, hidden, logits = actor(obs, hidden)
                    times.append(time.perf_counter()-begin)
                    assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
                    differences += int(not np.array_equal(action[0].numpy(), samples["source_action"][i,world]))
                    decisions += 1
        assert differences == 0
        results.append(dict(candidate=label, artifact_sha256=manifest["artifact"]["sha256"],
                            native_decisions=decisions, action_differences=differences,
                            inference_ms_p50=float(np.percentile(times,50)*1000),
                            inference_ms_p99=float(np.percentile(times,99)*1000),
                            inference_ms_max=float(max(times)*1000)))
    report = dict(python_executable=sys.executable, python_version=sys.version, torch_version=torch.__version__,
                  result=results, live_gameplay=False, external_files_changed=False, optimizer_steps=0,
                  source_repo_import_required=False,
                  timing_scope="CPU forward only, includes initial cold call; not live packet/network latency guarantee")
    path = OUT / "target_runtime.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    run()
