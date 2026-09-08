"""Freeze the user's stationary-kickoff amendment before resuming accepted PPO."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.run_sustained_acquisition_v1 import OUT, EXTERNAL, CKPTS
from benchmarks.run_sustained_gameplay_v1 import text_sha
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, write_json, utc
from rivalsim.fresh_ground_30hz import scenario_hash
from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.sustained_acquisition_v1 import training_starts
from rivalsim.sustained_standing_kickoff_v1 import specification, standing_training_starts


def freeze():
    destination = OUT / "standing_kickoff_package.json"
    if destination.exists():
        raise RuntimeError("Prospective amendment already frozen; do not overwrite")
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["status"] == "stopped_at_accepted_boundary" and state["accepted_updates"] == 76
    assert (EXTERNAL / "STOP").exists() and not (EXTERNAL / "failure.json").exists()
    source = CKPTS / "plus_000076.pt"
    expected = "FBC1715C787CDD9DAFD1434D0B9DBEF9D717F9187C6858D1D5C25A3E12EF7C73"
    assert sha(source) == expected == state["latest_checkpoint"]["sha256"]
    parent = torch.load(source, map_location="cpu", weights_only=False)
    assert parent["accepted_updates"] == 76 and parent["optimizer_steps"] == 11098
    suites = ET.parse(OUT / "standing_kickoff_tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    audits, hashes = {}, {}
    for retired, label in ((False, "active"), (True, "retired")):
        old, new = training_starts(32768, retired), standing_training_starts(32768, retired)
        rows = np.flatnonzero(new.kickoff_indicator)
        native = make_standard_kickoff_state(len(rows), new.kickoff_layout[rows])
        for name in new.state.__dataclass_fields__:
            a, b = getattr(old.state, name), getattr(new.state, name)
            np.testing.assert_array_equal(b[rows], getattr(native, name))
            np.testing.assert_array_equal(a[new.kickoff_indicator == 0], b[new.kickoff_indicator == 0])
            if name != "car_vel":
                np.testing.assert_array_equal(a, b)
        for name in ("family", "focal_side", "kickoff_indicator", "kickoff_layout", "wall_aerial_variant"):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        moving = np.linalg.norm(old.state.car_vel[rows], axis=-1).max(axis=-1)
        audits[label] = dict(worlds=32768, kickoff_rows=len(rows),
            old_assisted_rows=int((moving > 0).sum()), old_max_speed=float(moving.max()),
            new_max_speed=float(np.linalg.norm(new.state.car_vel[rows], axis=-1).max()),
            standard_state_every_field_exact=True, nonkickoff_every_field_exact=True,
            only_changed_state_field="car_vel", metadata_unchanged=True,
            layouts={str(i): int((new.kickoff_layout[rows] == i).sum()) for i in range(5)})
        hashes[label] = scenario_hash(new)
    audit = dict(utc=utc(),checks=audits,optimizer_steps_during_freeze=0,
        source_unchanged=sha(source)==expected,focused_tests=sum(int(s.get("tests",0)) for s in suites),
        runtime_test="128 actual CUDA worlds: stationary native startup and forced native goal/reset, before next policy action; sustained reward/memory/GAE tests also pass")
    write_json(OUT / "standing_kickoff_audit.json", audit)
    relative = lambda p: p.relative_to(ROOT).as_posix()
    runner = "benchmarks/run_sustained_acquisition_v1.py"
    base = json.loads((OUT / "package.json").read_text())
    sources = (runner,"rivalsim/sustained_standing_kickoff_v1.py","rivalsim/static_world.py",
        "tests/test_sustained_standing_kickoff_v1.py","benchmarks/freeze_sustained_standing_kickoff_v1.py")
    package = dict(specification=specification(),base_authority_sha256=base["authority_sha256"],
        base_implementation_commit="0460643df78642a20e4864a7394b434e5e4b33c5",
        replaced_sources={runner:base["sources"][runner]},sources={p:text_sha(ROOT/p) for p in sources},
        resume=dict(path=relative(source),sha256=expected,accepted_updates=76),
        scenario_sha256=hashes,evidence={relative(OUT/p):sha(OUT/p) for p in (
            "standing_kickoff_audit.json","standing_kickoff_tests.xml")})
    write_json(destination, package)
    print(json.dumps(dict(package_sha256=sha(destination),audit=audit)))


if __name__ == "__main__":
    torch.set_num_threads(4)
    freeze()
