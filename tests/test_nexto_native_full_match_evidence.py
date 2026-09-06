import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.compare_nexto_native_integration import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/nexto_native_full_match_v1"
SHORT=ROOT/"results/rival2/nexto_native_causal_v1"


def test_protocol_source_and_parent_integrity():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for p,h in protocol["sources"].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/protocol["checkpoint"])==protocol["checkpoint_sha256"]
    assert protocol["ticks"]==36000 and protocol["worlds"]==10


@pytest.mark.parametrize("name,prior", [("baseline","baseline"),("native_timing_and_table","table_and_timing")])
def test_finished_full_match_trace_preserves_exact_prefix_and_no_update(name,prior):
    result=json.loads((OUT/(name+".json")).read_text())
    data=np.load(OUT/(name+".npz")); old=np.load(SHORT/(prior+".npz"))
    assert data["actions"].shape==(36000,10,2,8)
    assert data["poses"].shape==(9000,10,9)
    np.testing.assert_array_equal(data["actions"][:2400],old["actions"])
    np.testing.assert_array_equal(data["poses"][:600],old["poses"])
    assert np.isfinite(data["actions"]).all() and np.isfinite(data["poses"]).all()
    assert sha(OUT/(name+".npz"))==result["trace_sha256"]
    assert result["protocol_sha256"]==sha(OUT/"protocol.json")
    assert result["checkpoint_unchanged"] and result["optimizer_steps"]==0
    summary=result["summary"]
    assert summary["wins"]+summary["losses"]+summary["unresolved"]==10


def test_final_summary_does_not_claim_overtime_or_learning():
    result=json.loads((OUT/"results.json").read_text())
    assert result["optimizer_steps"]==0 and not result["production_modified"]
    assert result["full_regulation"] and not result["overtime_run"]
    for name,summary in result["summaries"].items():
        assert summary==json.loads((OUT/(name+".json")).read_text())["summary"]
