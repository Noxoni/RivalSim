import json
from pathlib import Path

import numpy as np

from benchmarks.compare_nexto_native_integration import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/nexto_native_causal_v1"


def test_source_trace_and_checkpoint_bindings():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for p,h in protocol["sources"].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/protocol["checkpoint"])==protocol["checkpoint_sha256"]
    for name in ("baseline","table_only","timing_only","table_and_timing"):
        result=json.loads((OUT/(name+".json")).read_text())
        assert result["protocol_sha256"]==sha(OUT/"protocol.json")
        assert result["trace_sha256"]==sha(OUT/(name+".npz"))


def test_table_only_is_not_misreported_as_physical_cause():
    for a,b in (("baseline","table_only"),("timing_only","table_and_timing")):
        x,y=np.load(OUT/(a+".npz")),np.load(OUT/(b+".npz"))
        np.testing.assert_array_equal(x["poses"],y["poses"])
        changed=np.any(x["actions"]!=y["actions"],axis=(0,1,2))
        assert np.flatnonzero(changed).tolist()==[3] # yaw only


def test_all_arms_are_bounded_not_claimed_full_matches():
    summary=json.loads((OUT/"results.json").read_text())
    assert summary["optimizer_steps"]==0 and not summary["completed_regulation"]
    for result in summary["summaries"].values():
        assert result["wins"]==result["losses"]==0
        assert result["unresolved"]==10
