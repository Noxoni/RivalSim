"""V2 capture reducer plus proof of first-kickoff-only projection."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"deployment/entity_native_v2"))
sys.path.insert(0,str(ROOT/"benchmarks"))
spec=importlib.util.spec_from_file_location("reset_record_schema",ROOT/"deployment/entity_native_v2/bot.py")
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
from report_entity_native_comparison import read_records,report,report_delivery,sha

OUT=ROOT/"results/rival2/entity_native_reset_v2"
WHEELS=np.r_[35:39,74:78]


def validate_projection(ds,ticks):
    flags=ds["reset_projection"].astype(bool)
    expected=ds["base_observation"].copy()
    expected[np.ix_(flags,WHEELS)]=0
    np.testing.assert_array_equal(ds["observation"],expected)
    assert np.all(ds["quality"][:,WHEELS]==3)
    first=np.r_[True,ds["resets"][1:]!=ds["resets"][:-1]]
    assert not np.any(flags & ~first),"Projection beyond the first reset decision"
    decision_ticks=np.flatnonzero(ticks["decision"])
    assert len(decision_ticks)==len(ds)
    for row in np.flatnonzero(flags):
        tick=decision_ticks[row]
        assert ticks["phase"][tick]==schema.PHASES["Kickoff"]
        prior=0 if row==0 else decision_ticks[row-1]+1
        assert np.any(ticks["phase"][prior:tick]==schema.PHASES["Countdown"]),"Missing actual countdown"
    return dict(projected_decisions=int(flags.sum()),total_decisions=len(ds),
        all_other_fields_exact=True,unavailable_masks_preserved=True,first_kickoff_only=True)


def run(case):
    source=schema.EXTERNAL/case
    ds=read_records(source/"decisions.bin.gz",schema.DECISION)
    ticks=read_records(source/"ticks.bin.gz",schema.TICK)
    projection=validate_projection(ds,ticks)
    state=json.loads((source/"bot_state.json").read_text())
    assert projection["projected_decisions"]==state["runtime"]["reset_projection_count"]
    assert projection["projected_decisions"]>0,"Compatibility hypothesis was not exercised"
    report(case,external=schema.EXTERNAL,out=OUT,decision_dtype=schema.DECISION,tick_dtype=schema.TICK)
    report_delivery(case,out=OUT,decision_dtype=schema.DECISION)
    projection.update(decisions_sha256=sha(source/"decisions.bin.gz"),ticks_sha256=sha(source/"ticks.bin.gz"))
    (OUT/case/"projection_audit.json").write_text(json.dumps(projection,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps(projection))


if __name__=="__main__":
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument("case")
    run(p.parse_args().case)
