import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("native_report_test",ROOT/"benchmarks/report_entity_native_comparison.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_record_round_trip(tmp_path):
    x=np.zeros(3,dtype=module.DECISION)
    x["frame"]=[100,104,108]
    x["quality"][:,35:39]=3
    p=tmp_path/"records.gz"
    with gzip.open(p,"wb") as f:
        f.write(x.tobytes())
    y=module.read_records(p,module.DECISION)
    np.testing.assert_array_equal(x,y)


def test_partial_record_is_not_silently_dropped(tmp_path):
    p=tmp_path/"records.gz"
    with gzip.open(p,"wb") as f:
        f.write(b"bad")
    with pytest.raises(AssertionError,match="Partial binary"):
        module.read_records(p,module.DECISION)


def test_incomplete_gzip_is_not_clean_capture(tmp_path):
    p=tmp_path/"records.gz"
    p.write_bytes(gzip.compress(bytes(module.DECISION.itemsize))[:-8])
    with pytest.raises(EOFError):
        module.read_records(p,module.DECISION)


def test_packet_record_round_trip(tmp_path):
    import rlbot.flat as flat
    import struct
    p=tmp_path/"packets.gz"
    packet=flat.GamePacket()
    raw=packet.pack()
    with gzip.open(p,"wb") as f:
        f.write(struct.pack("<I",len(raw))+raw)
    recovered=list(module.packets(p))
    assert len(recovered)==1 and recovered[0].pack()==raw


def delivery_fixture():
    ds=np.zeros(5,dtype=module.DECISION)
    ds["frame"]=[0,4,8,12,16]
    ds["resets"]=[1,1,1,2,2]
    ds["action"][:,0]=[1,-1,1,-1,1]
    ds["observation"][1:,167:175]=ds["action"][:-1]
    return ds,ds["observation"][:,167:175].copy()


def test_delivery_uses_prior_not_current_action():
    ds,native=delivery_fixture()
    result=module.compare_delivered_history(ds,native)
    assert result["eligible_decisions"]==3
    assert result["excluded_reset_boundaries"]==2
    assert result["native_vs_previous_observation_mismatch_rows"]==0
    assert result["observation_vs_previous_issued_mismatch_rows"]==0


def test_delivery_detects_actual_mismatch():
    ds,native=delivery_fixture()
    native[2,6]=1
    result=module.compare_delivered_history(ds,native)
    assert result["native_vs_previous_observation_mismatch_rows"]==1
    assert result["per_channel_native_mismatches"]["boost"]==1
    assert result["maximum_absolute_difference"]==1


def test_delivery_excludes_gap_and_reset_without_hiding_other_mismatch():
    ds,native=delivery_fixture()
    native[0]=7;native[2]=7;native[3]=7
    ds["missed"][2]=2
    ds["observation"][4,167]=0
    result=module.compare_delivered_history(ds,native)
    assert result["eligible_decisions"]==2
    assert result["excluded_gap_decisions"]==1
    assert result["observation_vs_previous_issued_mismatch_rows"]==1
    assert result["native_vs_previous_observation_mismatch_rows"]==1
