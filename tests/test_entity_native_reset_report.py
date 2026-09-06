import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("reset_report_test",Path(__file__).resolve().parents[1]/"benchmarks/report_entity_native_reset_v2.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    ds=np.zeros(2,dtype=m.schema.DECISION);ticks=np.zeros(3,dtype=m.schema.TICK)
    ds["resets"]=[1,1];ds["reset_projection"][0]=1
    ds["quality"][:,m.WHEELS]=3
    ds["base_observation"][:,m.WHEELS]=1
    ds["observation"]=ds["base_observation"]
    ds["observation"][0,m.WHEELS]=0
    ticks["decision"]=[0,1,1];ticks["phase"]=[1,2,2]
    return ds,ticks


def test_projection_exact_first_only():
    ds,ticks=fixture()
    assert m.validate_projection(ds,ticks)["projected_decisions"]==1


def test_forbidden_other_field_change_rejected():
    ds,ticks=fixture();ds["observation"][0,0]=1
    with pytest.raises(AssertionError):m.validate_projection(ds,ticks)


def test_repeated_projection_rejected():
    ds,ticks=fixture();ds["reset_projection"][1]=1;ds["observation"][1,m.WHEELS]=0
    with pytest.raises(AssertionError,match="beyond"):m.validate_projection(ds,ticks)


def test_exact_quality_promotion_rejected():
    ds,ticks=fixture();ds["quality"][0,35]=0
    with pytest.raises(AssertionError):m.validate_projection(ds,ticks)


def test_projection_without_countdown_rejected():
    ds,ticks=fixture();ticks["phase"][0]=3
    with pytest.raises(AssertionError,match="countdown"):m.validate_projection(ds,ticks)
