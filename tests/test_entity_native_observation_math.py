"""Checks for the CPU formatting audit, not native-state/physics equivalence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rivalsim.rival2_contracts import POSITION_SCALE,ORANGE_PAD_REMAP
from rivalsim.rival2_env import Rival2TensorBridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/entity_native_observation_math_v1"


def bridge():
    data=np.load(OUT/"inputs.npz")
    excluded={"recorded_observation","recorded_action","resets","frame"}
    obj=object.__new__(Rival2TensorBridge)
    obj.num_envs=len(data["frame"]);obj.device=torch.device("cpu")
    obj.views={k:torch.from_numpy(data[k].copy()) for k in data.files if k not in excluded}
    obj.position_scale=torch.tensor(POSITION_SCALE,dtype=torch.float32)
    obj.pad_durations=torch.tensor([10.]*6+[4.]*28,dtype=torch.float32)
    obj.orange_pad_remap=torch.tensor(ORANGE_PAD_REMAP,dtype=torch.long)
    obj.blue_pad_remap=torch.arange(34)
    obj.team_signs=torch.tensor(((1,1,1),(-1,-1,1)),dtype=torch.float32)
    return obj


def test_hashes_bind_actual_source_and_data():
    for name in ("export.json","audit.json"):
        meta=json.loads((OUT/name).read_text())
        for p,h in meta["hashes"].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest().upper()==h,p


def test_shared_proxy_inputs_are_not_mislabeled_independent_measurements():
    meta=json.loads((OUT/"export.json").read_text())
    shared=meta["shared_reconstruction_indices"]
    assert len(shared)==len(set(shared))==97
    assert set(range(99,167))<=set(shared)
    assert set(range(175,182))<=set(shared)
    assert set(np.r_[35:39,74:78])<=set(shared)
    assert set(range(9)).isdisjoint(shared)
    assert set(range(167,175)).isdisjoint(shared)


def test_actual_production_builder_reproduces_stored_comparison():
    actual=bridge().observation()[:,0].numpy()
    result=np.load(OUT/"comparison.npz")
    np.testing.assert_array_equal(actual,result["rebuilt"])
    np.testing.assert_array_equal(result["recorded_action"],result["rebuilt_action"])
    assert not torch.cuda.is_initialized()


def test_raw_ball_velocity_drives_independent_observation_fields():
    obj=bridge();before=obj.observation().clone()
    obj.views["ball_vel"][0,0]+=600
    after=obj.observation()
    difference=after[0,0]-before[0,0]
    assert set(torch.nonzero(difference,as_tuple=False).flatten().tolist())=={3,90}
    torch.testing.assert_close(difference[[3,90]],torch.tensor([.1,.1]),rtol=0,atol=1e-7)
