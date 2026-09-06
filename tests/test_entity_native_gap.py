import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec=importlib.util.spec_from_file_location("gap_test",Path(__file__).resolve().parents[1]/"benchmarks/diagnose_entity_native_kickoff_gap.py")
gap=importlib.util.module_from_spec(spec);spec.loader.exec_module(gap)


def test_wheel_ablation_does_not_mutate_other_fields_or_source():
    original=np.arange(182,dtype=np.float32)
    before=original.copy()
    probe=gap.wheel_probe(original,0)
    np.testing.assert_array_equal(original,before)
    rest=np.setdiff1d(np.arange(182),gap.WHEELS)
    np.testing.assert_array_equal(probe[rest],original[rest])
    assert np.all(probe[gap.WHEELS]==0)


def test_quaternion_ground_yaw():
    q=gap.quaternion(SimpleNamespace(roll=0,pitch=0,yaw=np.pi/2))
    np.testing.assert_allclose(q,[0,0,np.sqrt(.5),np.sqrt(.5)],atol=1e-7)


def test_quaternion_basis_matches_rl_rotator_convention():
    rng=np.random.default_rng(5)
    for pitch,yaw,roll in rng.uniform(-3,3,(100,3)):
        x,y,z,w=gap.quaternion(SimpleNamespace(roll=roll,pitch=pitch,yaw=yaw)).astype(float)
        forward=[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)]
        up=[2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)]
        cp,sp,cy,sy,cr,sr=np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw),np.cos(roll),np.sin(roll)
        np.testing.assert_allclose(forward,[cp*cy,cp*sy,sp],atol=3e-7)
        np.testing.assert_allclose(up,[-cy*sp*cr-sy*sr,-sy*sp*cr+cy*sr,cp*cr],atol=3e-7)
