import numpy as np
import pytest

from benchmarks.diagnose_direct_skills_shooting_000650 import MODES, SEED, authority, bank_for
from rivalsim.direct_skills_v1 import scenarios
from rivalsim.fresh_ground_30hz import scenario_hash


def test_established_case_is_exact_published_finishing_bank():
    assert scenario_hash(bank_for('established_keeper')) == scenario_hash(scenarios(64,SEED,family_only=2))


@pytest.mark.parametrize('mode',MODES)
def test_rebuild_and_physical_validity(mode):
    a,b=bank_for(mode),bank_for(mode)
    assert scenario_hash(a)==scenario_hash(b)
    a.state.validate()
    assert (a.state.on_ground==1).all()
    assert (np.abs(a.state.car_pos[...,:2]) < np.array([4096,5120])).all()
    assert (np.linalg.norm(a.state.car_pos-a.state.ball_pos[:,None],axis=-1)>180).all()
    assert (np.linalg.norm(a.state.car_pos[:,0]-a.state.car_pos[:,1],axis=-1)>180).all()


@pytest.mark.parametrize('mode',MODES[1:])
def test_only_opponent_position_momentum_heading_change(mode):
    old,new=bank_for(MODES[0]),bank_for(mode)
    assert (old.focal_side==new.focal_side).all()
    rows=np.arange(64);side=old.focal_side
    for name in old.state.__dataclass_fields__:
        a,b=getattr(old.state,name),getattr(new.state,name)
        if name in ('car_pos','car_vel','car_quat'):
            assert np.array_equal(a[rows,side],b[rows,side]),name
        else:
            assert np.array_equal(a,b),name
    opponent=1-side;sign=np.where(side==0,1,-1)
    p=new.state.car_pos[rows,opponent]
    if mode=='initial_open_net':
        assert (p[:,1]*sign==-3500).all()
        assert (new.state.car_vel[rows,opponent]==0).all()
    else:
        assert np.allclose(p[:,1]*sign,new.state.ball_pos[:,1]*sign-1300)
        assert np.allclose(np.linalg.norm(new.state.car_vel[rows,opponent],axis=-1),600)


def test_fixed_scope_not_reward_or_learning_change():
    a=authority()
    assert a['optimizer_steps']==0 and a['worlds_per_mode']==64 and a['horizon_seconds']==12
    assert a['modes']==list(MODES)
    assert 'active pinnedNexto' in a['scope']
    with pytest.raises(ValueError):
        bank_for('unfrozen_variant')
