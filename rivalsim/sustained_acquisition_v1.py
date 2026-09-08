"""Temporary easier starts, not a new reward, policy or task-termination rule."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import warp as wp

from rivalsim.direct_skills_v1 import NAMES as BASE_NAMES, _rotate_and_swap
from rivalsim.direct_skills_kickoff_race_v1 import scenarios as base_scenarios
from rivalsim.ssl_foundation_v1 import SslFoundationScenarioBatch, _set_coherent_ground_route
from rivalsim.state import StateSnapshot
from rivalsim.sustained_gameplay_v1 import SustainedEnv
from rivalsim.sustained_gameplay_training_v1 import SustainedCollector
from rivalsim.ssl_entity_mixed_training import MixedEntityRolloutCollector

VERSION = "RIVAL2_SUSTAINED_ACQUISITION_CURRICULUM_V1"
SEED = 2026090802
NAMES = (*BASE_NAMES, "ball_acquisition")
FAMILY = len(BASE_NAMES)


def specification():
    return dict(version=VERSION, training_seed=SEED, probe_seed=SEED+100,
        acquisition_reset_bank_fraction=.5, preserve_other_families_relative_mix=True,
        easy="Half of acquisition rows:350..850uu away, stationary/slow rolling ball<=120uu/s, coherent heading error at most0.7rad, half standing cars otherwise0..150uu/s",
        varied="Half:850..1800uu, stationary/rolling ball<=300uu/s, heading error at most1.4rad, half standing cars otherwise0..400uu/s",
        geometry="Grounded cars and ground ball, initial center separation>=350uu, ball centrally placed, active opponent initially3000..3400uu upfield. Both focal sides. No controller prefix, frozen opponent or disabled collision.",
        reward_change=False, action_or_observation_change=False, early_touch_reset=False,
        termination="Unchanged sustained goal or45s without any physical car-ball contact; keep playing after first touch",
        probe=dict(worlds=1024,easy_seconds=5.,varied_seconds=8.,seconds=8.,
            policy="Deterministic current actor, active native_v5 Nexto at original cadence, both sides",
            success="Native focal first contact in ORIGINAL episode by deadline. Opponent contact is not focal success. Goal before focal contact is failure. Diagnostic ends after8s but does not generate training data.",
            every_additional_updates=10, initial_parent_probe=True,
            easy_success_fraction=.95,varied_success_fraction=.90,consecutive_checks=2),
        retirement="After two consecutive passing probes AND a full10-match Nexto check at that SAME checkpoint has>=5 Rival touches/min and zero matches without Rival touch. Remove only from future resets, not live episodes. Retired status persists across process resume; no automatic reinsertion.",
        limitations="Development curriculum criterion, not held-out acceptance, kickoff-winning, possession or SSL proof. Fixed bank share is not learner-frame share; report actual exposures. No guarantee scenario changes alone solve acquisition.")


def acquisition_starts(worlds, seed=SEED):
    rng = np.random.default_rng(seed)
    state = StateSnapshot.empty(worlds)
    state.on_ground.fill(1)
    state.car_pos[...,2] = 17.
    state.ball_pos[:,2] = 93.15
    # Every 4 rows cover easy/varied x blue/orange, irrespective of world parity.
    focal = (np.arange(worlds)//2 % 2).astype(np.int32)
    for row in range(worlds):
        varied = row % 2 == 1
        ball = state.ball_pos[row,:2]
        ball[:] = rng.uniform((-1000,-1000),(1000,1000))
        distance = rng.uniform(850,1800) if varied else rng.uniform(350,850)
        angle = rng.uniform(-.9,.9)
        state.car_pos[row,0,:2] = ball + distance*np.array([np.sin(angle),-np.cos(angle)])
        state.car_pos[row,1,:2] = ball + [rng.uniform(-700,700),rng.uniform(3000,3400)]
        if rng.random() < .5:
            direction = rng.uniform(-np.pi,np.pi)
            speed = rng.uniform(0,300 if varied else 120)
            state.ball_vel[row,:2] = speed*np.array([np.cos(direction),np.sin(direction)])
            state.ball_ang_vel[row,:2] = [-state.ball_vel[row,1]/93.15,state.ball_vel[row,0]/93.15]
        offset = 1.2 if varied else .5
        speed_range = (0,0) if rng.random()<.5 else (0,400 if varied else 150)
        _set_coherent_ground_route(state,row,0,rng,ball,speed_range,(-offset,offset))
        _set_coherent_ground_route(state,row,1,rng,ball,(0,250),(-1.,1.))
        state.boost[row] = rng.uniform(30,100,2)
        if focal[row]:
            _rotate_and_swap(state,row)
    state.validate()
    return SslFoundationScenarioBatch(state,np.full(worlds,FAMILY,np.int32),focal,
        np.zeros(worlds,np.int32),np.full(worlds,-1,np.int32),np.full(worlds,-1,np.int32))


def training_starts(worlds, retired=False):
    batch = base_scenarios(worlds)
    if retired:
        return batch
    rng = np.random.default_rng(SEED+1)
    selected = []
    # Stratification preserves all original start types rather than wiping one out.
    for i in range(FAMILY):
        rows = rng.permutation(np.flatnonzero(batch.family==i))
        selected.extend(rows[:len(rows)//2].tolist())
    rest = np.setdiff1d(np.arange(worlds),selected)
    selected.extend(rng.permutation(rest)[:worlds//2-len(selected)].tolist())
    rows = np.asarray(selected,dtype=np.int64)
    # Shuffle so easy/varied/focal side cannot be correlated with a base family.
    rng.shuffle(rows)
    extra = acquisition_starts(len(rows))
    for name in batch.state.__dataclass_fields__:
        getattr(batch.state,name)[rows] = getattr(extra.state,name)
    for name in ("family","focal_side","kickoff_indicator","kickoff_layout","wall_aerial_variant"):
        getattr(batch,name)[rows] = getattr(extra,name)
    return batch


class AcquisitionEnv(SustainedEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.statistics = torch.zeros((len(NAMES)*2,len(self.stat_names)),device=self.device,dtype=torch.float64)
        template=self.world.ssl_foundation_reset
        source_family=template.family.numpy()
        template.summary["counts"]={name:int((source_family==i).sum()) for i,name in enumerate(NAMES)}
        template.summary["family_names_authority"]=VERSION

    def retire_starts(self, batch):
        """Swap FUTURE source arrays in place; don't reset physics, histories or roles."""
        self._activate_torch_stream()
        template = self.world.ssl_foundation_reset
        assert len(batch.family)==self.num_envs and not (batch.family==FAMILY).any()
        for name in batch.state.__dataclass_fields__:
            target = wp.to_torch(getattr(template.state,name))
            source = torch.as_tensor(getattr(batch.state,name),device=self.device)
            target.copy_(source.reshape_as(target))
        for name in ("family","focal_side","kickoff_indicator","kickoff_layout"):
            wp.to_torch(getattr(template,name)).copy_(torch.as_tensor(getattr(batch,name),device=self.device))
        template.summary["counts"]={name:int((batch.family==i).sum()) for i,name in enumerate(NAMES)}


class AcquisitionCollector(SustainedCollector):
    def collect(self):
        self.env.statistics.zero_()
        result = MixedEntityRolloutCollector.collect(self)
        rows = self.env.statistics.cpu().tolist()
        self.last_metrics["by_start_family_and_opponent"] = {
            f"{name}_{opponent}":dict(zip(self.env.stat_names,rows[i*2+j],strict=True))
            for i,name in enumerate(NAMES) for j,opponent in enumerate(("selfplay","nexto"))}
        return result


@dataclass
class Retirement:
    streak: int = 0
    retired: bool = False
    last_probe_update: int = -1
    last_match_update: int = -1
    retired_at: int | None = None

    def observe(self, update, probe):
        if update <= self.last_probe_update:
            raise ValueError("Duplicate or out-of-order probe cannot advance retirement")
        assert probe["model_unchanged"] and probe["optimizer_steps"]==0
        passed = probe["easy"]["success_fraction"] >= .95 and probe["varied"]["success_fraction"] >= .90
        self.streak = self.streak+1 if passed else 0
        self.last_probe_update = update
        return not self.retired and self.streak>=2

    def confirm_match(self, update, summary):
        if self.retired or self.streak<2 or self.last_probe_update!=update:
            return False
        if summary["touches_per_minute"] < 5 or summary["matches_without_rival_touch"] != 0:
            return False
        self.retired,self.retired_at = True,update
        return True
