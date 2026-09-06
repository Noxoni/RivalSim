"""Explicit first-kickoff cache projection for unavailable wheel observations.

Read-only research compatibility path. No physical wheel measurement, scripted
action, or persistent feature masking is introduced. V1 source is unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"deployment/entity_native_v1"))
from runtime import EntityPacketRuntime,sha
from packet_observation import _phase_name

VERSION="RIVAL2_ENTITY_NATIVE_RESET_CACHE_V2"
WHEEL_INDICES=np.r_[35:39,74:78]


class ResetProjectionAdapter:
    def __init__(self,inner,owner):
        self.inner,self.owner=inner,owner

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def observation(self,packet):
        observation=self.inner.observation(packet)
        owner=self.owner
        owner.base_observation=observation[0,owner.current_team].copy()
        if (owner.pending_countdown and _phase_name(packet)=="Kickoff"
                and owner.scheduler.last_decision is None):
            observation=observation.copy()
            observation[0,owner.current_team,WHEEL_INDICES]=0.0
            owner.pending_countdown=False
            owner.projected_this_packet=True
            owner.projection_count+=1
        return observation


class ResetAlignedPacketRuntime(EntityPacketRuntime):
    def __init__(self,manifest,field_info,*,model=None):
        if manifest["format"]!=VERSION:
            raise RuntimeError("Wrong reset projection runtime")
        path=(ROOT/manifest["base_manifest"]["path"]).resolve()
        path.relative_to(ROOT)
        if sha(path)!=manifest["base_manifest"]["sha256"]:
            raise RuntimeError("V1 source manifest changed")
        base=json.loads(path.read_text())
        if manifest["source"]!=base["source"] or manifest["artifact"]!=base["artifact"]:
            raise RuntimeError("Reset projection cannot change actor lineage")
        super().__init__(base,field_info,model=model)
        if not np.all(self.base_quality[WHEEL_INDICES]==3):
            raise RuntimeError("Wheel projection must remain unavailable, never exact")
        self.adapter=ResetProjectionAdapter(self.adapter,self)
        self.pending_countdown=False
        self.projected_this_packet=False
        self.projection_count=0
        self.current_team=None

    def step(self,packet,own_player_id):
        self.projected_this_packet=False
        phase=_phase_name(packet)
        frame=int(packet.match_info.frame_num)
        forward=self.last_packet_frame is None or frame>self.last_packet_frame
        own=[p for p in packet.players if int(p.player_id)==int(own_player_id)]
        if len(own)!=1:
            raise RuntimeError("Need uniquely bound local player")
        self.current_team=int(own[0].team)
        binding=tuple(int(p.player_id) for p in sorted(packet.players,key=lambda p:int(p.team)))
        rebound=self.binding is not None and binding!=self.binding
        if forward:
            if phase=="Countdown" and self.last_phase!="Countdown":
                self.pending_countdown=True
            elif phase in {"Active","Ended","Inactive"} or (rebound and phase!="Countdown"):
                self.pending_countdown=False
        action=super().step(packet,own_player_id)
        if self.last_decision is not None:
            self.last_decision["reset_projection"]=self.projected_this_packet
            self.last_decision["base_observation"]=self.base_observation
            assert np.all(self.last_decision["quality"][WHEEL_INDICES]==3)
        return action

    def summary(self):
        result=super().summary()
        result.update(format=VERSION,reset_projection_count=self.projection_count,
            wheel_projection_semantics="Unavailable fields only: simulator empty cache on first Kickoff decision after observed Countdown. Not measured zero contacts. No projection on midplay binding, ordinary pause or subsequent decisions.")
        return result
