"""Versioned native-v5 Nexto control scheduler; legacy adapter remains reproducible.

Adapted from VirxEC/NectoFamily 0bdb6b49072f6f3829319e68bd6210a0ca4b24a2
and Rolv-Arild/Necto. CC BY-NC-SA 4.0, see LICENSE and PROVENANCE.json.
This fixes controller timing/table semantics, not all simulator/native physics
or observation differences. Worlds must receive one call per active physics tick.
"""
from __future__ import annotations

import torch

from third_party.nexto.adapter import (
    MODEL_SHA256, NextoObservation, NextoPolicyAdapter, build_nexto_observation,
)

VERSION = "NEXTO_NATIVE_V5_CONTROLLER_V1"
SAMPLING_MODES = ("native_v5", "deterministic_argmax")


def build_native_kickoff_sequence(device):
    # Exact upstream literal, including neutral yaw when only steer is specified.
    groups = (
        (44, [1, 0, 0, 0, 0, 0, 1, 0]),
        (16, [1, -1, 0, 0, 0, 0, 1, 0]),
        (8, [1, 0, 0, 0, 0, 1, 1, 0]),
        (4, [1, 0, 0, 0, 0, 0, 1, 0]),
        (4, [1, 0, -.7, .8, 0, 1, 1, 0]),
        (52, [1, 0, 1, 0, 0, 0, 1, 0]),
        (40, [1, 0, .5, 0, 1, 0, 0, 0]),
    )
    return torch.tensor([row for count,row in groups for _ in range(count)],
                        dtype=torch.float32, device=device)


class NativeNextoController:
    """Device-resident per-world controller state, independently testable on CPU.

    Inactive masks suspend a world without fabricating physics ticks. Reassignment
    must call activate; an ordinary kickoff notification resets script admission
    only, never the neural clock or pending action. No native countdown is invented.
    """
    STATE_NAMES = ("pending_action", "emitted_control", "ticks", "update_action",
                   "kickoff_index", "player_index")

    def __init__(self, worlds, device):
        if worlds < 1:
            raise ValueError("positive world count required")
        self.num_worlds, self.device = int(worlds), torch.device(device)
        self.pending_action = torch.zeros(worlds,8,device=self.device)
        self.emitted_control = torch.zeros_like(self.pending_action)
        self.ticks = torch.full((worlds,),8,dtype=torch.long,device=self.device)
        self.update_action = torch.ones(worlds,dtype=torch.bool,device=self.device)
        self.kickoff_index = torch.full((worlds,),-1,dtype=torch.long,device=self.device)
        self.player_index = torch.zeros(worlds,dtype=torch.long,device=self.device)
        self.kickoff_sequence = build_native_kickoff_sequence(self.device)
        self.rows = torch.arange(worlds,device=self.device)

    def _mask(self, value):
        if value.shape != (self.num_worlds,) or value.device != self.device or value.dtype != torch.bool:
            raise ValueError("world mask must be bool, correctly shaped and on the controller device")
        return value

    def set_player_index(self, value):
        if value.shape != self.player_index.shape or value.device != self.device or value.dtype != torch.long:
            raise ValueError("player indices must be device int64 world indices")
        if not bool(((value==0)|(value==1)).all()):
            raise ValueError("only standard two-car 1v1 is supported")
        self.player_index.copy_(value)

    def activate(self, mask):
        mask=self._mask(mask)
        self.pending_action.masked_fill_(mask[:,None],0)
        self.emitted_control.masked_fill_(mask[:,None],0)
        self.ticks.masked_fill_(mask,8)
        self.update_action.masked_fill_(mask,True)
        self.kickoff_index.masked_fill_(mask,-1)

    def notify_kickoff(self, mask=None):
        if mask is None:
            mask=torch.ones(self.num_worlds,dtype=torch.bool,device=self.device)
        self.kickoff_index.masked_fill_(self._mask(mask),-1)

    @torch.inference_mode()
    def step(self, state, kickoff_active, predict, active_mask=None):
        kickoff=self._mask(kickoff_active)
        active=(torch.ones_like(kickoff) if active_mask is None else self._mask(active_mask))
        self.ticks.add_(active.long())
        compute=active & self.update_action
        indices=None
        if bool(compute.any()):
            action,indices=predict(compute,kickoff)
            if action.shape != self.pending_action.shape or action.device != self.device:
                raise ValueError("predictor must return device [world,8] actions")
            if not bool(torch.isfinite(action[compute]).all()):
                raise RuntimeError("nonfinite Nexto action")
            self.pending_action.copy_(torch.where(compute[:,None],action,self.pending_action))
        self.update_action.masked_fill_(compute,False)
        emit=active & (self.ticks>=7)
        self.emitted_control.copy_(torch.where(emit[:,None],self.pending_action,self.emitted_control))
        period=active & (self.ticks>=8)
        self.ticks.masked_fill_(period,0)
        self.update_action.masked_fill_(period,True)

        # Native maybe_do_kickoff increments an admitted script before use.
        # New admission begins at index zero; -2 means not this kickoff's taker.
        in_phase=active & kickoff
        continuing=in_phase & (self.kickoff_index>=0)
        fresh=in_phase & (self.kickoff_index==-1)
        self.kickoff_index.add_(continuing.long())
        positions=state.car_pos[:,:,:2].to(torch.float64)
        ball=state.ball_pos[:,:2].to(torch.float64)
        distance=torch.linalg.vector_norm(positions-ball[:,None,:],dim=-1)
        own=distance[self.rows,self.player_index]
        eligible=(own-distance.min(dim=1).values).abs()<=10
        admit=torch.where(eligible,torch.zeros_like(self.kickoff_index),torch.full_like(self.kickoff_index,-2))
        self.kickoff_index.copy_(torch.where(fresh,admit,self.kickoff_index))
        self.kickoff_index.masked_fill_(active & ~kickoff,-1)
        scripted=in_phase & (self.kickoff_index>=0) & (self.kickoff_index<168) & (state.ball_pos[:,1]==0)
        script=self.kickoff_sequence[self.kickoff_index.clamp(0,167)]
        self.pending_action.copy_(torch.where(scripted[:,None],script,self.pending_action))
        self.emitted_control.copy_(torch.where(scripted[:,None],script,self.emitted_control))
        return self.emitted_control,indices

    def state_dict(self):
        return {"version":VERSION,"num_worlds":self.num_worlds,
                "tensors":{k:getattr(self,k).detach().clone() for k in self.STATE_NAMES}}

    def validate_state(self, state):
        if state.get("version")!=VERSION or state.get("num_worlds")!=self.num_worlds:
            raise ValueError("wrong native Nexto controller state identity")
        if set(state["tensors"])!=set(self.STATE_NAMES):
            raise ValueError("incomplete native Nexto controller state")
        values={}
        for name in self.STATE_NAMES:
            old,value=getattr(self,name),state["tensors"][name]
            if value.shape!=old.shape or value.dtype!=old.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid controller tensor: "+name)
            values[name]=value.to(self.device)
        if not bool(((values["ticks"]>=0)&(values["ticks"]<=8)).all()):
            raise ValueError("invalid native clock")
        if not bool(((values["player_index"]==0)|(values["player_index"]==1)).all()):
            raise ValueError("invalid player index")
        if not bool((values["kickoff_index"]>=-2).all()):
            raise ValueError("invalid kickoff index")
        if not all(bool((values[k].abs()<=1).all()) for k in ("pending_action","emitted_control")):
            raise ValueError("out-of-range controller action")
        return values

    def load_state_dict(self,state):
        values=self.validate_state(state) # Validate completely before mutation.
        for name,value in values.items():getattr(self,name).copy_(value)


class NextoNativeV5PolicyAdapter(NextoPolicyAdapter):
    """Production-capable opt-in controller correction around the pinned CUDA model.

    Explicit mode prevents silently changing deterministic benchmark RNG semantics.
    native_v5: beta=1 ordinary argmax, beta=.5 kickoff softmax categorical.
    deterministic_argmax: declared diagnostic variant, not native kickoff RNG.
    Model/observation math is inherited, not claimed independently native-exact.
    """
    def __init__(self,num_worlds,*,device="cuda:0",sampling_mode,seed,model_path=None):
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError("explicit native_v5 or deterministic_argmax mode required")
        super().__init__(num_worlds,device=device,model_path=model_path)
        self.sampling_mode=sampling_mode
        self.generator=torch.Generator(device=self.device).manual_seed(int(seed))
        self.controller=NativeNextoController(self.num_worlds,self.device)
        # Existing observation builder must consume pending self.action, not the
        # last emitted control; these differ during the native six-tick latency.
        self.previous_action=self.controller.pending_action
        self.player_index=self.controller.player_index
        self.kickoff_index=self.controller.kickoff_index
        self.kickoff_sequence=self.controller.kickoff_sequence
        for parameter in self.actor.parameters():parameter.requires_grad_(False)

    def set_player_index(self,value):self.controller.set_player_index(value)
    def activate(self,mask):self.controller.activate(mask)
    def notify_kickoff(self,reset_mask=None):self.controller.notify_kickoff(reset_mask)

    @torch.inference_mode()
    def _predict_native(self,state,compute,kickoff):
        selected=torch.nonzero(compute,as_tuple=False).flatten()
        observation=build_nexto_observation(state,self.player_index,self.previous_action,constants=self.constants)
        self.observation_builds+=1
        chosen=NextoObservation(q=observation.q[selected],kv=observation.kv[selected],mask=observation.mask[selected])
        logits=self.logits(chosen)
        if not bool(torch.isfinite(logits).all()):raise RuntimeError("nonfinite native Nexto logits")
        picked=logits.argmax(dim=-1)
        random_rows=kickoff[selected] if self.sampling_mode=="native_v5" else torch.zeros_like(selected,dtype=torch.bool)
        if bool(random_rows.any()):
            # log_3((.5+1)/(1-.5)) == 1: native beta=.5 is unscaled logits.
            picked[random_rows]=torch.multinomial(logits[random_rows].softmax(dim=-1),1,generator=self.generator).flatten()
        indices=torch.full((self.num_worlds,),-1,dtype=torch.long,device=self.device)
        indices[selected]=picked
        actions=self.previous_action.clone()
        actions[selected]=self.action_table[picked]
        return actions,indices

    @torch.inference_mode()
    def tick_action(self,state,kickoff_active,active_mask=None):
        return self.controller.step(state,kickoff_active,
            lambda compute,kickoff:self._predict_native(state,compute,kickoff),active_mask)

    def checkpoint_state(self):
        return dict(version=VERSION,model_sha256=MODEL_SHA256,sampling_mode=self.sampling_mode,
                    controller=self.controller.state_dict(),rng_state=self.generator.get_state().clone(),
                    inference_calls=self.inference_calls,observation_builds=self.observation_builds)

    def load_checkpoint_state(self,state):
        if (state.get("version")!=VERSION or state.get("model_sha256")!=MODEL_SHA256
                or state.get("sampling_mode")!=self.sampling_mode):
            raise ValueError("wrong native Nexto adapter identity or sampling mode")
        values=self.controller.validate_state(state["controller"])
        probe=torch.Generator(device=self.device)
        probe.set_state(state["rng_state"].cpu())
        counts=[state[k] for k in ("inference_calls","observation_builds")]
        if any(not isinstance(v,int) or v<0 for v in counts):raise ValueError("invalid Nexto counters")
        for name,value in values.items():getattr(self.controller,name).copy_(value)
        self.generator.set_state(probe.get_state())
        self.inference_calls,self.observation_builds=counts

    def neural_action(self,*args,**kwargs):
        raise RuntimeError("Use tick_action: direct inference bypasses native pending/emission timing")
