"""Diagnostic-only native get_output neural compute/emission timing.

Not a replacement production adapter or full native opponent implementation.
Kickoff admission/script lifecycle and deterministic neural selection deliberately
remain the baseline simulator's so this experiment isolates timing and table.
"""
from __future__ import annotations

import torch


def advance_native_clock(ticks, update_action):
    """One delivered physics frame, extracted get_output clock semantics."""
    ticks += 1
    compute = bool(update_action)
    update_action = False if compute else update_action
    emit = ticks >= 7
    if ticks >= 8:
        ticks = 0
        update_action = True
    return ticks, update_action, compute, emit


def make_timing_adapter(base_class):
    class TimingProbe(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.native_ticks = 8
            self.native_update_action = True
            self.emitted = self.previous_action.clone()
            self.probe_frames = 0
            self.compute_frames = []
            self.emission_frames = []

        def tick_action(self, state, kickoff_active, active_mask=None):
            if active_mask is None:
                active_mask = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
            # Frozen 20s experiment: all full matches remain active. Do not
            # extrapolate this diagnostic to dynamic training-world assignments.
            assert bool(active_mask.all())
            self.probe_frames += 1
            self.native_ticks, self.native_update_action, compute, emit = advance_native_clock(
                self.native_ticks, self.native_update_action)
            indices = None
            if compute:
                _, indices = self.neural_action(state)
                self.compute_frames.append(self.probe_frames)
            if emit:
                self.emitted.copy_(self.previous_action)
                self.emission_frames.append(self.probe_frames)
            # Retain *exactly* the baseline kickoff admission and script-index
            # semantics. Avoid an additional base-class neural inference.
            self.neural_counter.fill_(1)
            next_index = torch.where(kickoff_active & (self.kickoff_index < 0),
                                     torch.zeros_like(self.kickoff_index), self.kickoff_index)
            script = kickoff_active & (next_index >= 0) & (next_index < 168) & (state.ball_pos[:, 1] == 0)
            action, _ = super().tick_action(state, kickoff_active, active_mask=active_mask)
            self.emitted.copy_(torch.where(script[:, None], action, self.emitted))
            return self.emitted, indices

    return TimingProbe
