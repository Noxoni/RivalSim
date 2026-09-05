"""Lossless deterministic interface to the frozen existing evaluation harness.

The harness expects 13 hybrid-format outputs; training NEVER uses this view.
Table values are exactly {-1,0,1}/Boolean, so tanh(20*a) reconstructs analog a
exactly in float32. Button signs reconstruct the three exact booleans. This
allows unchanged scenario/metric code and hashes, without replacing that frozen
code or sampling any hybrid distribution. Only the categorical network acts.
"""

import torch


def evaluation_output(action):
    return torch.cat(
        (action[..., :5] * 20, torch.zeros_like(action[..., :5]), action[..., 5:] * 2 - 1), -1
    )


class DeterministicEvaluationView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.policy = model

    def initial_hidden(self, *args, **kwargs):
        return self.policy.initial_hidden(*args, **kwargs)

    def forward_actor(self, *args, **kwargs):
        logits, hidden = self.policy.forward_actor(*args, **kwargs)
        return evaluation_output(self.policy.deterministic(logits)), hidden
