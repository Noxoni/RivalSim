"""Reset-aware GRU execution without changing recurrent model/state semantics."""

from __future__ import annotations

import torch
from torch import nn

RESET_EXECUTION_VERSION = "RIVAL2_RESET_BOUNDARY_GRU_V1"


def gru_reset_spans(
    gru: nn.GRU,
    encoded: torch.Tensor,
    hidden: torch.Tensor,
    reset_before: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse reset-free spans; preserve gradients until each row's actual reset.

    A reset at t=0 must not make a whole rollout run as single-tick GRU calls.
    Find the union of reset times with one metadata transfer, zero only the
    affected rows at each boundary, and retain the full autograd graph. There
    are no detached states, packed/padded episode copies, or parameter changes.
    The one-tick inference path needs no host synchronization at all.
    """
    if reset_before is None:
        return gru(encoded, hidden)
    if reset_before.shape != encoded.shape[:2]:
        raise ValueError("reset_before must have shape [batch, sequence]")
    resets = reset_before.to(torch.bool)
    if encoded.shape[1] == 1:
        return gru(encoded, hidden.masked_fill(resets[:, 0].view(1, -1, 1), 0.0))
    boundaries = resets.any(dim=0).nonzero().flatten().cpu().tolist()
    if not boundaries:
        return gru(encoded, hidden)
    outputs = []
    start = 0
    for boundary in boundaries:
        if boundary > start:
            output, hidden = gru(encoded[:, start:boundary], hidden)
            outputs.append(output)
        hidden = hidden.masked_fill(resets[:, boundary].view(1, -1, 1), 0.0)
        start = boundary
    output, hidden = gru(encoded[:, start:], hidden)
    outputs.append(output)
    return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], hidden
