"""Isolate graph-captured match evaluation from borrowed training CUDA streams."""

from contextlib import contextmanager

import torch
import warp as wp


@contextmanager
def owned_match_stream(device="cuda:0"):
    """Own one registered Warp stream for the entire match, restore both callers.

    Rival2Env repeatedly wraps a Torch stream. In the installed Warp release,
    destruction of an older wrapper can unregister that shared handle, making
    graph capture reject the surviving wrapper as an unknown stream. A new,
    owned handle avoids that aliasing without changing simulation or sampling.
    Synchronization is evaluation-boundary-only, never in the PPO hot path.
    """
    wp.init()
    old_torch, old_warp = torch.cuda.current_stream(device), wp.get_stream(device)
    torch.cuda.synchronize(device)
    stream = wp.Stream(device)
    torch_stream = wp.stream_to_torch(stream)
    torch.cuda.set_stream(torch_stream)
    wp.set_stream(stream, device=device, sync=False)
    try:
        yield stream
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.set_stream(old_torch)
        wp.set_stream(old_warp, device=device, sync=False)
