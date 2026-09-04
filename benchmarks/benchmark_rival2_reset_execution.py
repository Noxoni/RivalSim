"""Compare equivalent reset-aware GRU execution, without training a policy."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.recurrent_execution import gru_reset_spans  # noqa: E402


def legacy(gru, encoded, hidden, resets):
    if resets is None or not bool(torch.any(resets)):
        return gru(encoded, hidden)
    outputs = []
    for tick in range(encoded.shape[1]):
        reset = resets[:, tick].bool()
        if bool(torch.any(reset)):
            hidden = hidden.masked_fill(reset.view(1, -1, 1), 0)
        output, hidden = gru(encoded[:, tick : tick + 1], hidden)
        outputs.append(output)
    return torch.cat(outputs, 1), hidden


def spans(gru, encoded, hidden, resets):
    return gru_reset_spans(gru, encoded, hidden, resets)


def packed_episodes(gru, encoded, hidden, resets):
    if resets is None:
        return gru(encoded, hidden)
    if resets.shape != encoded.shape[:2]:
        raise ValueError("reset_before must have shape [batch, sequence]")
    batch, ticks, width = encoded.shape
    if ticks == 1:
        return gru(encoded, hidden.masked_fill(resets[:, 0].bool().view(1, -1, 1), 0))
    # Only metadata moves to CPU. Episodes remain full-gradient, GPU tensors.
    reset_cpu = resets.bool().cpu()
    if not bool(reset_cpu.any()):
        return gru(encoded, hidden)
    owners, starts, lengths, carry, last = [], [], [], [], []
    for row in range(batch):
        boundaries = [0, *(reset_cpu[row, 1:].nonzero().flatten() + 1).tolist(), ticks]
        for start, end in itertools.pairwise(boundaries):
            owners.append(row)
            starts.append(start)
            lengths.append(end - start)
            carry.append(start == 0 and not bool(reset_cpu[row, 0]))
        last.append(len(owners) - 1)
    device = encoded.device
    owner = torch.tensor(owners, device=device)
    start = torch.tensor(starts, device=device)
    length_cpu = torch.tensor(lengths, dtype=torch.long)
    offset = torch.arange(max(lengths), device=device)
    valid = offset[None, :] < length_cpu.to(device)[:, None]
    source = owner[:, None] * ticks + start[:, None] + offset[None, :]
    padded = encoded.reshape(-1, width)[source.clamp_max(batch * ticks - 1)]
    initial = hidden.index_select(1, owner) * torch.tensor(carry, device=device)[None, :, None]
    packed = pack_padded_sequence(padded, length_cpu, batch_first=True, enforce_sorted=False)
    output, final = gru(packed, initial)
    unpacked, _ = pad_packed_sequence(output, batch_first=True)
    restored = encoded.new_zeros(batch * ticks, gru.hidden_size).index_copy(
        0, source[valid], unpacked[valid]
    )
    return (
        restored.reshape(batch, ticks, -1),
        final.index_select(1, torch.tensor(last, device=device)),
    )


def packed_fp32(gru, encoded, hidden, resets):
    # Packed cuDNN can select TF32 kernels where the unpacked path did not.
    with torch.backends.cudnn.flags(allow_tf32=False):
        return packed_episodes(gru, encoded, hidden, resets)


CANDIDATES = {
    "legacy": legacy,
    "spans": spans,
    "packed_episodes": packed_episodes,
    "packed_fp32": packed_fp32,
}


def masks(kind, batch, ticks, device):
    result = torch.zeros(batch, ticks, device=device, dtype=torch.bool)
    if kind == "first":
        result[:, 0] = True
    elif kind == "staggered":
        result[:, 0] = True
        result[torch.arange(batch), (torch.arange(batch) * 11 + 1) % ticks] = True
    elif kind == "dense":
        result = torch.rand(batch, ticks, generator=torch.Generator().manual_seed(52)) < 0.04
        result = result.to(device)
    elif kind == "every":
        result[:] = True
    elif kind == "mixed":
        result[0, 0] = True
        result[-1, -1] = True
        result[::2, ticks // 2] = True
    elif kind != "none":
        raise ValueError(kind)
    return result


def benchmark(args):
    torch.set_num_threads(2)
    torch.manual_seed(904)
    device = torch.device(args.device)
    gru = torch.nn.GRU(256, 256, batch_first=True).to(device)
    x = torch.randn(args.batch, args.ticks, 256, device=device)
    h = torch.randn(1, args.batch, 256, device=device)
    rows = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for kind in args.patterns:
        reset = masks(kind, args.batch, args.ticks, device)
        for name, function in CANDIDATES.items():
            for backward in (False, True):
                times = []
                for repeat in range(args.repeats + 1):
                    gru.zero_grad(set_to_none=True)
                    synchronize()
                    started = time.perf_counter()
                    with torch.set_grad_enabled(backward):
                        y, final = function(gru, x, h, reset)
                        if backward:
                            (y.square().mean() + final.square().mean()).backward()
                    synchronize()
                    elapsed = time.perf_counter() - started
                    del y, final
                    if repeat:
                        times.append(elapsed)
                row = dict(
                    pattern=kind,
                    method=name,
                    backward=backward,
                    median_seconds=statistics.median(times),
                    measurements=times,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = dict(
        device=str(device),
        batch=args.batch,
        ticks=args.ticks,
        rows=rows,
        optimizer_steps=0,
        production_checkpoint_modified=False,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--patterns", nargs="+", default=["first", "staggered", "dense"])
    parser.add_argument("--output", required=True)
    benchmark(parser.parse_args())
