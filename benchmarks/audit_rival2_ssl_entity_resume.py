"""Read-only CPU watcher proving exact pre-update model/Adam restoration."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(identical(x, y) for x, y in zip(a, b, strict=True))
        )
    return type(a) is type(b) and a == b


def main(args):
    source_bytes = args.before.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
    assert source_hash == args.sha256.upper()
    before = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=False)
    assert not args.output.exists(), "Do not overwrite existing proof"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        candidate_bytes = args.after.read_bytes()
        candidate_hash = hashlib.sha256(candidate_bytes).hexdigest().upper()
        if candidate_hash == source_hash:
            time.sleep(2)
            continue
        after = torch.load(io.BytesIO(candidate_bytes), map_location="cpu", weights_only=False)
        keys = (
            "model",
            "optimizer",
            "accepted_updates",
            "cumulative_optimizer_steps",
            "new_agent_samples",
            "new_physics_ticks",
            "continuation_authority_sha256",
            "continuation_parent_sha256",
            "policy_config_sha256",
            "ppo_config_sha256",
            "policy_generator_state",
            "shuffle_generator_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
        )
        checks = {key: identical(before[key], after[key]) for key in keys}
        checks["resume_counter_increased"] = after["resume_count"] == before["resume_count"] + 1
        assert all(checks.values()), checks
        result = dict(
            verdict="PASS",
            utc=datetime.now(UTC).isoformat(),
            checks=checks,
            accepted_updates=before["accepted_updates"],
            cumulative_optimizer_steps=before["cumulative_optimizer_steps"],
            source_checkpoint_sha256=source_hash,
            resaved_checkpoint_sha256=candidate_hash,
            optimizer_steps_in_audit=0,
            observed_training_optimizer_steps_since_resume=0,
            source_path=str(args.before),
            resumed_path=str(args.after),
            episode_semantics=(
                "Saved model/Adam/counters/RNG; fresh physical episodes, zero hidden, "
                "newly assigned opponents from restored generator. "
                "Old assignments/cache are provenance only."
            ),
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result), flush=True)
        return
    raise TimeoutError("No initial resaved checkpoint observed; do not claim equality")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    torch.set_num_threads(8)
    main(parser.parse_args())
