"""Guard recovery of the frozen T2 runner; never change its training semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-v1")
AMENDMENT_SHA = "CE9E6EB5DDC1F4BD50975F47C0E1FFF7A0D3198FA9C24C254CA6A9C79B1A45D4"


def validate_resume(path, expected_sha256, *, external=EXTERNAL):
    """Under the exclusive lease, reject stale, unbound or intentional-stop resumes.

    The frozen runner still performs strict model/Adam/contract/source checks.
    This wrapper adds latest-accepted selection and cannot select another policy.
    """
    if (external / "STOP").exists():
        raise RuntimeError("STOP present; never clear it in a recovery launcher")
    state = json.loads((external / "campaign_state.json").read_text())
    latest = json.loads((external / "latest.json").read_text())
    if state.get("status") == "stopped_for_exploration_review":
        raise RuntimeError("Intentional review boundary; review before continuation")
    if state.get("exploration_amendment_sha256") != AMENDMENT_SHA:
        raise RuntimeError("Campaign does not identify the frozen exploration amendment")
    # Accepted-boundary state may lag an atomic latest publication by one update
    # if the process exits between files. latest is the durable resume identity.
    offset = latest["accepted_updates"]
    if type(offset) is not int or not 554 <= offset <= 600:
        raise RuntimeError("Latest checkpoint is outside the frozen exploration segment")
    if state["accepted_updates"] not in (offset, offset - 1):
        raise RuntimeError("State/latest discrepancy requires an operational audit")
    if not expected_sha256 or expected_sha256.upper() != latest["sha256"].upper():
        raise RuntimeError("Explicit resume SHA must identify the latest accepted checkpoint")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    if digest != expected_sha256.upper():
        raise RuntimeError("Resume file hash mismatch; do not replace or fall back")
    return dict(path=str(Path(path).resolve()), sha256=digest, accepted_updates=offset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate, never launch PPO")
    args = parser.parse_args()
    # Import does not start a campaign. Acquire the same nonblocking exclusive
    # lease before checking mutable files and retain it throughout frozen run().
    from benchmarks import run_direct_skills_exploration_v1 as frozen

    frozen.torch.set_num_threads(8)
    with frozen.base.gpu_lease():
        identity = validate_resume(args.resume, args.resume_sha256)
        print("VERIFIED_LATEST_RESUME " + json.dumps(identity), flush=True)
        if not args.check_only:
            frozen.run(args)


if __name__ == "__main__":
    main()
