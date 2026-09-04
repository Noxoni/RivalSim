"""Run the corrected fresh Unified-V5-rooted SSL Foundation PPO campaign.

V1 is retained as immutable evidence of the stopped, invalid lineage.  This
entry point binds a separate authority, result directory, run directory, and
checkpoint namespace and has no launch-time dependency on any V1 PPO update.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_ppo_v1 as engine  # noqa: E402

FORMAT = "RIVAL2_SSL_FOUNDATION_PPO_V2"
RESULTS = ROOT / "results/rival2/ssl_foundation_ppo_v2"
AUTHORITY = RESULTS / "authority.json"
SCHEDULE_AUTHORITY = RESULTS / "launch_authority.json"
CHECKPOINT = (
    ROOT / "checkpoints/rival2/ssl_foundation_ppo_v2" / "rival2_ssl_foundation_ppo_v2.pt"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ssl-foundation-ppo-v2")
SNAPSHOT_INTERVAL = 50
CONTINUATION_REVIEW_MARKER = 600
SUPERSEDED_BROKEN_AUTHORITY_SHA256 = (
    "F4745DCD19AE968E2B844DAA5F8E4350F20928DCD0B13B6D3938A62D69713FF3"
)

_base_authority_payload = engine.authority_payload
_base_load_authority = engine.load_authority


def _configure_engine() -> None:
    engine.FORMAT = FORMAT
    engine.CHECKPOINT_FORMAT = f"{FORMAT}_CHECKPOINT"
    # The trainer phase names the unchanged reward/rollout implementation;
    # checkpoint format and lineage provide the V2 campaign isolation.
    engine.TRAINER_PHASE = "ssl_foundation_v1"
    engine.LINEAGE = "Unified Capability V5 -> SSL Foundation PPO V2"
    engine.RESULTS = RESULTS
    engine.AUTHORITY = AUTHORITY
    engine.SCHEDULE_AUTHORITY = SCHEDULE_AUTHORITY
    engine.CHECKPOINT = CHECKPOINT
    engine.DEFAULT_RUN_DIR = DEFAULT_RUN_DIR
    engine.SNAPSHOT_INTERVAL = SNAPSHOT_INTERVAL
    engine.ACTIVE_SNAPSHOT_INTERVAL = SNAPSHOT_INTERVAL
    engine.CONTINUATION_REVIEW_MARKER = CONTINUATION_REVIEW_MARKER
    engine.SUPERSEDED_AUTHORITY_SHA256 = SUPERSEDED_BROKEN_AUTHORITY_SHA256
    engine.BOUND_RESUME_UPDATE = None
    engine.BOUND_RESUME_SHA256 = None


def authority_payload(implementation_commit: str) -> dict[str, Any]:
    _configure_engine()
    payload = _base_authority_payload(implementation_commit)
    payload["supersession_reason"] = (
        "replace invalid 120 Hz lifecycle and update-20-bound launch lineage with a "
        "fresh Unified-V5-rooted corrected campaign"
    )
    payload["source_transition"] = {
        "initial_checkpoint": payload["source"],
        "initial_accepted_ppo_updates": 0,
        "initial_policy_version_increment": 0,
        "initial_resume_checkpoint": None,
        "fresh_optimizer": True,
        "forbidden_lineage": "RIVAL2_SSL_FOUNDATION_PPO_V1",
        "forbidden_update_20_sha256": (
            "95B38C239F38B86E6699410813F73ACCC21C23D872AF6284F383F2B17BB1E05E"
        ),
    }
    payload["reset_curriculum"]["heading_generation"] = {
        "global_face_ball_postprocess": False,
        "ground_heading_momentum": "coherent_with_off_angle_coverage",
        "intentionally_aligned_families": ["shooting_finishing", "contested_fifty"],
    }
    payload["reset_curriculum"]["wall_aerial_variants"] = payload["reset_curriculum"][
        "realized"
    ]["wall_aerial_variant_counts"]
    payload["campaign"]["snapshot_interval"] = SNAPSHOT_INTERVAL
    payload["campaign"]["evaluation_interval"] = SNAPSHOT_INTERVAL
    payload["campaign"]["continuation_review_marker"] = CONTINUATION_REVIEW_MARKER
    payload["campaign"]["continuation_after_marker"] = (
        "requires explicit review of update-600 evaluation progress"
    )
    return payload


def launch_authority_payload() -> dict[str, Any]:
    return {
        "format": f"{FORMAT}_FRESH_LAUNCH_AUTHORITY_V1",
        "parent_authority_sha256": engine.sha256_file(AUTHORITY),
        "launch": {
            "mode": "fresh_from_unified_v5",
            "resume_checkpoint": None,
            "accepted_updates": 0,
            "fresh_optimizer": True,
            "source_sha256": engine.SOURCE_SHA256,
        },
        "resume_checkpoint": {
            "path": None,
            "sha256": None,
            "accepted_updates": 0,
            "purpose": (
                "initial launch only; rolling checkpoint is reserved for operational recovery"
            ),
        },
        "evaluation_and_snapshot_interval": SNAPSHOT_INTERVAL,
        "continuation_review_marker": CONTINUATION_REVIEW_MARKER,
        "reward_or_ppo_hyperparameters_changed": False,
        "scenario_corpus_changed_from_broken_v1": True,
        "scenario_corpus_changes": [
            "standard kickoff reset family",
            "coherent ground heading and momentum with off-angle coverage",
            "grounded, side-wall, and airborne wall/aerial variants",
        ],
        "broken_v1_update_20_resume_forbidden": True,
    }


def load_authority() -> dict[str, Any]:
    _configure_engine()
    payload = _base_load_authority()
    checks = {
        "fresh_zero_update_source": payload.get("source_transition", {}).get(
            "initial_accepted_ppo_updates"
        )
        == 0,
        "no_initial_resume": payload.get("source_transition", {}).get(
            "initial_resume_checkpoint"
        )
        is None,
        "fresh_optimizer": payload.get("source_transition", {}).get("fresh_optimizer") is True,
        "v1_lineage_forbidden": payload.get("source_transition", {}).get("forbidden_lineage")
        == "RIVAL2_SSL_FOUNDATION_PPO_V1",
        "global_face_ball_removed": payload.get("reset_curriculum", {})
        .get("heading_generation", {})
        .get("global_face_ball_postprocess")
        is False,
        "coherent_heading": payload.get("reset_curriculum", {})
        .get("heading_generation", {})
        .get("ground_heading_momentum")
        == "coherent_with_off_angle_coverage",
        "three_wall_aerial_variants": all(
            count > 0
            for count in payload.get("reset_curriculum", {})
            .get("wall_aerial_variants", {})
            .values()
        )
        and len(payload.get("reset_curriculum", {}).get("wall_aerial_variants", {})) == 3,
    }
    if not all(checks.values()):
        raise RuntimeError(f"corrected SSL authority mismatch: {checks}")
    return payload


def load_schedule_authority() -> dict[str, Any]:
    payload = json.loads(SCHEDULE_AUTHORITY.read_text(encoding="utf-8"))
    launch = payload.get("launch", {})
    resume = payload.get("resume_checkpoint", {})
    checks = {
        "format": payload.get("format") == f"{FORMAT}_FRESH_LAUNCH_AUTHORITY_V1",
        "parent": payload.get("parent_authority_sha256") == engine.sha256_file(AUTHORITY),
        "fresh_mode": launch.get("mode") == "fresh_from_unified_v5",
        "source": launch.get("source_sha256") == engine.SOURCE_SHA256,
        "fresh_optimizer": launch.get("fresh_optimizer") is True,
        "zero_updates": launch.get("accepted_updates") == resume.get("accepted_updates") == 0,
        "no_resume_path": launch.get("resume_checkpoint") is None
        and resume.get("path") is None
        and resume.get("sha256") is None,
        "interval": payload.get("evaluation_and_snapshot_interval") == SNAPSHOT_INTERVAL,
        "marker": payload.get("continuation_review_marker") == CONTINUATION_REVIEW_MARKER,
        "old_update_20_forbidden": payload.get("broken_v1_update_20_resume_forbidden") is True,
        "reward_and_ppo_unchanged": payload.get("reward_or_ppo_hyperparameters_changed") is False,
        "scenario_corpus_corrected": payload.get("scenario_corpus_changed_from_broken_v1") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"corrected SSL launch authority mismatch: {checks}")
    return payload


def run(args: argparse.Namespace) -> int:
    _configure_engine()
    engine.load_authority = load_authority
    engine.load_schedule_authority = load_schedule_authority
    if args.write_authority:
        if not args.implementation_commit:
            raise ValueError("--implementation-commit is required with --write-authority")
        engine.write_json(AUTHORITY, authority_payload(args.implementation_commit))
        engine.write_json(SCHEDULE_AUTHORITY, launch_authority_payload())
        print(AUTHORITY)
        print(SCHEDULE_AUTHORITY)
        return 0
    return engine.run(args)


def parser() -> argparse.ArgumentParser:
    result = engine.parser()
    result.set_defaults(run_dir=str(DEFAULT_RUN_DIR))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
