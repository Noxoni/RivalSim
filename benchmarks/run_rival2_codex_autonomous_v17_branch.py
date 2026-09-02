"""One prospectively frozen full-policy PPO branch from the protected match parent.

The prior sustained continuation proved that another analog-only optimizer step
destroys the selected policy's continuous-play behavior.  V17 instead starts a
fresh, much smaller optimizer, lets the complete actor/trunk learn, retains one
reviewed-human anchor step per rollout, and selects only with paired continuous
Nexto play.  Branch outputs remain provisional until the complete five-minute
matrix independently promotes one.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign


BRANCH_SEEDS = (2_026_090_233, 2_026_090_237, 2_026_090_241, 2_026_090_243)
SOURCE = (
    ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
SOURCE_SHA256 = "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v17/authority.json"
WINDOW_SECONDS = 60
EVALUATION_SEED = 2_026_090_206


def _window_evaluation(
    checkpoint_path: Path,
    *,
    campaign_step: int,
    run_dir: Path,
    device: str,
    collision_dir: Path,
    worlds_per_side: int = 128,
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    del device, worlds_per_side
    digest = campaign.sha256_file(checkpoint_path)
    label = f"codex_autonomous_u{campaign_step:04d}"
    work_dir = run_dir / "continuous_evaluations" / label
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "evaluation.json"
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/run_rival2_codex_autonomous_match_window_eval.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        digest,
        "--output",
        str(result_path),
        "--window-seconds",
        str(WINDOW_SECONDS),
        "--seed",
        str(EVALUATION_SEED if evaluation_seed is None else evaluation_seed),
        "--collision-root",
        str(collision_dir.parent),
    ]
    with (work_dir / "stdout.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as stdout, (work_dir / "stderr.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError((work_dir / "stderr.txt").read_text(encoding="utf-8")[-4000:])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    overall = result["overall"]
    return {
        "campaign_step": int(campaign_step),
        "checkpoint_sha256": digest,
        "episodes": 10,
        "goals_for": int(overall["goals_for"]),
        "goals_against": int(overall["goals_against"]),
        "draws": 0,
        "win_rate": 0.0,
        "touches": int(overall["rival_touches"]),
        "opponent_touches": int(overall["opponent_touches"]),
        "first_touches": int(overall["rival_kickoff_first_touches"]),
        "no_touch_episodes": 0,
        "hard_timeouts": 0,
        "mean_speed_uu_per_s": 0.0,
        "full_result": str(result_path),
        "selection_method": "continuous_60_second_ten_layout_matrix",
    }


def _configure(branch_seed: int) -> tuple[Path, Path]:
    branch = str(branch_seed)
    results = ROOT / f"results/rival2/codex_autonomous_v17/branches/{branch}"
    checkpoints = ROOT / f"checkpoints/rival2/codex_autonomous_v17/branches/{branch}"
    run_dir = Path(f"G:/dev/RivalSim-runs/codex-autonomous-v17/branches/{branch}")
    campaign.SOURCE = SOURCE
    campaign.SOURCE_SHA256 = SOURCE_SHA256
    campaign.AUTHORITY = AUTHORITY
    campaign.RESULTS = results
    campaign.CHECKPOINTS = checkpoints
    campaign.DEFAULT_RUN_DIR = run_dir
    campaign.SEED = branch_seed
    campaign.POLICY_LR = 1.0e-7
    campaign.CRITIC_LR = 5.0e-7
    campaign.EXPLORATION_SIGMA = 0.05
    campaign.EXPLORATION_BUTTON_TEMPERATURE = 1.15
    campaign.HUMAN_REPLAY_STEPS = 1
    campaign.OPTIMIZER_STEP_LIMIT = 1
    campaign.POLICY_TRAINING_BOUNDARY = "full"
    campaign.CURRENT_OPPONENT_PROBABILITY = 0.5
    campaign.NEXTO_OPPONENT_PROBABILITY = 0.5
    campaign.AUTHORIZED_BRANCH_SEEDS = BRANCH_SEEDS
    campaign.MATERIAL_REGRESSION_PATIENCE = 100
    campaign.EVALUATION_INTERVAL = 2
    campaign.NEXTO_WIN_TARGET = 999.0
    campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V17"
    campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V17_AUTHORITY"
    campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V17_PREFLIGHT"
    campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V17_RESULT"
    campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V17_STATE"
    campaign.run_nexto_evaluation = _window_evaluation
    return results, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-seed", type=int, choices=BRANCH_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=campaign.DEFAULT_COLLISION_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _results, run_dir = _configure(args.branch_seed)
    run_args = Namespace(
        collision_dir=args.collision_dir,
        device=args.device,
        run_dir=run_dir,
        worlds=campaign.WORLD_COUNT,
        target_updates=8,
        evaluation_interval=2,
        resume=False,
        preflight_only=args.preflight_only,
    )
    return campaign.run(run_args)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
