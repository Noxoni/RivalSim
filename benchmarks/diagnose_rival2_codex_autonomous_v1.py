"""Bounded ablations for the closed-loop-gated autonomous Rival campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign
from rivalsim.human_demo.behavior_cloning import MechanicHierarchySampler


def replay_only(blocks: int, output: Path, *, device: str) -> None:
    train, validation, teacher, _identity = campaign.load_human_data(device=device)
    trainer, _source = campaign.build_trainer(
        campaign.DEFAULT_COLLISION_DIR,
        worlds=1,
        device=device,
    )
    teacher_gameplay = campaign.precompute_teacher_actor(
        teacher, train.gameplay_observation, device=device
    )
    teacher_mechanic = campaign.precompute_teacher_actor(
        teacher, train.mechanic_observation, device=device
    )
    generator = torch.Generator(device="cpu").manual_seed(campaign.SEED ^ 0xBCCD)
    sampler = MechanicHierarchySampler(
        train.mechanic_label,
        train.mechanic_attempt,
        uniform_label_fraction=0.10,
        maximum_oversampling_ratio=4.0,
        generator=generator,
    )
    for index in range(blocks):
        print(
            {
                "replay_block": index + 1,
                "metrics": campaign.human_replay(
                    trainer,
                    train,
                    teacher_gameplay,
                    teacher_mechanic,
                    sampler,
                    generator,
                ),
            },
            flush=True,
        )
    campaign.checkpoint(
        trainer,
        output,
        campaign_step=0,
        human_generator=generator,
        best={},
    )
    print(
        {
            "human_validation": campaign.human_validation(
                trainer.model, validation, device=device
            )
        },
        flush=True,
    )
    print(
        {
            "nexto_evaluation": campaign.run_nexto_evaluation(
                output,
                campaign_step=blocks * campaign.HUMAN_REPLAY_STEPS,
                run_dir=output.parent,
                device=device,
                collision_dir=campaign.DEFAULT_COLLISION_DIR,
                worlds_per_side=128,
            )
        },
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-only-blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    torch.cuda.set_device(arguments.device)
    replay_only(
        arguments.replay_only_blocks,
        arguments.output,
        device=arguments.device,
    )
