"""Read-only census for the V12 direct aerial-option router in V23 self-play."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.census_rival2_v23_ground_to_air_selfplay import (  # noqa: E402
    BLUE,
    BLUE_SHA256,
    COLLISION_ROOT,
    ORANGE,
    ORANGE_SHA256,
    SideSpecializedSelfPlayRunner,
    sha256_file,
    write_json,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    GROUND_TO_AIR_SELFPLAY_V12_VERSION,
    ROUTE_NAMES,
    AerialOptionRouterConfig,
    aerial_route_eligibility,
)

OUTPUT = ROOT / "results/rival2/ground_to_air_selfplay_v12/opportunity_census.json"


class V12OpportunityCensus:
    def __init__(self, worlds: int, *, device: torch.device) -> None:
        self.worlds = worlds
        self.device = device
        self.player_ticks = torch.zeros((), dtype=torch.int64, device=device)
        self.eligible_ticks = torch.zeros(len(ROUTE_NAMES), dtype=torch.int64, device=device)
        self.eligible_starts = torch.zeros_like(self.eligible_ticks)
        self.previous = torch.zeros(
            (worlds, 2, len(ROUTE_NAMES)), dtype=torch.bool, device=device
        )
        self.feature_sum = torch.zeros(
            (len(ROUTE_NAMES), 5), dtype=torch.float64, device=device
        )

    def step(
        self,
        observation: torch.Tensor,
        *,
        active_world: torch.Tensor,
        config: AerialOptionRouterConfig,
    ) -> None:
        flat = observation.reshape(-1, 182)
        eligibility = aerial_route_eligibility(flat, config)
        active = active_world[:, None].expand(-1, 2).reshape(-1)
        self.player_ticks += active.sum()
        for route_id in range(len(ROUTE_NAMES)):
            mask = (eligibility.route == route_id) & active
            shaped = mask.reshape(self.worlds, 2)
            starts = shaped & ~self.previous[:, :, route_id]
            self.eligible_ticks[route_id] += mask.sum()
            self.eligible_starts[route_id] += starts.sum()
            self.previous[:, :, route_id].copy_(shaped)
            if bool(mask.any()):
                values = torch.stack(
                    (
                        eligibility.ball_height_uu,
                        eligibility.ball_vertical_speed_uu_per_second,
                        eligibility.planar_distance_uu,
                        eligibility.opponent_ball_distance_uu,
                        eligibility.forward_alignment,
                    ),
                    dim=-1,
                )
                self.feature_sum[route_id] += values[mask].sum(
                    dim=0, dtype=torch.float64
                )

    def export(self) -> dict[str, Any]:
        ticks = self.eligible_ticks.detach().cpu()
        starts = self.eligible_starts.detach().cpu()
        sums = self.feature_sum.detach().cpu()
        total = int(self.player_ticks.item())
        names = (
            "ball_height_uu",
            "ball_vertical_speed_uu_per_second",
            "planar_distance_uu",
            "opponent_ball_distance_uu",
            "forward_alignment",
        )
        routes: dict[str, Any] = {}
        for route_id, name in enumerate(ROUTE_NAMES):
            count = int(ticks[route_id].item())
            routes[name] = {
                "eligible_player_ticks": count,
                "eligible_fraction": count / max(total, 1),
                "sequence_starts": int(starts[route_id].item()),
                "feature_means": {
                    field: None if count == 0 else float(sums[route_id, index] / count)
                    for index, field in enumerate(names)
                },
            }
        return {
            "player_ticks": total,
            "eligible_player_ticks": int(ticks.sum().item()),
            "eligible_fraction": int(ticks.sum().item()) / max(total, 1),
            "sequence_starts": int(starts.sum().item()),
            "routes": routes,
        }


def run(args: argparse.Namespace) -> int:
    if sha256_file(BLUE) != BLUE_SHA256 or sha256_file(ORANGE) != ORANGE_SHA256:
        raise RuntimeError("protected V23 checkpoint identity changed")
    worlds = int(args.worlds)
    layout = np.arange(worlds, dtype=np.int32) % 5
    designated_side = np.arange(worlds, dtype=np.int32) % 2
    runner = SideSpecializedSelfPlayRunner(
        worlds,
        str(args.collision_root),
        BLUE,
        starting_layout=layout,
        rival_side=designated_side,
        stochastic_rival=False,
        evaluation_seed=int(args.seed),
        orange_checkpoint=ORANGE,
        device=args.device,
    )
    config = AerialOptionRouterConfig()
    census = V12OpportunityCensus(worlds, device=runner.device)
    for tick in range(int(args.ticks)):
        census.step(
            runner.rival_observation,
            active_world=runner.match_views["done"] == 0,
            config=config,
        )
        runner.tick()
        if tick and tick % 2_000 == 0:
            print(json.dumps({"tick": tick, "worlds": worlds}), flush=True)
    torch.cuda.synchronize(runner.device)
    payload = {
        "format": "RIVAL2_V23_GROUND_TO_AIR_SELFPLAY_V12_CENSUS",
        "router_version": GROUND_TO_AIR_SELFPLAY_V12_VERSION,
        "router_config": asdict(config),
        "policies": {
            "blue": {"path": BLUE.relative_to(ROOT).as_posix(), "sha256": BLUE_SHA256},
            "orange": {
                "path": ORANGE.relative_to(ROOT).as_posix(),
                "sha256": ORANGE_SHA256,
            },
        },
        "worlds": worlds,
        "ticks": int(args.ticks),
        "seed": int(args.seed),
        "policy_mutation": False,
        "reward_mutation": False,
        "census": census.export(),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=2_026_090_301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
