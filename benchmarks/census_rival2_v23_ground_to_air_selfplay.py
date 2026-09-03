"""Measure low-bounce aerial-option opportunities in deterministic V23 self-play.

This is a read-only diagnostic.  It does not invoke the option, mutate policy
weights, or alter simulator rewards.  The purpose is to decide whether natural
self-play supplies enough physically eligible possession states for the next
training stage, and to identify whether boost availability is the limiting
factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.full_match import FullMatchRunner  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_hybrid import (  # noqa: E402
    NaturalGroundToAirGateConfig,
    natural_ground_to_air_eligibility,
)
from rivalsim.rival2_ground_to_air_option import (  # noqa: E402
    GroundToAirConfig,
    ground_to_air_eligibility,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
OPTION_AUTHORITY = ROOT / "results/rival2/ground_to_air_hybrid_v1/authority.json"
OUTPUT = ROOT / "results/rival2/ground_to_air_selfplay_census_v1/result.json"
COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _authority_configs() -> tuple[GroundToAirConfig, NaturalGroundToAirGateConfig]:
    authority = json.loads(OPTION_AUTHORITY.read_text(encoding="utf-8"))
    option_names = {field.name for field in fields(GroundToAirConfig)}
    gate_names = {field.name for field in fields(NaturalGroundToAirGateConfig)}
    option = GroundToAirConfig(
        **{name: authority["option_config"][name] for name in option_names}
    )
    gate = NaturalGroundToAirGateConfig(
        **{name: authority["gate"][name] for name in gate_names}
    )
    return option, gate


class SideSpecializedSelfPlayRunner(FullMatchRunner):
    """Run the protected V23 Blue/Orange policies against each other."""

    def __init__(self, *args: Any, orange_checkpoint: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        payload = torch.load(orange_checkpoint, map_location="cpu", weights_only=False)
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != self.checkpoint_identity["policy_config"]:
            raise RuntimeError("V23 side-specialized policy architectures differ")
        if payload.get("contract_hashes") != self.checkpoint_identity["contract_hashes"]:
            raise RuntimeError("V23 side-specialized policy contracts differ")
        self.orange_policy = Rival2ActorCritic(config).to(self.device)
        self.orange_policy.load_state_dict(payload["model"], strict=True)
        self.orange_policy.eval()

    def _update_all_actions(self) -> None:
        observation = self.rival_observation
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation[:, 0])
            orange_actor, _ = self.orange_policy(observation[:, 1])
            self.actions[:, 0].copy_(deterministic_hybrid_action(blue_actor))
            self.actions[:, 1].copy_(deterministic_hybrid_action(orange_actor))

    def tick(self) -> None:
        self._activate_stream()
        self._update_all_actions()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.match_views["rival_scheduler_tick"].add_(1).remainder_(
            self.rival_cadence_ticks
        )
        self.match_views["nexto_scheduler_tick"].add_(1).remainder_(1)
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            wp.copy(self.world.rival2.reset_mask, self.match.pending_reset)
            self.world.apply_interval_resets()
            self.telemetry.after_resets(self.world, self.world.rival2.reset_mask)
        self.rival_observation = self.bridge.observation()


class OpportunityCensus:
    """Device-resident counters for exact and relaxed possession envelopes."""

    BOOST_EDGES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 1.01)

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.names = (
            "player_ticks",
            "grounded",
            "height_125_190",
            "distance_le_100",
            "alignment_ge_050",
            "physical_without_boost",
            "physical_with_boost",
            "natural_gate",
            "broad_without_boost",
            "broad_with_boost",
            "touch_in_broad_envelope",
        )
        self.counts = {
            name: torch.zeros(2, dtype=torch.int64, device=device) for name in self.names
        }
        self.sequence_starts = {
            "physical_without_boost": torch.zeros(2, dtype=torch.int64, device=device),
            "physical_with_boost": torch.zeros(2, dtype=torch.int64, device=device),
            "natural_gate": torch.zeros(2, dtype=torch.int64, device=device),
            "broad_without_boost": torch.zeros(2, dtype=torch.int64, device=device),
            "broad_with_boost": torch.zeros(2, dtype=torch.int64, device=device),
        }
        self.previous: dict[str, torch.Tensor] = {}
        self.boost_histogram = torch.zeros(
            (2, len(self.BOOST_EDGES) - 1), dtype=torch.int64, device=device
        )
        self.broad_feature_sum = torch.zeros((2, 6), dtype=torch.float64, device=device)
        self.broad_feature_count = torch.zeros(2, dtype=torch.int64, device=device)
        self.canonical_positive_y = torch.zeros(2, dtype=torch.int64, device=device)
        self.canonical_positive_vy = torch.zeros(2, dtype=torch.int64, device=device)

    @staticmethod
    def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1
        )

    def step(
        self,
        observation: torch.Tensor,
        *,
        option: GroundToAirConfig,
        gate: NaturalGroundToAirGateConfig,
        active_world: torch.Tensor,
    ) -> None:
        for side in (0, 1):
            obs = observation[:, side]
            exact = ground_to_air_eligibility(obs, option)
            no_boost_option = GroundToAirConfig(
                **{**asdict(option), "minimum_boost_fraction": 0.0}
            )
            physical_without_boost = ground_to_air_eligibility(
                obs, no_boost_option
            ).eligible & active_world
            physical_with_boost = exact.eligible & active_world
            natural = natural_ground_to_air_eligibility(
                obs, option_config=option, gate_config=gate
            ).eligible & active_world

            scale = torch.as_tensor(
                POSITION_SCALE, dtype=obs.dtype, device=obs.device
            )
            relative = self._vector(obs, "relative.ball_position") * scale
            planar = torch.linalg.vector_norm(relative[:, :2], dim=-1)
            planar_direction = relative[:, :2] / planar[:, None].clamp_min(1.0e-6)
            forward = self._vector(obs, "self.forward")
            forward /= torch.linalg.vector_norm(
                forward, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            alignment = (forward[:, :2] * planar_direction).sum(dim=-1)
            ball_height = obs[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
            grounded = obs[:, FIELD["self.on_ground"]] >= 0.5
            boost = obs[:, FIELD["self.boost"]]
            finite = torch.isfinite(obs).all(dim=1)
            broad_without_boost = (
                grounded
                & (obs[:, FIELD["self.is_demoed"]] < 0.5)
                & (ball_height >= 105.0)
                & (ball_height <= 230.0)
                & (planar >= 5.0)
                & (planar <= 160.0)
                & (alignment >= 0.25)
                & finite
                & active_world
            )
            broad_with_boost = broad_without_boost & (boost >= 0.25)
            touch = obs[:, FIELD["lifecycle.self_touch_event"]] >= 0.5

            masks = {
                "player_ticks": active_world,
                "grounded": grounded & active_world,
                "height_125_190": (ball_height >= 125.0)
                & (ball_height <= 190.0)
                & active_world,
                "distance_le_100": (planar >= 5.0)
                & (planar <= 100.0)
                & active_world,
                "alignment_ge_050": (alignment >= 0.5) & active_world,
                "physical_without_boost": physical_without_boost,
                "physical_with_boost": physical_with_boost,
                "natural_gate": natural,
                "broad_without_boost": broad_without_boost,
                "broad_with_boost": broad_with_boost,
                "touch_in_broad_envelope": touch & broad_without_boost,
            }
            for name, mask in masks.items():
                self.counts[name][side] += mask.sum()
            for name in self.sequence_starts:
                mask = masks[name]
                key = f"{name}:{side}"
                previous = self.previous.get(key)
                rising = mask if previous is None else mask & ~previous
                self.sequence_starts[name][side] += rising.sum()
                self.previous[key] = mask.clone()

            broad_boost = boost[broad_without_boost]
            for index, (low, high) in enumerate(
                zip(self.BOOST_EDGES[:-1], self.BOOST_EDGES[1:], strict=True)
            ):
                self.boost_histogram[side, index] += (
                    (broad_boost >= low) & (broad_boost < high)
                ).sum()
            if bool(broad_without_boost.any()):
                ball_y = obs[:, FIELD["ball.position.y"]] * POSITION_SCALE[1]
                ball_vy = (
                    obs[:, FIELD["ball.linear_velocity.y"]]
                    * BALL_LINEAR_SPEED_SCALE
                )
                features = torch.stack(
                    (ball_height, planar, alignment, boost, ball_y, ball_vy), dim=-1
                )
                self.broad_feature_sum[side] += features[
                    broad_without_boost
                ].sum(dim=0, dtype=torch.float64)
                self.broad_feature_count[side] += broad_without_boost.sum()
                self.canonical_positive_y[side] += (
                    ball_y[broad_without_boost] > 0.0
                ).sum()
                self.canonical_positive_vy[side] += (
                    ball_vy[broad_without_boost] > 0.0
                ).sum()

    def export(self) -> dict[str, Any]:
        counts = {
            name: value.detach().cpu().tolist() for name, value in self.counts.items()
        }
        sequence = {
            name: value.detach().cpu().tolist()
            for name, value in self.sequence_starts.items()
        }
        hist = self.boost_histogram.detach().cpu().numpy()
        feature_sum = self.broad_feature_sum.detach().cpu().numpy()
        feature_count = self.broad_feature_count.detach().cpu().numpy()
        feature_names = (
            "ball_height_uu",
            "planar_distance_uu",
            "forward_alignment",
            "boost_fraction",
            "observation_ball_y_uu",
            "observation_ball_vy_uu_per_second",
        )
        means: list[dict[str, float | None]] = []
        for side in (0, 1):
            means.append(
                {
                    name: None
                    if feature_count[side] == 0
                    else float(feature_sum[side, index] / feature_count[side])
                    for index, name in enumerate(feature_names)
                }
            )
        return {
            "tick_counts_by_side": counts,
            "sequence_starts_by_side": sequence,
            "broad_envelope_boost_histogram_by_side": [
                {
                    f"[{low:.2f},{high:.2f})": int(hist[side, index])
                    for index, (low, high) in enumerate(
                        zip(self.BOOST_EDGES[:-1], self.BOOST_EDGES[1:], strict=True)
                    )
                }
                for side in (0, 1)
            ],
            "broad_envelope_feature_means_by_side": means,
            "broad_envelope_observation_direction_by_side": [
                {
                    "positive_ball_y": int(
                        self.canonical_positive_y[side].detach().cpu().item()
                    ),
                    "positive_ball_vy": int(
                        self.canonical_positive_vy[side].detach().cpu().item()
                    ),
                    "denominator": int(feature_count[side]),
                }
                for side in (0, 1)
            ],
        }


def run(args: argparse.Namespace) -> int:
    if sha256_file(BLUE) != BLUE_SHA256 or sha256_file(ORANGE) != ORANGE_SHA256:
        raise RuntimeError("protected V23 checkpoint identity changed")
    option, gate = _authority_configs()
    worlds = int(args.worlds)
    layout = np.arange(worlds, dtype=np.int32) % 5
    # FullMatchState needs a designated telemetry side even though both physical
    # sides are V23 here.  Alternate it so match bookkeeping stays balanced.
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
    census = OpportunityCensus(device=runner.device)
    for tick in range(int(args.ticks)):
        active_world = runner.match_views["done"] == 0
        census.step(
            runner.rival_observation,
            option=option,
            gate=gate,
            active_world=active_world,
        )
        runner.tick()
        if tick and tick % 2_000 == 0:
            print(json.dumps({"tick": tick, "worlds": worlds}), flush=True)
    torch.cuda.synchronize(runner.device)
    result = {
        "format": "RIVAL2_V23_GROUND_TO_AIR_SELFPLAY_CENSUS_V1",
        "policies": {
            "blue": {"path": BLUE.relative_to(ROOT).as_posix(), "sha256": BLUE_SHA256},
            "orange": {
                "path": ORANGE.relative_to(ROOT).as_posix(),
                "sha256": ORANGE_SHA256,
            },
        },
        "option_authority": {
            "path": OPTION_AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(OPTION_AUTHORITY),
            "option_config": asdict(option),
            "natural_gate": asdict(gate),
        },
        "worlds": worlds,
        "ticks": int(args.ticks),
        "world_ticks": worlds * int(args.ticks),
        "seed": int(args.seed),
        "policy_mutation": False,
        "reward_mutation": False,
        "census": census.export(),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--ticks", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=2_026_092_101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
