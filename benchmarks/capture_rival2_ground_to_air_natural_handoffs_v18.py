"""Capture exact full RivalSim states at natural V23 aerial handoffs.

This is a read-only corpus capture. It runs the immutable V23 side policies and
the exact protected V3 aerial scorer with the frozen physical router, and saves
the complete public physical state, lifecycle snapshot, all zero-copy Rival2
bridge views, exact observations, actions, side, route, source world, and tick
for every newly activated natural route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_rival2_ground_to_air_selfplay_v12_natural import (  # noqa: E402
    BLUE,
    BLUE_SHA256,
    ORANGE,
    ORANGE_SHA256,
    SELECTED,
    V12NaturalSelfPlayRunner,
    handoff_features,
    sha256_file,
)
from rivalsim.lifecycle_state import LifecycleSnapshot  # noqa: E402
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402
from rivalsim.state import StateSnapshot  # noqa: E402

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_HANDOFF_CORPUS_V18"
OPTION_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"
DEFAULT_OUTPUT = (
    ROOT
    / "results/rival2/ground_to_air_natural_handoffs_v18"
    / "natural_handoffs.pt"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".json")
DEFAULT_COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")


def _canonical_array_digest(groups: dict[str, dict[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for group_name in sorted(groups):
        digest.update(group_name.encode("utf-8"))
        for name in sorted(groups[group_name]):
            value = np.ascontiguousarray(groups[group_name][name])
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest().upper()


def _slice_dataclass(value: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        item.name: np.ascontiguousarray(getattr(value, item.name)[rows])
        for item in fields(value)
    }


class NaturalHandoffCaptureRunner(V12NaturalSelfPlayRunner):
    """Read-only natural runner that snapshots each activation lane."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capture_state: dict[str, list[np.ndarray]] = {
            item.name: [] for item in fields(StateSnapshot)
        }
        self.capture_lifecycle: dict[str, list[np.ndarray]] = {
            item.name: [] for item in fields(LifecycleSnapshot)
        }
        self.capture_views: dict[str, list[torch.Tensor]] = {
            name: [] for name in sorted(self.bridge.views)
        }
        self.capture_observation: list[torch.Tensor] = []
        self.capture_base_action: list[torch.Tensor] = []
        self.capture_option_action: list[torch.Tensor] = []
        self.capture_world: list[torch.Tensor] = []
        self.capture_side: list[torch.Tensor] = []
        self.capture_route: list[torch.Tensor] = []
        self.capture_tick: list[torch.Tensor] = []

    def _capture_activations(
        self,
        activated: torch.Tensor,
        route: torch.Tensor,
        base_actions: torch.Tensor,
        option_actions: torch.Tensor,
    ) -> None:
        lanes = torch.nonzero(activated, as_tuple=False).flatten()
        if lanes.numel() == 0:
            return
        worlds = torch.div(lanes, 2, rounding_mode="floor")
        sides = lanes.remainder(2)
        rows = worlds.detach().cpu().numpy().astype(np.int64, copy=False)
        state = self.world.snapshot()
        lifecycle = self.world.lifecycle_snapshot()
        for name, value in _slice_dataclass(state, rows).items():
            self.capture_state[name].append(value)
        for name, value in _slice_dataclass(lifecycle, rows).items():
            self.capture_lifecycle[name].append(value)
        for name, view in self.bridge.views.items():
            per_world = view.reshape(self.num_worlds, -1)
            self.capture_views[name].append(per_world.index_select(0, worlds).cpu())
        self.capture_observation.append(
            self.rival_observation.index_select(0, worlds).cpu()
        )
        self.capture_base_action.append(base_actions.index_select(0, worlds).cpu())
        self.capture_option_action.append(option_actions.index_select(0, worlds).cpu())
        self.capture_world.append(worlds.cpu())
        self.capture_side.append(sides.cpu())
        self.capture_route.append(route.index_select(0, lanes).cpu())
        self.capture_tick.append(
            torch.full_like(worlds.cpu(), int(self.host_tick), dtype=torch.int64)
        )

    def _update_all_actions(self) -> None:
        observation = self.rival_observation
        flat = observation.reshape(-1, 182)
        lifecycle = self.match_views["kickoff_active"][:, None].expand(-1, 2)
        done = self.match_views["done"][:, None].expand(-1, 2)
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation[:, 0])
            orange_actor, _ = self.orange_policy(observation[:, 1])
            base_actions = torch.stack(
                (
                    deterministic_hybrid_action(blue_actor),
                    deterministic_hybrid_action(orange_actor),
                ),
                dim=1,
            )
            option_actor, _ = self.option_policy(flat)
            option_actions = deterministic_hybrid_action(option_actor).reshape(
                self.num_worlds, 2, 8
            )
            selection = self.aerial_router.select(
                flat,
                kickoff_active=(lifecycle != 0).reshape(-1),
                match_done=(done != 0).reshape(-1),
            )
            self._capture_activations(
                selection.activated,
                selection.route,
                base_actions,
                option_actions,
            )
            self._activation_features.add(
                handoff_features(flat, option_actions.reshape(-1, 8)),
                mask=selection.activated,
                route=selection.route,
            )
            active = selection.active.reshape(self.num_worlds, 2)
            self.actions.copy_(
                torch.where(active[..., None], option_actions, base_actions)
            )
            self._before.copy_(observation)
            self._active.copy_(active)
            self._route_before.copy_(selection.route.reshape(self.num_worlds, 2))

    def corpus_payload(self) -> dict[str, Any]:
        if not self.capture_side:
            raise RuntimeError("natural handoff capture produced no activations")
        state = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self.capture_state.items()
        }
        lifecycle = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self.capture_lifecycle.items()
        }
        bridge_views = {
            name: torch.cat(chunks, dim=0).contiguous()
            for name, chunks in self.capture_views.items()
        }
        observation = torch.cat(self.capture_observation, dim=0).contiguous()
        base_action = torch.cat(self.capture_base_action, dim=0).contiguous()
        option_action = torch.cat(self.capture_option_action, dim=0).contiguous()
        world = torch.cat(self.capture_world).to(torch.int64)
        side = torch.cat(self.capture_side).to(torch.int64)
        route = torch.cat(self.capture_route).to(torch.int64)
        tick = torch.cat(self.capture_tick).to(torch.int64)
        count = int(side.numel())
        groups = {
            "state": state,
            "lifecycle": lifecycle,
            "bridge_views": {
                name: value.numpy() for name, value in bridge_views.items()
            },
            "samples": {
                "observation": observation.numpy(),
                "base_action": base_action.numpy(),
                "option_action": option_action.numpy(),
                "world": world.numpy(),
                "side": side.numpy(),
                "route": route.numpy(),
                "tick": tick.numpy(),
            },
        }
        return {
            "format": VERSION,
            "count": count,
            "state": state,
            "lifecycle": lifecycle,
            "bridge_views": bridge_views,
            "observation": observation,
            "base_action": base_action,
            "option_action": option_action,
            "source_world": world,
            "attacker_side": side,
            "route": route,
            "host_tick": tick,
            "semantic_sha256": _canonical_array_digest(groups),
        }


def run(args: argparse.Namespace) -> int:
    option = args.option.resolve()
    identities = {
        "blue_v23": sha256_file(BLUE),
        "orange_v23": sha256_file(ORANGE),
        "aerial_v3": sha256_file(option),
    }
    expected = {
        "blue_v23": BLUE_SHA256,
        "orange_v23": ORANGE_SHA256,
        "aerial_v3": OPTION_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"natural handoff capture identity mismatch: {identities}")
    runner = NaturalHandoffCaptureRunner(
        args.worlds,
        str(args.collision_root.resolve()),
        BLUE,
        starting_layout=np.arange(args.worlds, dtype=np.int32) % 5,
        rival_side=np.arange(args.worlds, dtype=np.int32) % 2,
        stochastic_rival=False,
        evaluation_seed=args.seed,
        orange_checkpoint=ORANGE,
        option_checkpoint=option,
        option_sha256=OPTION_SHA256,
        device=args.device,
    )
    timing = runner.run_ticks(args.ticks)
    payload = runner.corpus_payload()
    payload.update(
        {
            "sources": identities,
            "worlds": args.worlds,
            "ticks": args.ticks,
            "seed": args.seed,
            "timing_seconds": timing.seconds,
            "optimizer_steps": 0,
            "policy_mutation": False,
            "reward_changes": 0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    manifest = {
        "format": f"{VERSION}_MANIFEST",
        "corpus": {
            "path": args.output.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(args.output),
            "semantic_sha256": payload["semantic_sha256"],
            "bytes": args.output.stat().st_size,
        },
        "sources": identities,
        "worlds": args.worlds,
        "ticks": args.ticks,
        "seed": args.seed,
        "count": payload["count"],
        "attacker_side_counts": {
            str(side): int((payload["attacker_side"] == side).sum())
            for side in (0, 1)
        },
        "route_counts": {
            str(route): int((payload["route"] == route).sum())
            for route in torch.unique(payload["route"]).tolist()
        },
        "capture_tick_min": int(payload["host_tick"].min()),
        "capture_tick_max": int(payload["host_tick"].max()),
        "observation_shape": list(payload["observation"].shape),
        "bridge_view_count": len(payload["bridge_views"]),
        "state_fields": sorted(payload["state"]),
        "lifecycle_fields": sorted(payload["lifecycle"]),
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=Path, default=SELECTED)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2_026_090_327)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
