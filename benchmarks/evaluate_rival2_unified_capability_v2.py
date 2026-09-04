"""Direct physical evaluation of the single-network unified Rival V2 policy.

The policy controls every scenario directly.  This evaluator has no router,
teacher selection, expert action splice, or task/scenario input.  Specialist
checkpoints are evaluated only as frozen comparison controls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v2 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as aerial  # noqa: E402
from benchmarks import run_rival2_unified_capability_distillation_v1 as distill  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.kernels.rival2 import EPISODE_LIMIT_TICKS, REWARD_MODE_GAMEPLAY  # noqa: E402
from rivalsim.nexto_short_eval import (  # noqa: E402
    PHYSICS_HZ,
    RIVAL_CADENCE_TICKS,
    TERMINATION_GOAL,
    TERMINATION_HARD_TIME,
    TERMINATION_NO_TOUCH,
    ShortEvalTelemetry,
)
from rivalsim.open_play import TOUCH_FORWARD, OpenPlayTelemetry  # noqa: E402
from rivalsim.rival2_capability_curriculum_v2 import (  # noqa: E402
    SCENARIO_FLOOR_LANDING,
    SCENARIO_OFFENSIVE_DEMO,
    SCENARIO_WALL_LANDING,
)
from rivalsim.rival2_contracts import ACTION_NAMES  # noqa: E402
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic  # noqa: E402
from rivalsim.rival2_unified_policy import (  # noqa: E402
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors  # noqa: E402

FORMAT = "RIVAL2_UNIFIED_CAPABILITY_V2_PHYSICAL_EVALUATION"
CHECKPOINT = (
    ROOT / "checkpoints/rival2/unified_capability_distillation_v2/rival2_unified_capability_v2.pt"
)
AUTHORITY = ROOT / "results/rival2/unified_capability_distillation_v2/authority.json"
OUTPUT = ROOT / "results/rival2/unified_capability_distillation_v2/physical_evaluation.json"
COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def report_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_unified(path: Path, device: str) -> tuple[dict[str, Any], Rival2UnifiedActorCritic]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in {
        "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V2",
        "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V3",
        "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V4",
        "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V5",
        "RIVAL2_UNIFIED_GROUND_CURRICULUM_PPO_V2_CHECKPOINT",
    }:
        raise RuntimeError("unsupported unified checkpoint format")
    config = Rival2UnifiedPolicyConfig(**payload["policy_config"])
    if payload.get("policy_config_sha256") != config.content_hash:
        raise RuntimeError("unified policy config hash mismatch")
    model = Rival2UnifiedActorCritic(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return payload, model


class StatefulUnifiedAdapter:
    """Expose the feed-forward evaluator API while retaining one GRU state."""

    def __init__(self, policy: Rival2UnifiedActorCritic):
        self.policy = policy
        self.config = policy.config.base_policy_config
        self.hidden: torch.Tensor | None = None

    def eval(self) -> StatefulUnifiedAdapter:
        self.policy.eval()
        return self

    def __call__(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden is None or self.hidden.shape[1] != observation.shape[0]:
            self.hidden = self.policy.initial_hidden(
                observation.shape[0], device=observation.device, dtype=observation.dtype
            )
        actor, value, self.hidden = self.policy(observation, self.hidden)
        return actor, value


def aerial_evaluation(
    policy: Rival2UnifiedActorCritic,
    teacher: Rival2ActorCritic,
    *,
    authority: dict[str, Any],
    collision_root: Path,
    worlds_per_side: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    goal_authority = aerial.load_authority()
    geometry = ArenaGeometry.load_soccar(collision_root / "soccar")
    meshes = WarpArenaMeshes(geometry, device)
    distribution = aerial.distribution_override(goal_authority)
    rows: dict[str, list[dict[str, Any]]] = {"unified": [], "specialist_control": []}
    for side in (0, 1):
        unified_adapter = StatefulUnifiedAdapter(policy)
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xA500 + side))
        _rollout, metrics = aerial.collect_rollout(
            unified_adapter,
            geometry,
            meshes,
            authority=goal_authority,
            side=side,
            worlds=worlds_per_side,
            horizon=int(authority["corpora"]["validation"]["aerial_horizon_ticks"]),
            seed=seed,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
            collision_dir=collision_root / "soccar",
            phase=1,
        )
        rows["unified"].append(metrics)
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xA500 + side))
        _rollout, metrics = aerial.collect_rollout(
            teacher,
            geometry,
            meshes,
            authority=goal_authority,
            side=side,
            worlds=worlds_per_side,
            horizon=int(authority["corpora"]["validation"]["aerial_horizon_ticks"]),
            seed=seed,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
            collision_dir=collision_root / "soccar",
            phase=1,
        )
        rows["specialist_control"].append(metrics)
    acceptance = goal_authority["acceptance"]
    for _label, values in rows.items():
        aggregate: dict[str, float] = {}
        for name in values[0]["fractions"]:
            aggregate[name] = float(np.mean([row["fractions"][name] for row in values]))
        values.append({"aggregate_fraction": aggregate})
        values[-1]["passes_frozen_aerial_thresholds"] = bool(
            aggregate["pop_touch"] >= acceptance["pop_touch_fraction_min"]
            and aggregate["elevated_follow_touch"]
            >= acceptance["elevated_follow_touch_fraction_min"]
            and aggregate["high_follow_touch"] >= acceptance["high_follow_touch_fraction_min"]
            and aggregate["second_airborne_touch"]
            >= acceptance["second_airborne_touch_fraction_min"]
            and aggregate["productive_continuation"]
            >= acceptance["productive_continuation_fraction_min"]
            and aggregate["goal_within_contact_budget"]
            >= acceptance["goal_within_contact_budget_fraction_min"]
            and aggregate["contact_budget_exceeded"]
            <= acceptance["contact_budget_exceeded_fraction_max"]
            and aggregate["unassisted_or_ground_goal"]
            <= acceptance["unassisted_or_ground_goal_fraction_max"]
        )
    return rows


def capability_evaluation(
    policy: Rival2UnifiedActorCritic,
    teachers: distill.FrozenTeachers,
    *,
    collision_root: Path,
    worlds_per_side: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    geometry = ArenaGeometry.load_soccar(collision_root / "soccar")
    meshes = WarpArenaMeshes(geometry, device)
    distribution = capability.HybridDistributionOverride(
        analog_log_std=np.log(0.08), button_temperature=1.5
    )
    rows: dict[str, list[dict[str, Any]]] = {"unified": [], "specialist_control": []}
    for side, teacher in enumerate((teachers.capability_blue, teachers.capability_orange)):
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xC500 + side))
        _rollout, metrics = capability.collect_scenario_rollout(
            StatefulUnifiedAdapter(policy),
            geometry,
            meshes,
            side=side,
            collision_dir=collision_root / "soccar",
            worlds=worlds_per_side,
            horizon=256,
            seed=seed ^ side,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
        )
        rows["unified"].append(metrics)
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xC500 + side))
        _rollout, metrics = capability.collect_scenario_rollout(
            teacher,
            geometry,
            meshes,
            side=side,
            collision_dir=collision_root / "soccar",
            worlds=worlds_per_side,
            horizon=256,
            seed=seed ^ side,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
        )
        rows["specialist_control"].append(metrics)
    scenario_by_name = {
        "offensive_demo_v2": SCENARIO_OFFENSIVE_DEMO,
        "floor_landing_v2": SCENARIO_FLOOR_LANDING,
        "wall_landing_v2": SCENARIO_WALL_LANDING,
    }
    for values in rows.values():
        aggregate: dict[str, Any] = {}
        for key in values[0]["telemetry"]:
            aggregate[key] = sum(
                int(row["telemetry"][key])
                if isinstance(row["telemetry"][key], int)
                else float(row["telemetry"][key])
                for row in values
            )
        aggregate["scenario_ids"] = scenario_by_name
        values.append({"aggregate_telemetry": aggregate})
    return rows


class UnifiedNextoRunner:
    """Bounded deterministic Nexto evaluation for one unified recurrent policy."""

    def __init__(
        self,
        num_worlds: int,
        collision_root: Path,
        policy: Rival2UnifiedActorCritic,
        *,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
        seed: int,
        device: str,
    ):
        self.num_worlds = num_worlds
        self.device = torch.device(device)
        capability.activate_fresh_persistent_stream(self.device)
        self.world = Rival2WorldSim(
            num_worlds,
            str(collision_root),
            device=device,
            seed=seed,
            kickoff_selector=starting_layout,
            car_lifecycle_seed=seed,
            reward_mode=REWARD_MODE_GAMEPLAY,
        )
        self.warp_stream = wp.get_stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)
        self.bridge = Rival2TensorBridge(self.world)
        self.policy = policy
        self.rival_side = torch.as_tensor(rival_side, dtype=torch.long, device=device)
        self.nexto_side = 1 - self.rival_side
        self.batch = torch.arange(num_worlds, device=device)
        self.nexto = NextoPolicyAdapter(num_worlds, device=device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.observation = self.bridge.observation()
        self.hidden = policy.initial_hidden(num_worlds, device=device)
        self.rival_action = torch.zeros((num_worlds, 8), device=device)
        self.actions = torch.zeros((num_worlds, 2, 8), device=device)
        self.telemetry = ShortEvalTelemetry(self.world)
        self.telemetry.attach()
        self.open_play = OpenPlayTelemetry(self.world)
        self.open_play.attach(self.world)
        self.host_tick = 0
        self.last_reset_mask = torch.zeros(num_worlds, dtype=torch.bool, device=self.device)
        self.world.reset_transfer_counters()
        self.world.capture_graph(block_ticks=1)

    @torch.inference_mode()
    def initial_action_probe(self) -> np.ndarray:
        observation = self.observation[self.batch, self.rival_side]
        hidden = self.policy.initial_hidden(self.num_worlds, device=self.device)
        actor, _value, _hidden = self.policy(observation, hidden)
        return deterministic_unified_action(actor).cpu().numpy()

    @torch.inference_mode()
    def tick(self) -> None:
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)
        actor, _value, self.hidden = self.policy(
            self.observation[self.batch, self.rival_side], self.hidden
        )
        self.rival_action.copy_(deterministic_unified_action(actor))
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            self.world.begin_decision()
        kickoff = self.bridge.views["rival2.kickoff_indicator"] != 0
        nexto_action, _indices = self.nexto.tick_action(self.nexto_state, kickoff)
        self.actions[self.batch, self.rival_side] = self.rival_action
        self.actions[self.batch, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.host_tick += 1
        self.last_reset_mask.zero_()
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            reset = self.bridge.views["rival2.reset_mask"].to(torch.bool)
            self.last_reset_mask.copy_(reset)
            self.nexto.notify_kickoff(reset)
            self.world.apply_interval_resets()
            if bool(torch.any(reset)):
                self.hidden[:, reset] = 0.0
        self.observation = self.bridge.observation()

    def run(self) -> dict[str, Any]:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _tick in range(EPISODE_LIMIT_TICKS):
            self.tick()
        torch.cuda.synchronize(self.device)
        return {
            "seconds": time.perf_counter() - started,
            "raw": self.telemetry.numpy(),
            "open_raw": self.open_play.numpy(),
        }


def nexto_evaluation(
    policy: Rival2UnifiedActorCritic,
    *,
    collision_root: Path,
    worlds_per_side: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    side = np.concatenate(
        (np.zeros(worlds_per_side, dtype=np.int32), np.ones(worlds_per_side, dtype=np.int32))
    )
    layout = np.tile(np.arange(5, dtype=np.int32), int(np.ceil(side.size / 5)))[: side.size]
    runner = UnifiedNextoRunner(
        side.size,
        collision_root,
        policy,
        starting_layout=layout,
        rival_side=side,
        seed=seed,
        device=device,
    )
    initial = runner.initial_action_probe()
    initial_vectors = []
    available_blue_layouts = sorted(set(layout[side == 0].tolist()))
    for layout_index in available_blue_layouts:
        row = next(
            index
            for index in range(side.size)
            if side[index] == 0 and layout[index] == layout_index
        )
        initial_vectors.append(
            {
                "layout": layout_index,
                "controller_output": {
                    name: float(initial[row, channel]) for channel, name in enumerate(ACTION_NAMES)
                },
            }
        )
    run = runner.run()
    raw = run["raw"]
    open_raw = run["open_raw"]
    rows = np.arange(side.size, dtype=np.int64)
    opponent = 1 - side
    rival_touch = raw["touch_count"][rows, side]
    nexto_touch = raw["touch_count"][rows, opponent]
    termination = raw["termination_kind"]
    winner = raw["winner"]
    goal = termination == TERMINATION_GOAL
    episodes = int(side.size)
    behavior = np.flatnonzero(termination != TERMINATION_NO_TOUCH)
    behavior_side = side[behavior]
    ticks = int(raw["simulated_ticks"][rows, side].sum())
    return {
        "episodes": episodes,
        "five_initial_kickoff_actions": initial_vectors,
        "rival_touches": int(rival_touch.sum()),
        "nexto_touches": int(nexto_touch.sum()),
        "episodes_with_rival_touch": int((rival_touch > 0).sum()),
        "episodes_with_rival_touch_fraction": float((rival_touch > 0).mean()),
        "rival_goals": int((goal & (winner == side)).sum()),
        "nexto_goals": int((goal & (winner == opponent)).sum()),
        "no_touch_truncations": int((termination == TERMINATION_NO_TOUCH).sum()),
        "hard_timeouts": int((termination == TERMINATION_HARD_TIME).sum()),
        "forward_contacts": int(
            open_raw["direction_count"][behavior, behavior_side, TOUCH_FORWARD].sum()
        ),
        "demos": None,
        "demo_note": "short-episode telemetry does not expose demolition events",
        "mean_speed_uu_per_second": (
            None
            if ticks == 0
            else float(raw["speed_sum"][rows, side].sum(dtype=np.float64) / ticks)
        ),
        "wall_seconds": run["seconds"],
    }


def run(args: argparse.Namespace) -> int:
    args.checkpoint = args.checkpoint.resolve()
    args.authority = args.authority.resolve()
    args.output = args.output.resolve()
    args.collision_root = args.collision_root.resolve()
    authority = json.loads(args.authority.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(args.checkpoint)
    payload, policy = load_unified(args.checkpoint, args.device)
    _payloads, teachers = distill.load_teachers(authority, args.device)
    aerial_rows = aerial_evaluation(
        policy,
        teachers.aerial,
        authority=authority,
        collision_root=args.collision_root,
        worlds_per_side=args.aerial_worlds_per_side,
        seed=int(
            authority["seeds"].get(
                "physical_aerial", authority["seeds"].get("untouched_test_aerial")
            )
        ),
        device=args.device,
    )
    capability_rows = capability_evaluation(
        policy,
        teachers,
        collision_root=args.collision_root,
        worlds_per_side=args.capability_worlds_per_side,
        seed=int(
            authority["seeds"].get(
                "physical_capability",
                authority["seeds"].get("untouched_test_capability"),
            )
        ),
        device=args.device,
    )
    nexto_rows = nexto_evaluation(
        policy,
        collision_root=args.collision_root,
        worlds_per_side=args.nexto_worlds_per_side,
        seed=int(
            authority["seeds"].get(
                "physical_nexto", authority["seeds"].get("untouched_test_natural")
            )
        ),
        device=args.device,
    )
    result = {
        "format": FORMAT,
        "created_utc": utc_now(),
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
        "runtime_router": False,
        "task_identifier_input": False,
        "checkpoint": {
            "path": report_path(args.checkpoint),
            "sha256": checkpoint_sha,
            "selected_step": (
                None
                if payload.get("accepted_supervised_steps") is None
                else int(payload["accepted_supervised_steps"])
            ),
            "accepted_updates_total": (
                None
                if payload.get("accepted_updates_total") is None
                else int(payload["accepted_updates_total"])
            ),
        },
        "execution": {
            "deterministic": True,
            "physics_hz": PHYSICS_HZ,
            "policy_hz": PHYSICS_HZ,
            "hidden_reset_at_episode_boundary": True,
        },
        "controlled_aerial": aerial_rows,
        "controlled_capability": capability_rows,
        "natural_nexto": nexto_rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    del policy, teachers
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aerial-worlds-per-side", type=int, default=1024)
    parser.add_argument("--capability-worlds-per-side", type=int, default=2048)
    parser.add_argument("--nexto-worlds-per-side", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
