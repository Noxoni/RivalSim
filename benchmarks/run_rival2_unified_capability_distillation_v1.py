"""Train one recurrent Rival policy from the frozen V23 and capability teachers.

Teacher identity is used only to construct supervised targets.  The deployed
student receives no teacher, scenario, expert, or route identifier and emits one
action directly from one recurrent network.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v2 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as aerial  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.full_match import FullMatchRunner  # noqa: E402
from rivalsim.rival2_capability_curriculum_v2 import (  # noqa: E402
    SCENARIO_FLOOR_LANDING,
    SCENARIO_OFFENSIVE_DEMO,
    SCENARIO_WALL_LANDING,
    build_capability_scenarios_v2,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)
from rivalsim.rival2_unified_policy import (  # noqa: E402
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)

AUTHORITY = ROOT / "results/rival2/unified_capability_distillation_v1/authority.json"
RESULTS = ROOT / "results/rival2/unified_capability_distillation_v1"
CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/unified_capability_distillation_v1/rival2_unified_capability_v1.pt"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/unified-capability-distillation-v1")
DEFAULT_COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes")

FAMILY_NATURAL = "natural_v23"
FAMILY_AERIAL = "aerial_v3"
FAMILY_DEMO = "offensive_demo_v2"
FAMILY_FLOOR = "floor_landing_v2"
FAMILY_WALL = "wall_landing_v2"
FAMILIES = (FAMILY_NATURAL, FAMILY_AERIAL, FAMILY_DEMO, FAMILY_FLOOR, FAMILY_WALL)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tensor_tree_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        local = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(np.asarray(local.shape, dtype=np.int64).tobytes())
        digest.update(local.numpy().tobytes())
    return digest.hexdigest().upper()


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V1_AUTHORITY":
        raise RuntimeError("unified capability authority format mismatch")
    if authority["integrity"]["optimizer_steps_before_authority_commit"] != 0:
        raise RuntimeError("authority does not preserve the prospective boundary")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{AUTHORITY.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != AUTHORITY.read_bytes():
        raise RuntimeError("authority is not byte-identical to the committed Git object")
    for source in authority["sources"].values():
        path = ROOT / source["path"]
        observed = sha256_file(path)
        if observed != source["sha256"]:
            raise RuntimeError(f"source hash mismatch for {path}: {observed}")
    return authority


def load_feedforward(path: Path, device: str) -> tuple[dict[str, Any], Rival2ActorCritic]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = Rival2PolicyConfig(**payload["policy_config"])
    model = Rival2ActorCritic(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return payload, model


@dataclass(slots=True)
class FrozenTeachers:
    base_blue: Rival2ActorCritic
    base_orange: Rival2ActorCritic
    aerial: Rival2ActorCritic
    capability_blue: Rival2ActorCritic
    capability_orange: Rival2ActorCritic


def load_teachers(authority: dict[str, Any], device: str) -> tuple[dict[str, Any], FrozenTeachers]:
    payloads: dict[str, Any] = {}
    models: dict[str, Rival2ActorCritic] = {}
    for name, identity in authority["sources"].items():
        payloads[name], models[name] = load_feedforward(ROOT / identity["path"], device)
    reference = models["base_blue"].config
    if any(model.config != reference for model in models.values()):
        raise RuntimeError("teacher policy architectures differ")
    contracts = payloads["base_blue"]["contract_hashes"]
    if any(payload.get("contract_hashes") != contracts for payload in payloads.values()):
        raise RuntimeError("teacher policy contracts differ")
    return payloads, FrozenTeachers(
        base_blue=models["base_blue"],
        base_orange=models["base_orange"],
        aerial=models["aerial_teacher"],
        capability_blue=models["capability_blue"],
        capability_orange=models["capability_orange"],
    )


def corpus_path(run_dir: Path, split: str, kind: str) -> Path:
    return run_dir / "corpora" / split / f"{kind}.pt"


def save_corpus(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "observation_shape": list(payload["observation"].shape),
        "valid_samples": int(payload["valid"].sum()),
    }


def collect_natural_corpus(
    *,
    checkpoint: Path,
    collision_root: Path,
    worlds: int,
    ticks: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    layout = np.resize(np.arange(5, dtype=np.int32), worlds)
    side = np.resize(np.asarray((0, 1), dtype=np.int32), worlds)
    rng = np.random.default_rng(seed)
    rng.shuffle(layout)
    rng.shuffle(side)
    capability.activate_fresh_persistent_stream(device)
    runner = FullMatchRunner(
        worlds,
        str(collision_root),
        checkpoint,
        starting_layout=layout,
        rival_side=side,
        stochastic_rival=False,
        evaluation_seed=seed,
        device=device,
    )
    observation = torch.empty((ticks, worlds, 182), dtype=torch.float32)
    valid = torch.ones((ticks, worlds), dtype=torch.bool)
    for tick in range(ticks):
        observation[tick].copy_(
            runner.rival_observation[runner.batch_index, runner.rival_side].detach().cpu()
        )
        runner.tick()
    capability.release_full_match_runner(runner)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "format": "RIVAL2_UNIFIED_SEQUENCE_CORPUS_V1",
        "kind": "natural",
        "seed": seed,
        "observation": observation,
        "valid": valid,
        "side": torch.from_numpy(side.astype(np.int64)),
    }


def collect_aerial_corpus(
    *,
    teacher: Rival2ActorCritic,
    authority: dict[str, Any],
    collision_dir: Path,
    worlds_per_side: int,
    horizon: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    observations: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    sides: list[torch.Tensor] = []
    metrics: list[dict[str, Any]] = []
    distribution = aerial.distribution_override(authority)
    for side in (0, 1):
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xA000 + side))
        rollout, row = aerial.collect_rollout(
            teacher,
            geometry,
            meshes,
            authority=authority,
            side=side,
            worlds=worlds_per_side,
            horizon=horizon,
            seed=seed,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=False,
            collision_dir=collision_dir,
            phase=1,
        )
        assert rollout is not None
        observations.append(rollout.observation.detach().cpu())
        valid.append(rollout.mask.detach().cpu())
        rewards.append(rollout.reward.detach().cpu())
        sides.append(torch.full((worlds_per_side,), side, dtype=torch.int64))
        metrics.append(row)
        del rollout
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "format": "RIVAL2_UNIFIED_SEQUENCE_CORPUS_V1",
        "kind": "aerial",
        "seed": seed,
        "observation": torch.cat(observations, dim=1),
        "valid": torch.cat(valid, dim=1),
        "reward": torch.cat(rewards, dim=1),
        "side": torch.cat(sides),
        "source_metrics": metrics,
    }


def collect_capability_corpus(
    *,
    teachers: FrozenTeachers,
    collision_dir: Path,
    worlds_per_side: int,
    horizon: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    observations: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    scenarios: list[torch.Tensor] = []
    sides: list[torch.Tensor] = []
    metrics: list[dict[str, Any]] = []
    distribution = HybridDistributionOverride(
        analog_log_std=math.log(0.08), button_temperature=1.5
    )
    for side, teacher in enumerate((teachers.capability_blue, teachers.capability_orange)):
        local_seed = seed ^ side
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xC000 + side))
        rollout, row = capability.collect_scenario_rollout(
            teacher,
            geometry,
            meshes,
            side=side,
            collision_dir=collision_dir,
            worlds=worlds_per_side,
            horizon=horizon,
            seed=local_seed,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=False,
        )
        assert rollout is not None
        batch = build_capability_scenarios_v2(
            worlds_per_side, seed=local_seed, attacker_side=side
        )
        observations.append(rollout.observations[:, :, side].detach().cpu())
        valid.append(rollout.train_mask[:, :, side].detach().cpu())
        rewards.append(rollout.rewards[:, :, side].detach().cpu())
        scenarios.append(torch.from_numpy(batch.scenario.astype(np.int64)))
        sides.append(torch.full((worlds_per_side,), side, dtype=torch.int64))
        metrics.append(row)
        del rollout
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "format": "RIVAL2_UNIFIED_SEQUENCE_CORPUS_V1",
        "kind": "capability",
        "seed": seed,
        "observation": torch.cat(observations, dim=1),
        "valid": torch.cat(valid, dim=1),
        "reward": torch.cat(rewards, dim=1),
        "scenario": torch.cat(scenarios),
        "side": torch.cat(sides),
        "source_metrics": metrics,
    }


def build_corpora(
    run_dir: Path,
    authority: dict[str, Any],
    teachers: FrozenTeachers,
    *,
    collision_root: Path,
    device: str,
) -> dict[str, Any]:
    aerial_authority = aerial.load_authority()
    manifest: dict[str, Any] = {
        "format": "RIVAL2_UNIFIED_CORPUS_MANIFEST_V1",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY),
        "splits": {},
    }
    for split in ("train", "validation"):
        spec = authority["corpora"][split]
        seeds = authority["seeds"]
        rows: dict[str, Any] = {}
        natural_file = corpus_path(run_dir, split, "natural")
        natural = collect_natural_corpus(
            checkpoint=ROOT / authority["sources"]["base_blue"]["path"],
            collision_root=collision_root,
            worlds=int(spec["natural_worlds"]),
            ticks=int(spec["natural_ticks"]),
            seed=int(seeds[f"{split}_natural"]),
            device=device,
        )
        rows["natural"] = save_corpus(natural_file, natural)
        del natural

        aerial_file = corpus_path(run_dir, split, "aerial")
        aerial_corpus = collect_aerial_corpus(
            teacher=teachers.aerial,
            authority=aerial_authority,
            collision_dir=collision_root / "soccar",
            worlds_per_side=int(spec["aerial_worlds_per_side"]),
            horizon=int(spec["aerial_horizon_ticks"]),
            seed=int(seeds[f"{split}_aerial"]),
            device=device,
        )
        rows["aerial"] = save_corpus(aerial_file, aerial_corpus)
        del aerial_corpus

        capability_file = corpus_path(run_dir, split, "capability")
        capability_corpus = collect_capability_corpus(
            teachers=teachers,
            collision_dir=collision_root / "soccar",
            worlds_per_side=int(spec["capability_worlds_per_side"]),
            horizon=int(spec["capability_horizon_ticks"]),
            seed=int(seeds[f"{split}_capability"]),
            device=device,
        )
        rows["capability"] = save_corpus(capability_file, capability_corpus)
        del capability_corpus
        manifest["splits"][split] = rows
    write_json(run_dir / "corpora" / "manifest.json", manifest)
    write_json(RESULTS / "corpus_manifest.json", manifest)
    return manifest


@dataclass(slots=True)
class SequencePool:
    observation: torch.Tensor
    side: torch.Tensor
    candidates: torch.Tensor
    sequence_ticks: int

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        sequence_ticks: int,
        scenario: int | None = None,
    ) -> SequencePool:
        observation = payload["observation"]
        valid = payload["valid"].to(torch.bool)
        windows = valid.unfold(0, sequence_ticks, 1).all(dim=-1)
        if scenario is not None:
            selected_world = payload["scenario"] == scenario
            windows &= selected_world.unsqueeze(0)
        candidates = torch.nonzero(windows, as_tuple=False).to(torch.int64)
        if not candidates.numel():
            raise RuntimeError(f"no valid sequence windows for scenario {scenario}")
        return cls(observation, payload["side"].to(torch.int64), candidates, sequence_ticks)

    def sample(
        self, count: int, *, generator: torch.Generator, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = torch.randint(
            self.candidates.shape[0], (count,), generator=generator
        )
        pair = self.candidates.index_select(0, selected)
        ticks = pair[:, :1] + torch.arange(self.sequence_ticks, dtype=torch.int64)
        worlds = pair[:, 1:2].expand(-1, self.sequence_ticks)
        observation = self.observation[ticks, worlds]
        side = self.side.index_select(0, pair[:, 1])
        return observation.to(device), side.to(device)


def load_pools(run_dir: Path, split: str, sequence_ticks: int) -> dict[str, SequencePool]:
    natural = torch.load(corpus_path(run_dir, split, "natural"), map_location="cpu")
    aerial_payload = torch.load(corpus_path(run_dir, split, "aerial"), map_location="cpu")
    capability_payload = torch.load(
        corpus_path(run_dir, split, "capability"), map_location="cpu"
    )
    return {
        FAMILY_NATURAL: SequencePool.from_payload(
            natural, sequence_ticks=sequence_ticks
        ),
        FAMILY_AERIAL: SequencePool.from_payload(
            aerial_payload, sequence_ticks=sequence_ticks
        ),
        FAMILY_DEMO: SequencePool.from_payload(
            capability_payload,
            sequence_ticks=sequence_ticks,
            scenario=SCENARIO_OFFENSIVE_DEMO,
        ),
        FAMILY_FLOOR: SequencePool.from_payload(
            capability_payload,
            sequence_ticks=sequence_ticks,
            scenario=SCENARIO_FLOOR_LANDING,
        ),
        FAMILY_WALL: SequencePool.from_payload(
            capability_payload,
            sequence_ticks=sequence_ticks,
            scenario=SCENARIO_WALL_LANDING,
        ),
    }


def teacher_actor(
    teachers: FrozenTeachers,
    family: str,
    observation: torch.Tensor,
    side: torch.Tensor,
) -> torch.Tensor:
    flat = observation.reshape(-1, 182)
    flat_side = side[:, None].expand(-1, observation.shape[1]).reshape(-1)
    with torch.no_grad():
        if family == FAMILY_AERIAL:
            output, _value = teachers.aerial(flat)
        else:
            if family == FAMILY_NATURAL:
                blue, orange = teachers.base_blue, teachers.base_orange
            else:
                blue, orange = teachers.capability_blue, teachers.capability_orange
            blue_output, _ = blue(flat)
            orange_output, _ = orange(flat)
            output = torch.where(flat_side[:, None] == 1, orange_output, blue_output)
    return output.reshape(observation.shape[0], observation.shape[1], 13)


def expected_action(actor: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.tanh(actor[..., :5]), torch.sigmoid(actor[..., 10:13])), dim=-1)


def family_loss(
    student_actor: torch.Tensor,
    teacher_output: torch.Tensor,
    *,
    burn_in: int,
    button_weight: float,
    log_std_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    student = student_actor[:, burn_in:]
    teacher = teacher_output[:, burn_in:]
    analog = F.smooth_l1_loss(
        torch.tanh(student[..., :5]), torch.tanh(teacher[..., :5]), beta=0.1
    )
    buttons = F.binary_cross_entropy_with_logits(
        student[..., 10:13], torch.sigmoid(teacher[..., 10:13])
    )
    log_std = F.smooth_l1_loss(student[..., 5:10], teacher[..., 5:10], beta=0.1)
    loss = analog + button_weight * buttons + log_std_weight * log_std
    rmse = torch.sqrt(F.mse_loss(expected_action(student), expected_action(teacher)))
    return loss, {
        "loss": float(loss.detach()),
        "analog_loss": float(analog.detach()),
        "button_loss": float(buttons.detach()),
        "log_std_loss": float(log_std.detach()),
        "expected_action_rmse": float(rmse.detach()),
    }


@torch.no_grad()
def validate(
    model: Rival2UnifiedActorCritic,
    teachers: FrozenTeachers,
    pools: dict[str, SequencePool],
    *,
    authority: dict[str, Any],
    device: str,
    samples_per_family: int = 256,
) -> dict[str, Any]:
    optimization = authority["optimization"]
    generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"]) ^ 0x5151
    )
    metrics: dict[str, Any] = {}
    for family in FAMILIES:
        observation, side = pools[family].sample(
            samples_per_family, generator=generator, device=device
        )
        actor, _value, _hidden = model(observation)
        target = teacher_actor(teachers, family, observation, side)
        _loss, row = family_loss(
            actor,
            target,
            burn_in=int(authority["corpora"]["burn_in_ticks"]),
            button_weight=float(optimization["button_weight"]),
            log_std_weight=float(optimization["log_std_weight"]),
        )
        metrics[family] = row
    weights = optimization["family_weights"]
    metrics["weighted_expected_action_rmse"] = sum(
        float(weights[family]) * metrics[family]["expected_action_rmse"]
        for family in FAMILIES
    )
    metrics["finite"] = all(
        math.isfinite(float(metrics[family]["loss"])) for family in FAMILIES
    )
    return metrics


def eligible(
    metrics: dict[str, Any], baseline: dict[str, Any], authority: dict[str, Any]
) -> bool:
    if not metrics["finite"]:
        return False
    if metrics[FAMILY_NATURAL]["expected_action_rmse"] > float(
        authority["selection"]["natural_expected_action_rmse_max"]
    ):
        return False
    return all(
        metrics[family]["expected_action_rmse"]
        < baseline[family]["expected_action_rmse"]
        for family in FAMILIES
        if family != FAMILY_NATURAL
    )


def save_checkpoint(
    model: Rival2UnifiedActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    authority: dict[str, Any],
    source_payload: dict[str, Any],
    step: int,
    validation: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    model_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V1",
        "created_utc": utc_now(),
        "model": model_state,
        "policy_config": asdict(model.config),
        "policy_config_sha256": model.config.content_hash,
        "optimizer": {
            "format": "RIVAL2_UNIFIED_CONTEXT_ONLY_ADAMW_V1",
            "state": optimizer.state_dict(),
        },
        "accepted_supervised_steps": step,
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY),
        },
        "sources": copy.deepcopy(authority["sources"]),
        "contracts": copy.deepcopy(authority["contracts"]),
        "observation_version": source_payload["observation_version"],
        "action_version": source_payload["action_version"],
        "reward_version": source_payload["reward_version"],
        "episode_version": source_payload["episode_version"],
        "contract_hashes": copy.deepcopy(source_payload["contract_hashes"]),
        "physics_hz": 120,
        "policy_hz": 120,
        "validation": validation,
        "baseline_validation": baseline,
        "runtime_router": False,
        "task_identifier_input": False,
        "ppo_resumable": False,
        "base_model_tensor_sha256": tensor_tree_sha256(source_payload["model"]),
        "model_tensor_sha256": tensor_tree_sha256(model_state),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CHECKPOINT)
    return {
        "path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(CHECKPOINT),
        "bytes": CHECKPOINT.stat().st_size,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "accepted_supervised_steps": step,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    payloads, teachers = load_teachers(authority, args.device)
    base_payload = payloads["base_blue"]
    base = Rival2ActorCritic(teachers.base_blue.config)
    base.load_state_dict(base_payload["model"], strict=True)
    model = Rival2UnifiedActorCritic(Rival2UnifiedPolicyConfig()).to(args.device)
    model.load_feedforward_parent(base)
    model.freeze_base()
    model.train()

    generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"])
    )
    parity_observation = torch.randn((4096, 182), generator=generator).to(args.device)
    with torch.no_grad():
        parent_actor, parent_value = teachers.base_blue(parity_observation)
        actor, value, _hidden = model(parity_observation)
    parity = {
        "actor_exact": bool(torch.equal(actor, parent_actor)),
        "value_exact": bool(torch.equal(value, parent_value)),
        "action_exact": bool(
            torch.equal(
                deterministic_unified_action(actor),
                deterministic_hybrid_action(parent_actor),
            )
        ),
    }
    if not all(parity.values()):
        raise RuntimeError(f"zero-residual parent parity failed: {parity}")
    preflight = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V1_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY),
        "source_hashes_verified": True,
        "contracts_identical": True,
        "runtime_router": False,
        "task_identifier_input": False,
        "parent_parity": parity,
        "base_tensor_sha256": tensor_tree_sha256(base_payload["model"]),
        "optimizer_steps": 0,
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_corpora:
        raise RuntimeError("unified distillation requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_corpora:
        build_corpora(
            run_dir,
            authority,
            teachers,
            collision_root=args.collision_root,
            device=args.device,
        )
    sequence_ticks = int(authority["corpora"]["sequence_ticks"])
    train_pools = load_pools(run_dir, "train", sequence_ticks)
    validation_pools = load_pools(run_dir, "validation", sequence_ticks)

    optimizer_config = authority["optimization"]
    optimizer = torch.optim.AdamW(
        model.context_parameters,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    baseline = validate(
        model, teachers, validation_pools, authority=authority, device=args.device
    )
    write_json(RESULTS / "baseline.json", baseline)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = float("inf")
    best_step = 0
    stale = 0
    stop_reason = "maximum_accepted_steps"
    batch_count = int(optimizer_config["batch_sequences_per_family"])
    weights = optimizer_config["family_weights"]
    maximum_steps = min(int(optimizer_config["maximum_accepted_steps"]), args.max_steps)
    interval = int(optimizer_config["validation_interval_steps"])
    train_generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"]) ^ 0xA5A5
    )
    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=args.device)
        train_metrics: dict[str, Any] = {}
        for family in FAMILIES:
            observation, side = train_pools[family].sample(
                batch_count, generator=train_generator, device=args.device
            )
            student, _value, _hidden = model(observation)
            target = teacher_actor(teachers, family, observation, side)
            local, row = family_loss(
                student,
                target,
                burn_in=int(authority["corpora"]["burn_in_ticks"]),
                button_weight=float(optimizer_config["button_weight"]),
                log_std_weight=float(optimizer_config["log_std_weight"]),
            )
            total = total + float(weights[family]) * local
            train_metrics[family] = row
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.context_parameters, 0.5)
        if not bool(torch.isfinite(total) and torch.isfinite(gradient)):
            raise RuntimeError("nonfinite unified distillation loss or gradient")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.context_parameters
        ):
            raise RuntimeError("nonfinite unified recurrent context parameter")
        if step % interval != 0 and step != maximum_steps:
            continue
        model.eval()
        validation = validate(
            model, teachers, validation_pools, authority=authority, device=args.device
        )
        model.train()
        score = float(validation["weighted_expected_action_rmse"])
        is_eligible = eligible(validation, baseline, authority)
        improved = is_eligible and score < (
            best_score - float(optimizer_config["minimum_score_improvement"])
        )
        if improved:
            best_score = score
            best_step = step
            best_metrics = copy.deepcopy(validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            stale = 0
        else:
            stale += 1
        append_jsonl(
            curve,
            {
                "step": step,
                "created_utc": utc_now(),
                "train_loss": float(total.detach()),
                "gradient_norm": float(gradient.detach()),
                "train": train_metrics,
                "validation": validation,
                "eligible": is_eligible,
                "selected": improved,
                "stale_boundaries": stale,
            },
        )
        natural_rmse = validation[FAMILY_NATURAL]["expected_action_rmse"]
        print(
            f"step={step} score={score:.6f} natural={natural_rmse:.6f} "
            f"aerial={validation[FAMILY_AERIAL]['expected_action_rmse']:.6f} "
            f"demo={validation[FAMILY_DEMO]['expected_action_rmse']:.6f} "
            f"floor={validation[FAMILY_FLOOR]['expected_action_rmse']:.6f} "
            f"wall={validation[FAMILY_WALL]['expected_action_rmse']:.6f} "
            f"eligible={is_eligible} selected={improved}",
            flush=True,
        )
        if stale >= int(optimizer_config["plateau_patience_boundaries"]):
            stop_reason = "validation_plateau"
            break
    if best_state is None or best_optimizer is None or best_metrics is None:
        result = {
            "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V1_RESULT",
            "created_utc": utc_now(),
            "verdict": "BLOCKED",
            "reason": "no validation-eligible unified checkpoint",
            "baseline": baseline,
            "stop_reason": stop_reason,
        }
        write_json(RESULTS / "result.json", result)
        return 2
    model.load_state_dict(best_state, strict=True)
    optimizer.load_state_dict(best_optimizer)
    checkpoint = save_checkpoint(
        model,
        optimizer,
        authority=authority,
        source_payload=base_payload,
        step=best_step,
        validation=best_metrics,
        baseline=baseline,
    )
    result = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V1_RESULT",
        "created_utc": utc_now(),
        "verdict": "TRAINED_PENDING_PHYSICAL_EVALUATION",
        "runtime_router": False,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "baseline": baseline,
        "selected_validation": best_metrics,
        "checkpoint": checkpoint,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-corpora", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
