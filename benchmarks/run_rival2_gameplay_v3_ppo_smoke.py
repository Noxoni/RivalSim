"""Bounded real Gameplay V3 mixed-opponent PPO continuation (479 -> 489).

This harness deliberately calls the unchanged production mixed-opponent trainer.
It adds read-only rollout telemetry, saves every accepted update, and fails closed
on any hard safety rejection or integrity failure.  It never changes reward,
classifier, simulator, policy, PPO, or curriculum constants.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_rival2_gameplay_v3_validation import (  # noqa: E402
    _make_env,
    _object_digest,
    _tensor_digest,
)
from rivalsim.gameplay_v3 import (  # noqa: E402
    CANONICAL_MECHANIC_NAMES,
    OUTCOME_NAMES,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_V3_CONTRACT,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    Rival2MixedPPOSafetyConfig,
    mixed_optimizer_learning_rates,
)
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

SCHEMA_VERSION = 1
SOURCE_CHECKPOINT = Path(
    r"G:\dev\RivalSim-runs\opponent-curriculum-v1-safe-20260827-b2af03d"
    r"\checkpoints\rival2_opponent_curriculum_plus_120_resume.pt"
)
SOURCE_SHA256 = "3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A"
EXPECTED_V3_HASH = "174D94E19B3F053E250147F98835C18CF65260A82E23B6E58F234F6E81E0D4E7"
SOURCE_ITERATION = 479
FINAL_ITERATION = 489
SOURCE_SAMPLES = 3_655_854_038
WORLDS = 131_072
CAMPAIGN_SEED = 2_026_082_703
KL_GUARD = Rival2KLGuardConfig(
    minibatch_kl_limit=0.10,
    completed_update_mean_kl_limit=0.05,
)
MIXED_PPO_SAFETY = Rival2MixedPPOSafetyConfig()
CORRECTION_V2_SUMMARY = Path("docs/RIVAL2_GAMEPLAY_V3_VALIDATION_CORRECTION_V2.md")
CORRECTION_V2_CONTRACT = Path(
    "results/rival2/gameplay_v3_validation_correction_v2/contract.json"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _git_is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode == 0


def _nested_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_nested_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_nested_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _make_trainer(
    source: dict[str, Any], collision_dir: Path, device: str
) -> Rival2OpponentCurriculumTrainer:
    env = _make_env(WORLDS, collision_dir)
    if str(env.device) != device:
        raise RuntimeError(f"environment device mismatch: {env.device} != {device}")
    curriculum = Rival2OpponentCurriculumConfig(**source["opponent_curriculum"]["config"])
    return Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        opponent_curriculum=curriculum,
        seed=CAMPAIGN_SEED,
    )


def _launch_gate(
    source: dict[str, Any], source_checkpoint: Path, source_sha256: str
) -> dict[str, Any]:
    correction = json.loads((REPO_ROOT / CORRECTION_V2_CONTRACT).read_text(encoding="utf-8"))
    adaptive = source["opponent_curriculum"]["adaptive_ppo"]
    source_rates = {
        group.get("name"): float(group["lr"])
        for group in source["optimizer"]["param_groups"]
    }
    expected_mix = {
        "nexto_probability": 0.35,
        "wisp_probability": 0.35,
        "current_probability": 0.20,
        "historical_probability": 0.10,
        "seed": CAMPAIGN_SEED,
    }
    checks = {
        "origin_main_is_ancestor_of_head": _git_is_ancestor("origin/main"),
        "implementation_worktree_clean": not _git("status", "--short"),
        "correction_v2_commit_present": _git_is_ancestor("031feca"),
        "runtime_correction_commit_present": _git_is_ancestor("acffb4b"),
        "update_local_lr_commit_present": _git_is_ancestor("b2af03d"),
        "adaptive_kl_retention_commit_present": _git_is_ancestor("35b7110"),
        "mixed_curriculum_commit_present": _git_is_ancestor("7fd335f"),
        "correction_v2_summary_present": (REPO_ROOT / CORRECTION_V2_SUMMARY).is_file(),
        "source_checkpoint_path_exact": source_checkpoint.resolve() == SOURCE_CHECKPOINT.resolve(),
        "source_checkpoint_sha256_exact": _sha256(source_checkpoint)
        == source_sha256
        == SOURCE_SHA256,
        "source_iteration_exact": int(source["iteration"]) == SOURCE_ITERATION,
        "source_policy_version_exact": int(source["policy_version"]) == SOURCE_ITERATION,
        "source_sample_counter_exact": int(source["total_agent_samples"]) == SOURCE_SAMPLES,
        "source_reward_v2_exact": source["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        "source_episode_exact": source["episode_version"] == RIVAL2_EPISODE_VERSION,
        "source_world_count_exact": int(source["opponent_assignment"].numel()) == WORLDS,
        "v3_contract_hash_exact": REWARD_GAMEPLAY_V3_CONTRACT_HASH == EXPECTED_V3_HASH,
        "committed_contract_hash_exact": correction.get("sha256") == EXPECTED_V3_HASH,
        "ordinary_touch_reward_zero": REWARD_GAMEPLAY_V3_CONTRACT[
            "unconditional_unique_touch"
        ]
        == 0.0,
        "mechanics_reward_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["event_reward"]
        == 0.005,
        "mechanics_budget_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["episode_budget"]
        == 0.05,
        "mechanics_event_cap_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"][
            "max_paid_events_per_player_episode"
        ]
        == 10,
        "bad_flip_penalty_exact": REWARD_GAMEPLAY_V3_CONTRACT[
            "unnecessary_flip_through_contact"
        ]["penalty_to_offender_before_zero_sum"]
        == -0.01,
        "opponent_mix_exact": source["opponent_curriculum"]["config"] == expected_mix,
        "adaptive_schema_v2": adaptive["schema_version"] == 2,
        "adaptive_policy_lr_scope_update_local": adaptive["policy_learning_rate_scope"]
        == "ppo_update_local",
        "adaptive_config_exact": adaptive["config"]
        == {
            "initial_policy_learning_rate": 0.0001,
            "critic_learning_rate": 0.0003,
            "soft_minibatch_kl_target": 0.02,
            "retention_soft_mean_kl_target": 0.02,
            "policy_learning_rate_backoff": 0.5,
            "minimum_policy_learning_rate": 0.000025,
            "retention_corpus_size": 512,
        },
        "source_policy_lr_rearmed": source_rates.get("policy") == 0.0001,
        "source_critic_lr_exact": source_rates.get("critic") == 0.0003,
        "source_retention_present": adaptive.get("retention_observations") is not None,
        "source_historical_pool_present": bool(source["historical_opponents"]),
        "hard_minibatch_kl_exact": KL_GUARD.minibatch_kl_limit == 0.10,
        "hard_completed_kl_exact": KL_GUARD.completed_update_mean_kl_limit == 0.05,
        "ten_update_boundary_exact": FINAL_ITERATION - SOURCE_ITERATION == 10,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "source_checkpoint": source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": source_sha256,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if payload["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gameplay V3 PPO smoke launch gate failed: {failed}")
    return payload


class RolloutTelemetry:
    """Read-only GPU accumulation for one production rollout at a time."""

    COMPONENT_VIEWS: ClassVar[dict[str, str]] = {
        "goals": "rival2.v1_goal_component",
        "progress": "rival2.v1_progress_component",
        "demos": "rival2.v1_demo_component",
        "speed": "rival2.speed_component",
        "supersonic": "rival2.supersonic_component",
        "boost_use": "rival2.boost_use_component",
        "boost_pickups": "rival2.boost_pickup_component",
        "saves": "rival2.save_component",
        "mechanics": "gameplay_v3.mechanics_component",
        "unnecessary_flip": "gameplay_v3.bad_flip_component",
    }
    COUNTER_VIEWS = (
        "total_detected",
        "total_paid",
        "legitimate_touch_total",
        "flip_touch_total",
        "outcome_total",
        "exemption_flag_total",
        "budget_exhausted_total",
    )

    def __init__(self, trainer: Rival2OpponentCurriculumTrainer):
        self.trainer = trainer
        self.env = trainer.env
        self.device = trainer.device
        self.worlds = self.env.num_envs
        self.episode_budget_hit = torch.zeros(
            (self.worlds, 2), dtype=torch.bool, device=self.device
        )
        self.previous_budget_total = self._counter("budget_exhausted_total").reshape(
            self.worlds, 2
        ).clone()
        self._active = False

    def _counter(self, name: str) -> torch.Tensor:
        return self.env.bridge.views[f"gameplay_v3.{name}"]

    def begin_update(self) -> None:
        if self._active:
            raise RuntimeError("telemetry update already active")
        self._active = True
        self.counter_before = {name: self._counter(name).clone() for name in self.COUNTER_VIEWS}
        self.component_abs = {
            name: torch.zeros((), dtype=torch.float64, device=self.device)
            for name in self.COMPONENT_VIEWS
        }
        self.component_signed = {
            name: torch.zeros((), dtype=torch.float64, device=self.device)
            for name in self.COMPONENT_VIEWS
        }
        self.gameplay_abs = torch.zeros((), dtype=torch.float64, device=self.device)
        self.total_reward_abs = torch.zeros((), dtype=torch.float64, device=self.device)
        self.raw = {
            name: torch.zeros((), dtype=torch.float64, device=self.device)
            for name in (
                "goal_events",
                "progress_nonzero_world_decisions",
                "progress_abs_ball_y_uu",
                "demo_events",
                "save_events",
                "speed_nonzero_world_decisions",
                "speed_abs_normalized_net_units",
                "supersonic_player_decisions",
                "boost_use_player_decisions",
                "small_pad_pickups",
                "big_pad_pickups",
                "world_decisions",
                "player_decisions",
                "completed_player_episodes",
                "completed_player_episodes_hitting_mechanics_budget",
            )
        }
        self.action_abs_sum = torch.zeros(8, dtype=torch.float64, device=self.device)
        self.action_nonzero = torch.zeros(8, dtype=torch.float64, device=self.device)
        self.action_saturated = torch.zeros(5, dtype=torch.float64, device=self.device)

    def capture_pre_reset(self) -> None:
        if not self._active:
            raise RuntimeError("telemetry capture outside active update")
        views = self.env.bridge.views
        for name, view_name in self.COMPONENT_VIEWS.items():
            value = views[view_name]
            self.component_abs[name].add_(value.abs().sum(dtype=torch.float64))
            self.component_signed[name].add_(value.sum(dtype=torch.float64))
        mechanics = views["gameplay_v3.mechanics_component"]
        bad_flip = views["gameplay_v3.bad_flip_component"]
        reward = views["rival2.reward"].reshape(self.worlds, 2)
        gameplay_blue = reward[:, 0] - mechanics - bad_flip
        self.gameplay_abs.add_(gameplay_blue.abs().sum(dtype=torch.float64))
        self.total_reward_abs.add_(reward[:, 0].abs().sum(dtype=torch.float64))

        progress = views["rival2.v1_progress_component"]
        self.raw["goal_events"].add_(
            views["rival2.terminated"].sum(dtype=torch.float64)
        )
        self.raw["progress_nonzero_world_decisions"].add_(
            (progress != 0).sum(dtype=torch.float64)
        )
        self.raw["progress_abs_ball_y_uu"].add_(
            progress.abs().sum(dtype=torch.float64) * 10_240.0
        )
        self.raw["demo_events"].add_(
            views["rival2.demo_by_count"].sum(dtype=torch.float64)
        )
        self.raw["save_events"].add_(views["rival2.save_count"].sum(dtype=torch.float64))
        self.raw["speed_nonzero_world_decisions"].add_(
            (views["rival2.speed_component"] != 0).sum(dtype=torch.float64)
        )
        self.raw["speed_abs_normalized_net_units"].add_(
            views["rival2.speed_component"].abs().sum(dtype=torch.float64) / 0.0001
        )
        self.raw["supersonic_player_decisions"].add_(
            views["is_supersonic"].sum(dtype=torch.float64)
        )
        self.raw["boost_use_player_decisions"].add_(
            views["rival2.boost_use_event"].sum(dtype=torch.float64)
        )
        self.raw["small_pad_pickups"].add_(
            views["rival2.small_pad_pickup_count"].sum(dtype=torch.float64)
        )
        self.raw["big_pad_pickups"].add_(
            views["rival2.big_pad_pickup_count"].sum(dtype=torch.float64)
        )
        self.raw["world_decisions"].add_(float(self.worlds))
        self.raw["player_decisions"].add_(float(self.worlds * 2))

        current_budget = self._counter("budget_exhausted_total").reshape(self.worlds, 2)
        budget_onset = current_budget > self.previous_budget_total
        self.episode_budget_hit.logical_or_(budget_onset)
        completed = views["rival2.reset_mask"].to(torch.bool)
        self.raw["completed_player_episodes"].add_(
            completed.sum(dtype=torch.float64) * 2.0
        )
        self.raw["completed_player_episodes_hitting_mechanics_budget"].add_(
            self.episode_budget_hit[completed].sum(dtype=torch.float64)
        )
        self.episode_budget_hit[completed] = False
        self.previous_budget_total.copy_(current_budget)

    def capture_post_step(self, transition: Any) -> None:
        if not self._active:
            raise RuntimeError("telemetry capture outside active update")

        emitted = transition.emitted_action.reshape(-1, 8)
        self.action_abs_sum.add_(emitted.abs().sum(dim=0, dtype=torch.float64))
        self.action_nonzero.add_((emitted.abs() > 1.0e-6).sum(dim=0, dtype=torch.float64))
        self.action_saturated.add_(
            (emitted[:, :5].abs() > 0.95).sum(dim=0, dtype=torch.float64)
        )

    def finish_update(self) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("telemetry update is not active")
        self._active = False
        counter_delta = {
            name: self._counter(name).to(torch.int64) - before.to(torch.int64)
            for name, before in self.counter_before.items()
        }
        if any(bool((value < 0).any().item()) for value in counter_delta.values()):
            raise RuntimeError("Gameplay V3 lifetime telemetry counter moved backward")

        detected_tensor = counter_delta["total_detected"].reshape(
            self.worlds, 2, len(CANONICAL_MECHANIC_NAMES)
        )
        paid_tensor = counter_delta["total_paid"].reshape_as(detected_tensor)
        outcomes = counter_delta["outcome_total"].reshape(
            self.worlds, 2, len(OUTCOME_NAMES)
        )
        flags = counter_delta["exemption_flag_total"].reshape_as(outcomes)
        raw = {name: float(value.item()) for name, value in self.raw.items()}
        raw.update(
            {
                "touches": int(counter_delta["legitimate_touch_total"].sum().item()),
                "flip_active_touches": int(counter_delta["flip_touch_total"].sum().item()),
                "unnecessary_flip_contacts": int(
                    outcomes[..., OUTCOME_NAMES.index("UNNECESSARY_FLIP_THROUGH_CONTACT")]
                    .sum()
                    .item()
                ),
                "mechanics_detected": int(detected_tensor.sum().item()),
                "mechanics_paid": int(paid_tensor.sum().item()),
                "mechanics_budget_hit_onsets": int(
                    counter_delta["budget_exhausted_total"].sum().item()
                ),
            }
        )
        component = {
            name: {
                "absolute_blue_sum": float(self.component_abs[name].item()),
                "signed_blue_sum": float(self.component_signed[name].item()),
                "mean_absolute_per_world_decision": float(
                    self.component_abs[name].item() / max(raw["world_decisions"], 1.0)
                ),
            }
            for name in self.COMPONENT_VIEWS
        }
        gameplay_abs = float(self.gameplay_abs.item())
        progress_abs = float(self.component_abs["progress"].item())
        mechanics_abs = float(self.component_abs["mechanics"].item())
        bad_abs = float(self.component_abs["unnecessary_flip"].item())
        flip_touches = int(raw["flip_active_touches"])
        touches = int(raw["touches"])
        car_minutes = raw["player_decisions"] / 30.0 / 60.0
        action_denominator = max(raw["player_decisions"], 1.0)
        completed_player_episodes = raw["completed_player_episodes"]
        return {
            "schema_version": SCHEMA_VERSION,
            "raw_counts_and_activity": raw,
            "reward_contributions": component,
            "absolute_gameplay_reward_sum": gameplay_abs,
            "absolute_total_reward_sum": float(self.total_reward_abs.item()),
            "ratios": {
                "mechanics_reward_to_absolute_gameplay_reward": mechanics_abs
                / max(gameplay_abs, 1.0e-30),
                "unnecessary_flip_penalty_to_absolute_gameplay_reward": bad_abs
                / max(gameplay_abs, 1.0e-30),
                "mechanics_reward_to_progress": mechanics_abs / max(progress_abs, 1.0e-30),
                "bad_flip_penalty_to_progress": bad_abs / max(progress_abs, 1.0e-30),
                "player_episode_fraction_hitting_mechanics_budget": raw[
                    "completed_player_episodes_hitting_mechanics_budget"
                ]
                / max(completed_player_episodes, 1.0),
            },
            "ball_contact": {
                "car_minutes": car_minutes,
                "touches_per_min": touches / max(car_minutes, 1.0e-30),
                "flip_active_touches_per_min": flip_touches / max(car_minutes, 1.0e-30),
                "total_flip_active_touch_count": flip_touches,
                "unnecessary_flip_through_contacts_per_min": int(
                    raw["unnecessary_flip_contacts"]
                )
                / max(car_minutes, 1.0e-30),
                "unnecessary_flip_through_fraction": int(raw["unnecessary_flip_contacts"])
                / max(flip_touches, 1),
                "exemption_flags": {
                    OUTCOME_NAMES[index]: int(flags[..., index].sum().item())
                    for index in range(1, 5)
                },
                "primary_outcomes": {
                    OUTCOME_NAMES[index]: int(outcomes[..., index].sum().item())
                    for index in range(1, len(OUTCOME_NAMES))
                },
            },
            "mechanics": {
                "detected": {
                    name: int(detected_tensor[..., index].sum().item())
                    for index, name in enumerate(CANONICAL_MECHANIC_NAMES)
                },
                "paid": {
                    name: int(paid_tensor[..., index].sum().item())
                    for index, name in enumerate(CANONICAL_MECHANIC_NAMES)
                },
            },
            "action_activity": {
                "mean_absolute": {
                    name: float(self.action_abs_sum[index].item() / action_denominator)
                    for index, name in enumerate(
                        ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
                    )
                },
                "nonzero_fraction": {
                    name: float(self.action_nonzero[index].item() / action_denominator)
                    for index, name in enumerate(
                        ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
                    )
                },
                "analog_saturation_fraction": {
                    name: float(self.action_saturated[index].item() / action_denominator)
                    for index, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll"))
                },
            },
            "definitions": {
                "reward_contribution": (
                    "native zero-sum Blue component; Orange is its exact negation, so absolute "
                    "two-player sums are exactly twice these values and ratios are unchanged"
                ),
                "absolute_gameplay_reward": (
                    "absolute Blue reward after subtracting V3 mechanics and bad-flip components"
                ),
                "budget_fraction": (
                    "completed player episodes in this update whose persistent episode tracker "
                    "observed the native budget-exhausted onset"
                ),
                "speed_raw_activity": "world decisions with nonzero competitive speed component",
                "progress_raw_activity": "nonzero intervals and reconstructed absolute ball-Y uu",
                "pre_reset_capture": (
                    "all native event/state/component counts are read after reward composition and "
                    "before selective kickoff reset; no terminal-world events are lost"
                ),
            },
        }


def _transition_gate(
    source: dict[str, Any],
    trainer: Rival2OpponentCurriculumTrainer,
    transition: dict[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    source_adaptive = source["opponent_curriculum"]["adaptive_ppo"]
    checks = {
        "model_exact": _tensor_digest(trainer.model.state_dict())
        == _tensor_digest(source["model"]),
        "optimizer_exact": _object_digest(trainer.optimizer.state_dict())
        == _object_digest(source["optimizer"]),
        "iteration_exact": trainer.iteration == SOURCE_ITERATION,
        "policy_version_exact": trainer.policy_version == SOURCE_ITERATION,
        "sample_counter_exact": trainer.total_agent_samples == SOURCE_SAMPLES,
        "policy_rng_exact": torch.equal(
            trainer.policy_generator.get_state().cpu(), source["policy_generator_state"].cpu()
        ),
        "opponent_rng_exact": torch.equal(
            trainer.opponent_generator.get_state().cpu(), source["opponent_generator_state"].cpu()
        ),
        "curriculum_rng_exact": torch.equal(
            trainer.curriculum_generator.get_state().cpu(),
            source["opponent_curriculum"]["generator_state"].cpu(),
        ),
        "opponent_assignment_exact": torch.equal(
            trainer.opponent_assignment.cpu(), source["opponent_assignment"].cpu()
        ),
        "opponent_family_exact": torch.equal(
            trainer.opponent_family.cpu(), source["opponent_curriculum"]["family"].cpu()
        ),
        "rival_side_exact": torch.equal(
            trainer.rival_side.cpu(), source["opponent_curriculum"]["rival_side"].cpu()
        ),
        "realized_family_counts_exact": torch.equal(
            trainer.realized_family_assignments.cpu(),
            source["opponent_curriculum"]["realized_family_assignments"].cpu(),
        ),
        "historical_pool_exact": _object_digest(trainer.opponent_pool.checkpoint_state())
        == _object_digest(source["historical_opponents"]),
        "retention_exact": torch.equal(
            trainer.retention_observations.cpu(), source_adaptive["retention_observations"].cpu()
        ),
        "adaptive_config_exact": trainer.mixed_ppo_safety == MIXED_PPO_SAFETY,
        "policy_lr_exact": mixed_optimizer_learning_rates(trainer.optimizer)["policy"] == 0.0001,
        "critic_lr_exact": mixed_optimizer_learning_rates(trainer.optimizer)["critic"] == 0.0003,
        "destination_reward_v3": trainer.env.reward_version == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "destination_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION, RIVAL2_EPISODE_VERSION),
        "fresh_world": bool(
            (trainer.env.bridge.views["rival2.episode_ticks"] == 0).all().item()
            and (trainer.env.bridge.views["gameplay_v3.total_detected"] == 0).all().item()
            and (trainer.env.bridge.views["gameplay_v3.outcome_total"] == 0).all().item()
        ),
        "checkpoint_payload_v3": payload["reward_version"]
        == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "transition_changed_only_reward_and_fresh_world": transition["changed_semantics"]
        == ["reward_contract", "fresh_world_state"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "transition": transition,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _integrity_gate(
    trainer: Rival2OpponentCurriculumTrainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    *,
    policy_before: int,
    iteration_before: int,
    samples_before: int,
) -> dict[str, Any]:
    adaptive = trainer.last_adaptive_ppo_diagnostics
    expected_samples = int(rollout.train_mask.sum().item())
    curriculum = trainer.last_rollout_curriculum_metrics
    checks = {
        "finite_metrics": all(
            bool(torch.isfinite(value).all().item()) for value in metrics.values()
        ),
        "finite_rewards": bool(torch.isfinite(rollout.rewards).all().item()),
        "finite_returns": bool(torch.isfinite(rollout.returns).all().item()),
        "finite_advantages": bool(torch.isfinite(rollout.advantages).all().item()),
        "zero_sum_exact": bool((rollout.rewards.sum(dim=-1) == 0.0).all().item()),
        "iteration_increment_exact": trainer.iteration == iteration_before + 1,
        "policy_increment_exact": trainer.policy_version == policy_before + 1,
        "sample_increment_exact": trainer.total_agent_samples - samples_before == expected_samples,
        "family_sample_ledger_exact": curriculum is not None
        and sum(curriculum["trainable_agent_samples"].values()) == expected_samples,
        "family_world_decision_ledger_exact": curriculum is not None
        and sum(curriculum["world_decisions"].values())
        == WORLDS * trainer.ppo_config.rollout_horizon,
        "hard_completed_kl_within_guard": float(metrics["approx_kl"].item()) <= 0.05,
        "hard_minibatch_kl_within_guard": float(
            metrics["optimizer_post_step_approx_kl_max"].item()
        )
        <= 0.10,
        "adaptive_diagnostics_present": adaptive is not None,
        "adaptive_pass_green": adaptive is not None and adaptive["verdict"] == "PASS_GREEN",
        "family_local_advantage_normalization": adaptive is not None
        and adaptive["checks"]["family_local_advantage_normalization"],
        "value_loss_to_trunk_zero": adaptive is not None
        and adaptive["checks"]["value_loss_to_shared_trunk_gradient_exact_zero"],
        "value_loss_to_actor_zero": adaptive is not None
        and adaptive["checks"]["value_loss_to_actor_gradient_exact_zero"],
        "policy_lr_started_at_base": adaptive is not None
        and adaptive["policy_learning_rate_start"] == 0.0001,
        "policy_lr_rearmed_for_next_update": adaptive is not None
        and adaptive["checks"]["next_update_policy_learning_rate_rearmed"],
        "critic_lr_unchanged": adaptive is not None
        and adaptive["critic_learning_rate_start_end"] == 0.0003,
        "retention_within_soft_target": adaptive is not None
        and adaptive["retention_corpus_mean_kl"] <= 0.02,
        "model_and_optimizer_finite": _nested_finite(trainer.model.state_dict())
        and _nested_finite(trainer.optimizer.state_dict()),
        "hot_path_transfer_counters_zero": trainer.env.hot_path_transfer_bytes()
        == {"h2d": 0, "d2h": 0},
    }
    return {
        "expected_trainable_agent_samples": expected_samples,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _save_checkpoint(
    trainer: Rival2OpponentCurriculumTrainer, work_dir: Path
) -> dict[str, Any]:
    iteration = trainer.iteration
    path = (
        work_dir
        / "checkpoints"
        / f"rival2_gameplay_v3_smoke_iteration_{iteration:03d}_resume.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adaptive = payload["opponent_curriculum"]["adaptive_ppo"]
    rates = {
        group.get("name"): float(group["lr"])
        for group in payload["optimizer"]["param_groups"]
    }
    checks = {
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "iteration_exact": int(payload["iteration"]) == iteration,
        "policy_version_exact": int(payload["policy_version"]) == trainer.policy_version,
        "sample_counter_exact": int(payload["total_agent_samples"]) == trainer.total_agent_samples,
        "reward_v3_exact": payload["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "episode_exact": payload["episode_version"] == RIVAL2_EPISODE_VERSION,
        "contracts_exact": payload["contract_hashes"] == trainer.env.contract_hashes,
        "model_finite": _nested_finite(payload["model"]),
        "optimizer_finite": _nested_finite(payload["optimizer"]),
        "split_optimizer": len(payload["optimizer"]["param_groups"]) == 2,
        "policy_lr_rearmed": rates.get("policy") == 0.0001,
        "critic_lr_exact": rates.get("critic") == 0.0003,
        "curriculum_transition_present": "curriculum_transition" in payload,
        "curriculum_state_present": payload.get("opponent_curriculum") is not None,
        "opponent_assignments_present": payload.get("opponent_assignment") is not None,
        "historical_pool_present": bool(payload.get("historical_opponents")),
        "adaptive_schema_v2": adaptive["schema_version"] == 2,
        "adaptive_lr_scope_update_local": adaptive["policy_learning_rate_scope"]
        == "ppo_update_local",
        "retention_present": adaptive.get("retention_observations") is not None,
        "next_update_policy_lr_exact": adaptive["next_update_policy_learning_rate"] == 0.0001,
    }
    result = {
        "iteration": iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "historical_pool_versions": list(trainer.opponent_pool.versions),
        "audit": {
            "checks": checks,
            "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        },
    }
    if result["audit"]["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"checkpoint audit failed at iteration {iteration}: {checks}")
    return result


def _compact_safety(iteration: int, adaptive: dict[str, Any]) -> dict[str, Any]:
    gradients = adaptive["maximum_gradient_norms"]
    return {
        "iteration": iteration,
        "optimizer_step_proposals": adaptive["optimizer_step_proposals"],
        "accepted_optimizer_steps": adaptive["accepted_optimizer_steps"],
        "retention_budget_early_stop": adaptive["ppo_early_stop"],
        "early_stop_reason": adaptive["ppo_early_stop_reason"],
        "policy_learning_rate_start": adaptive["policy_learning_rate_start"],
        "policy_learning_rate_end": adaptive["policy_learning_rate_end"],
        "policy_learning_rate_after_update_rearm": adaptive[
            "policy_learning_rate_after_update_rearm"
        ],
        "policy_learning_rate_backoffs": adaptive["policy_learning_rate_backoffs"],
        "transactional_retries": adaptive["optimizer_step_retries"],
        "maximum_post_step_minibatch_kl": adaptive["maximum_post_step_minibatch_kl"],
        "completed_update_mean_kl": adaptive["completed_update_mean_kl"],
        "retention_mean_kl": adaptive["retention_corpus_mean_kl"],
        "rollout_kl_by_action_channel": adaptive["rollout_analytic_kl_by_action_channel"],
        "retention_kl_by_action_channel": adaptive["retention_kl_by_action_channel"],
        "maximum_gradient_norms": gradients,
        "value_loss_to_policy_trunk_gradient_exact_zero": gradients[
            "value_loss_to_trunk_gradient_norm"
        ]
        == 0.0,
        "value_loss_to_actor_gradient_exact_zero": gradients[
            "value_loss_to_actor_gradient_norm"
        ]
        == 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_checkpoint = args.source_checkpoint.resolve()
    source_sha256 = str(args.source_sha256).upper()
    work_dir = args.work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError("work directory must be absent or empty")
    work_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    launch = _launch_gate(source, source_checkpoint, source_sha256)
    _write_json(work_dir / "launch_gate.json", launch)
    source_sha_before = _sha256(source_checkpoint)

    torch.cuda.empty_cache()
    trainer = _make_trainer(source, args.collision_dir.resolve(), args.device)
    transition = trainer.load_checkpoint_curriculum_transition(
        source_checkpoint,
        source_reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": "user-authorized bounded real Gameplay V3 PPO smoke 479 to 489",
            "authorized_change": "fresh Gameplay V2 checkpoint to Gameplay V3 training environment",
            "source_checkpoint_sha256": source_sha256,
            "training_is_real_and_resumable": True,
            "maximum_accepted_updates": 10,
        },
    )
    transition_gate = _transition_gate(source, trainer, transition)
    _write_json(work_dir / "transition_gate.json", transition_gate)
    if transition_gate["verdict"] != "PASS_GREEN":
        raise RuntimeError("Gameplay V3 transition preservation gate failed")

    telemetry = RolloutTelemetry(trainer)
    original_step: Callable[[torch.Tensor], Any] = trainer._step_with_frozen_opponents
    original_apply_interval_resets: Callable[[], None] = trainer.env.world.apply_interval_resets

    def instrumented_apply_interval_resets() -> None:
        telemetry.capture_pre_reset()
        original_apply_interval_resets()

    def instrumented_step(action: torch.Tensor) -> Any:
        result = original_step(action)
        telemetry.capture_post_step(result)
        return result

    trainer.env.world.apply_interval_resets = instrumented_apply_interval_resets  # type: ignore[method-assign]
    trainer._step_with_frozen_opponents = instrumented_step  # type: ignore[method-assign]
    checkpoints: list[dict[str, Any]] = []
    compact_safety: list[dict[str, Any]] = []
    ledger = work_dir / "training_curve.jsonl"
    started = time.perf_counter()

    for target_iteration in range(SOURCE_ITERATION + 1, FINAL_ITERATION + 1):
        policy_before = trainer.policy_version
        iteration_before = trainer.iteration
        samples_before = trainer.total_agent_samples
        pool_before = list(trainer.opponent_pool.versions)
        trainer.env.reset_transfer_counters()
        telemetry.begin_update()
        update_started = time.perf_counter()
        rollout = trainer.collect_rollout()
        rollout_telemetry = telemetry.finish_update()
        try:
            metrics = trainer.update(rollout, kl_guard=KL_GUARD)
        except Rival2PolicyDisplacementRejected as error:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_HARD_SAFETY_GUARD",
                "target_iteration": target_iteration,
                "last_accepted_iteration": trainer.iteration,
                "last_accepted_checkpoint": checkpoints[-1] if checkpoints else None,
                "rollout_telemetry": rollout_telemetry,
                "diagnostic": error.diagnostics,
                "source_checkpoint_byte_identical": _sha256(source_checkpoint)
                == source_sha_before
                == source_sha256,
                "no_later_training_performed": True,
            }
            _write_json(work_dir / "hard_safety_failure.json", failure)
            _write_json(work_dir / "run_summary.json", failure)
            return failure
        except Exception as error:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_HARD_UPDATE_FAILURE",
                "target_iteration": target_iteration,
                "last_accepted_iteration": iteration_before,
                "last_accepted_checkpoint": checkpoints[-1] if checkpoints else None,
                "rollout_telemetry": rollout_telemetry,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "no_later_training_performed": True,
            }
            _write_json(work_dir / "hard_safety_failure.json", failure)
            _write_json(work_dir / "run_summary.json", failure)
            raise

        torch.cuda.synchronize(args.device)
        wall_seconds = time.perf_counter() - update_started
        integrity = _integrity_gate(
            trainer,
            rollout,
            metrics,
            policy_before=policy_before,
            iteration_before=iteration_before,
            samples_before=samples_before,
        )
        if integrity["verdict"] != "PASS_GREEN":
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_POST_UPDATE_INTEGRITY_FAILURE",
                "target_iteration": target_iteration,
                "last_accepted_iteration": iteration_before,
                "last_accepted_checkpoint": checkpoints[-1] if checkpoints else None,
                "integrity": integrity,
                "no_later_training_performed": True,
            }
            _write_json(work_dir / "hard_safety_failure.json", failure)
            _write_json(work_dir / "run_summary.json", failure)
            return failure
        if trainer.iteration != target_iteration:
            raise RuntimeError("accepted update landed on the wrong iteration")
        if list(trainer.opponent_pool.versions) != pool_before:
            raise RuntimeError("historical pool changed outside its production snapshot schedule")

        checkpoint = _save_checkpoint(trainer, work_dir)
        checkpoints.append(checkpoint)
        adaptive = copy.deepcopy(trainer.last_adaptive_ppo_diagnostics)
        safety = _compact_safety(target_iteration, adaptive)
        compact_safety.append(safety)
        point = {
            "schema_version": SCHEMA_VERSION,
            "phase": "GAMEPLAY_V3_MIXED_OPPONENT_BOUNDED_REAL_PPO_SMOKE",
            "created_utc": _utc_now(),
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
            "wall_seconds": wall_seconds,
            "reward_version": trainer.env.reward_version,
            "episode_version": trainer.env.episode_version,
            "family": trainer.last_rollout_curriculum_metrics,
            "adaptive_ppo": adaptive,
            "ppo_safety_summary": safety,
            "reward_and_behavior_telemetry": rollout_telemetry,
            "metrics": {name: float(value.item()) for name, value in metrics.items()},
            "integrity": integrity,
            "checkpoint": checkpoint,
            "historical_pool_unchanged_this_update": True,
            "verdict": "PASS_GREEN",
        }
        _append_jsonl(ledger, point)
        _write_json(work_dir / "checkpoints.json", checkpoints)
        _write_json(work_dir / "ppo_safety_summary.json", compact_safety)
        print(
            "gameplay-v3 update="
            f"{trainer.iteration} samples={trainer.total_agent_samples} "
            f"delta={trainer.total_agent_samples - samples_before} "
            f"seconds={wall_seconds:.3f} "
            f"mb_kl={safety['maximum_post_step_minibatch_kl']:.6f} "
            f"mean_kl={safety['completed_update_mean_kl']:.6f} "
            f"retention_kl={safety['retention_mean_kl']:.6f} "
            f"early_stop={safety['retention_budget_early_stop']} "
            "verdict=PASS_GREEN",
            flush=True,
        )
        del rollout, metrics, adaptive
        gc.collect()
        torch.cuda.empty_cache()

    final = checkpoints[-1]
    source_sha_after = _sha256(source_checkpoint)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "COMPLETE_10_ACCEPTED_UPDATES",
        "implementation_commit": _git("rev-parse", "HEAD"),
        "source_checkpoint": {
            "path": source_checkpoint.as_posix(),
            "sha256": source_sha256,
            "iteration": SOURCE_ITERATION,
            "policy_version": SOURCE_ITERATION,
            "agent_decision_samples": SOURCE_SAMPLES,
            "byte_identical_after_run": source_sha_before == source_sha_after == source_sha256,
        },
        "final_checkpoint": final,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "accepted_updates": trainer.iteration - SOURCE_ITERATION,
        "additional_agent_decision_samples": trainer.total_agent_samples - SOURCE_SAMPLES,
        "checkpoint_count": len(checkpoints),
        "checkpoint_every_accepted_update": [item["iteration"] for item in checkpoints]
        == list(range(480, 490)),
        "hard_safety_guard_fired": False,
        "historical_pool_versions": list(trainer.opponent_pool.versions),
        "wall_seconds": time.perf_counter() - started,
        "evaluation_pending": True,
        "no_updates_beyond_489": trainer.iteration == FINAL_ITERATION,
        "verdict": "PASS_GREEN",
    }
    _write_json(work_dir / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary.get("status") == "COMPLETE_10_ACCEPTED_UPDATES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
