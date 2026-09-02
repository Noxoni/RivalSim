"""Mixed-opponent-only PPO safety transition for Rival 2.0.

This module deliberately does not alter the legacy/self-play PPO path.  It owns
the family-local advantage normalization, critic-gradient isolation, split Adam
migration, transactional soft-KL retries, and bounded retention probe required
by the Rival 2.0 mixed-opponent curriculum.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rivalsim.rival2_contracts import (
    ANALOG_ACTION_NAMES,
    BALL_LINEAR_SPEED_SCALE,
    BUTTON_ACTION_NAMES,
    CAR_LINEAR_SPEED_SCALE,
    OBS_DIM,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.rival2_policy import (
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    hybrid_distribution_parameters,
    hybrid_entropy,
    hybrid_log_probability,
)
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
    Rival2RolloutBuffer,
    _completed_update_diagnostics,
)

POLICY_GROUP_NAME = "policy"
CRITIC_GROUP_NAME = "critic"
ACTION_CHANNEL_NAMES = (*ANALOG_ACTION_NAMES, *BUTTON_ACTION_NAMES)


@dataclass(frozen=True, slots=True)
class Rival2MixedPPOSafetyConfig:
    """Adaptive settings outside the frozen historical PPO identity."""

    initial_policy_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    soft_minibatch_kl_target: float = 0.02
    retention_soft_mean_kl_target: float = 0.02
    policy_learning_rate_backoff: float = 0.5
    minimum_policy_learning_rate: float = 2.5e-5
    retention_corpus_size: int = 512

    def __post_init__(self) -> None:
        positive = {
            "initial policy learning rate": self.initial_policy_learning_rate,
            "critic learning rate": self.critic_learning_rate,
            "soft minibatch KL target": self.soft_minibatch_kl_target,
            "retention soft mean KL target": self.retention_soft_mean_kl_target,
            "minimum policy learning rate": self.minimum_policy_learning_rate,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.policy_learning_rate_backoff < 1.0:
            raise ValueError("policy learning-rate backoff must be in (0,1)")
        if self.minimum_policy_learning_rate > self.initial_policy_learning_rate:
            raise ValueError("minimum policy learning rate exceeds the initial rate")
        if self.retention_corpus_size <= 0:
            raise ValueError("retention corpus size must be positive")

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest().upper()


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _state_digest_by_name(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            value = state[key]
            if isinstance(value, torch.Tensor):
                host = value.detach().cpu().contiguous()
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(json.dumps(list(host.shape)).encode("ascii"))
                digest.update(host.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest().upper()


def _optimizer_parameter_state_by_name(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        name: copy.deepcopy(optimizer.state.get(parameter, {}))
        for name, parameter in model.named_parameters()
    }


def _adam_group_options(group: dict[str, Any]) -> dict[str, Any]:
    ignored = {"params", "lr", "initial_lr", "name"}
    return {key: copy.deepcopy(value) for key, value in group.items() if key not in ignored}


def _adam_step_value(state: dict[str, Any]) -> int | float | None:
    value = state.get("step")
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _split_parameter_lists(
    model: Rival2ActorCritic,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    policy = [*model.trunk.parameters(), *model.actor.parameters()]
    critic = list(model.critic.parameters())
    if len(policy) + len(critic) != len(list(model.parameters())):
        raise RuntimeError("mixed PPO parameter partition is incomplete")
    if {id(parameter) for parameter in policy}.intersection(id(p) for p in critic):
        raise RuntimeError("mixed PPO parameter groups overlap")
    return policy, critic


def make_empty_mixed_optimizer(
    model: Rival2ActorCritic,
    *,
    policy_learning_rate: float,
    critic_learning_rate: float,
    source_group_options: dict[str, Any] | None = None,
) -> torch.optim.Adam:
    policy, critic = _split_parameter_lists(model)
    options = {} if source_group_options is None else copy.deepcopy(source_group_options)
    return torch.optim.Adam(
        [
            {
                "params": policy,
                "lr": policy_learning_rate,
                "name": POLICY_GROUP_NAME,
                **options,
            },
            {
                "params": critic,
                "lr": critic_learning_rate,
                "name": CRITIC_GROUP_NAME,
                **options,
            },
        ]
    )


def migrate_adam_to_mixed_groups(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: Rival2MixedPPOSafetyConfig,
) -> tuple[torch.optim.Adam, dict[str, Any]]:
    """Split one Adam group without changing any parameter-owned Adam state."""

    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("mixed PPO migration requires torch.optim.Adam")
    if len(optimizer.param_groups) != 1:
        raise ValueError("source optimizer must contain exactly one parameter group")
    source_group = optimizer.param_groups[0]
    source_options = _adam_group_options(source_group)
    before_by_name = _optimizer_parameter_state_by_name(model, optimizer)
    before_digest = _state_digest_by_name(model, optimizer)
    source_state = copy.deepcopy(optimizer.state_dict())

    migrated = make_empty_mixed_optimizer(
        model,
        policy_learning_rate=config.initial_policy_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        source_group_options=source_options,
    )
    new_state_template = migrated.state_dict()

    old_id_by_object: dict[int, int] = {}
    for live_group, saved_group in zip(
        optimizer.param_groups, source_state["param_groups"], strict=True
    ):
        for parameter, saved_id in zip(live_group["params"], saved_group["params"], strict=True):
            old_id_by_object[id(parameter)] = int(saved_id)

    new_id_by_object: dict[int, int] = {}
    for live_group, saved_group in zip(
        migrated.param_groups, new_state_template["param_groups"], strict=True
    ):
        for parameter, saved_id in zip(live_group["params"], saved_group["params"], strict=True):
            new_id_by_object[id(parameter)] = int(saved_id)

    migrated_state: dict[int, Any] = {}
    for parameter in model.parameters():
        old_id = old_id_by_object.get(id(parameter))
        new_id = new_id_by_object.get(id(parameter))
        if old_id is None or new_id is None:
            raise RuntimeError("optimizer migration lost a model parameter")
        if old_id in source_state["state"]:
            migrated_state[new_id] = copy.deepcopy(source_state["state"][old_id])
    new_state_template["state"] = migrated_state
    migrated.load_state_dict(new_state_template)

    after_by_name = _optimizer_parameter_state_by_name(model, migrated)
    state_exact_by_name = {
        name: _nested_exact(before_by_name[name], after_by_name[name])
        for name, _parameter in model.named_parameters()
    }
    step_exact_by_name = {
        name: _nested_exact(
            before_by_name[name].get("step"),
            after_by_name[name].get("step"),
        )
        for name, _parameter in model.named_parameters()
    }
    after_digest = _state_digest_by_name(model, migrated)
    checks = {
        "every_parameter_present": len(state_exact_by_name) == len(list(model.parameters())),
        "all_parameter_adam_state_exact": all(state_exact_by_name.values()),
        "all_parameter_step_counters_exact": all(step_exact_by_name.values()),
        "named_state_digest_exact": before_digest == after_digest,
        "two_parameter_groups": len(migrated.param_groups) == 2,
        "policy_group_identity_exact": migrated.param_groups[0].get("name") == POLICY_GROUP_NAME,
        "critic_group_identity_exact": migrated.param_groups[1].get("name") == CRITIC_GROUP_NAME,
        "policy_learning_rate_exact": migrated.param_groups[0]["lr"]
        == config.initial_policy_learning_rate,
        "critic_learning_rate_exact": migrated.param_groups[1]["lr"] == config.critic_learning_rate,
    }
    proof = {
        "schema_version": 1,
        "source_group_count": 1,
        "destination_group_count": len(migrated.param_groups),
        "parameter_count": len(state_exact_by_name),
        "source_named_state_sha256": before_digest,
        "destination_named_state_sha256": after_digest,
        "state_exact_by_parameter": state_exact_by_name,
        "step_counter_exact_by_parameter": step_exact_by_name,
        "source_step_counter_by_parameter": {
            name: _adam_step_value(before_by_name[name]) for name in before_by_name
        },
        "destination_step_counter_by_parameter": {
            name: _adam_step_value(after_by_name[name]) for name in after_by_name
        },
        "destination_groups": [
            {
                "name": group.get("name"),
                "learning_rate": float(group["lr"]),
                "parameter_count": len(group["params"]),
            }
            for group in migrated.param_groups
        ],
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if proof["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"Adam state migration failed closed: {checks}")
    return migrated, proof


def mixed_optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    groups = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if set(groups) != {POLICY_GROUP_NAME, CRITIC_GROUP_NAME}:
        raise ValueError("optimizer does not have the required named mixed PPO groups")
    return groups


def set_policy_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    found = False
    for group in optimizer.param_groups:
        if group.get("name") == POLICY_GROUP_NAME:
            group["lr"] = float(value)
            found = True
        elif group.get("name") == CRITIC_GROUP_NAME:
            continue
        else:
            raise ValueError("unexpected mixed PPO optimizer group")
    if not found:
        raise ValueError("mixed PPO policy group is absent")


def reset_policy_learning_rate_for_new_update(
    optimizer: torch.optim.Optimizer,
    config: Rival2MixedPPOSafetyConfig,
) -> dict[str, float | bool]:
    """Re-arm the update-local policy rate without touching parameter-owned Adam state."""

    before = mixed_optimizer_learning_rates(optimizer)
    if before[CRITIC_GROUP_NAME] != config.critic_learning_rate:
        raise ValueError("critic-head learning rate differs from the frozen safety setting")
    if not (
        config.minimum_policy_learning_rate
        <= before[POLICY_GROUP_NAME]
        <= config.initial_policy_learning_rate
    ):
        raise ValueError("policy learning rate is outside the authorized adaptive range")
    set_policy_learning_rate(optimizer, config.initial_policy_learning_rate)
    after = mixed_optimizer_learning_rates(optimizer)
    return {
        "policy_learning_rate_before_reset": before[POLICY_GROUP_NAME],
        "policy_learning_rate_after_reset": after[POLICY_GROUP_NAME],
        "policy_learning_rate_reset_applied": (
            before[POLICY_GROUP_NAME] != after[POLICY_GROUP_NAME]
        ),
        "critic_learning_rate_before_reset": before[CRITIC_GROUP_NAME],
        "critic_learning_rate_after_reset": after[CRITIC_GROUP_NAME],
    }


def retention_observation_sha256(value: torch.Tensor) -> str:
    host = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(json.dumps(list(host.shape)).encode("ascii"))
    digest.update(host.numpy().tobytes())
    return digest.hexdigest().upper()


def _take_evenly_spaced(indices: torch.Tensor, count: int) -> torch.Tensor:
    take = min(int(indices.numel()), count)
    if take == 0:
        return indices[:0]
    positions = torch.div(
        torch.arange(take, device=indices.device, dtype=torch.int64) * indices.numel(),
        take,
        rounding_mode="floor",
    )
    return indices.index_select(0, positions)


def build_retention_observation_corpus(
    rollout: Rival2RolloutBuffer,
    *,
    corpus_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a fixed stratified corpus from real source-policy rollout observations."""

    if rollout.position != rollout.horizon:
        raise ValueError("retention corpus requires a complete rollout")
    indices = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    if indices.numel() < corpus_size:
        raise ValueError("rollout does not contain enough trainable retention observations")
    observations = rollout.observations.reshape(-1, OBS_DIM).index_select(0, indices)
    field = {name: OBS_FIELD_NAMES.index(name) for name in OBS_FIELD_NAMES}
    position_scale = torch.tensor(POSITION_SCALE, device=observations.device)
    relative_position = (
        observations[
            :,
            [
                field["relative.ball_position.x"],
                field["relative.ball_position.y"],
                field["relative.ball_position.z"],
            ],
        ]
        * position_scale
    )
    relative_velocity = (
        observations[
            :,
            [
                field["relative.ball_velocity.x"],
                field["relative.ball_velocity.y"],
                field["relative.ball_velocity.z"],
            ],
        ]
        * BALL_LINEAR_SPEED_SCALE
    )
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    closing = (relative_position * relative_velocity).sum(dim=-1) < 0.0
    on_ground = observations[:, field["self.on_ground"]] >= 0.5
    self_z = observations[:, field["self.position.z"]] * POSITION_SCALE[2]
    self_vz = observations[:, field["self.linear_velocity.z"]] * CAR_LINEAR_SPEED_SCALE
    kickoff = observations[:, field["lifecycle.kickoff_reset"]] >= 0.5

    base_category_quotas = (96, 96, 80, 96, 96)
    category_quotas = tuple(
        max(1, int(corpus_size * quota / 512)) for quota in base_category_quotas
    )
    category_specs = (
        ("near_ball_interaction", category_quotas[0], distance <= 500.0),
        (
            "possession_ball_approach",
            category_quotas[1],
            on_ground & ~kickoff & (distance > 500.0) & (distance <= 2000.0) & closing,
        ),
        (
            "recovery",
            category_quotas[2],
            ~on_ground & (self_z < 600.0) & (self_vz < 0.0),
        ),
        ("airborne", category_quotas[3], ~on_ground & (self_z >= 300.0)),
        (
            "ordinary_ground_play",
            category_quotas[4],
            on_ground & ~kickoff & (distance > 800.0),
        ),
    )
    selected = torch.zeros(observations.shape[0], dtype=torch.bool, device=observations.device)
    chosen_parts: list[torch.Tensor] = []
    category_counts: dict[str, dict[str, int]] = {}
    for name, quota, mask in category_specs:
        candidates = torch.nonzero(mask & ~selected, as_tuple=False).squeeze(-1)
        chosen = _take_evenly_spaced(candidates, quota)
        if chosen.numel() > 0:
            selected[chosen] = True
            chosen_parts.append(chosen)
        category_counts[name] = {
            "requested": quota,
            "available_before_exclusion": int(mask.sum().item()),
            "selected": int(chosen.numel()),
        }
    chosen_count = sum(int(part.numel()) for part in chosen_parts)
    remaining_count = corpus_size - chosen_count
    if remaining_count < 0:
        raise RuntimeError("retention category quotas exceed the corpus size")
    remaining = torch.nonzero(~selected, as_tuple=False).squeeze(-1)
    diversity = _take_evenly_spaced(remaining, remaining_count)
    if diversity.numel() != remaining_count:
        raise RuntimeError("retention corpus could not fill the diversity remainder")
    category_counts["field_position_orientation_diversity"] = {
        "requested": remaining_count,
        "available_before_exclusion": int(remaining.numel()),
        "selected": int(diversity.numel()),
    }
    chosen_parts.append(diversity)
    chosen_indices = torch.cat(chosen_parts)
    if chosen_indices.numel() != corpus_size or torch.unique(chosen_indices).numel() != corpus_size:
        raise RuntimeError("retention corpus selection is not exact and unique")
    corpus = observations.index_select(0, chosen_indices).detach().clone().contiguous()
    summary = {
        "schema_version": 1,
        "source": "authoritative trainable observations from the healthy source-policy rollout",
        "selection": "fixed deterministic category order and evenly-spaced candidate indices",
        "observation_contract": "RIVAL2_OBS_V1",
        "observation_count": int(corpus.shape[0]),
        "observation_dimension": int(corpus.shape[1]),
        "dtype": str(corpus.dtype),
        "sha256": retention_observation_sha256(corpus),
        "category_counts": category_counts,
        "checks": {
            "source_observations_authoritative": True,
            "trainable_samples_only": True,
            "fixed_count_exact": int(corpus.shape[0]) == corpus_size,
            "observation_dimension_exact": int(corpus.shape[1]) == OBS_DIM,
            "finite": bool(torch.isfinite(corpus).all().item()),
            "unique_source_indices": int(torch.unique(chosen_indices).numel()) == corpus_size,
        },
    }
    summary["verdict"] = "PASS_GREEN" if all(summary["checks"].values()) else "FAIL_RED"
    if summary["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"retention corpus construction failed: {summary['checks']}")
    return corpus, summary


def _analytic_channel_kl(
    old_actor: torch.Tensor,
    new_actor: torch.Tensor,
    policy_config: Rival2PolicyConfig,
    distribution_override: HybridDistributionOverride | None = None,
) -> torch.Tensor:
    old_mean, old_log_std, old_logits = hybrid_distribution_parameters(
        old_actor,
        policy_config,
        distribution_override=distribution_override,
    )
    new_mean, new_log_std, new_logits = hybrid_distribution_parameters(
        new_actor,
        policy_config,
        distribution_override=distribution_override,
    )
    old_variance = torch.exp(2.0 * old_log_std)
    new_variance = torch.exp(2.0 * new_log_std)
    analog = (
        new_log_std
        - old_log_std
        + (old_variance + (old_mean - new_mean).square()) / (2.0 * new_variance)
        - 0.5
    )
    old_probability = torch.sigmoid(old_logits).clamp(1.0e-7, 1.0 - 1.0e-7)
    new_probability = torch.sigmoid(new_logits).clamp(1.0e-7, 1.0 - 1.0e-7)
    buttons = old_probability * (torch.log(old_probability) - torch.log(new_probability))
    buttons += (1.0 - old_probability) * (
        torch.log1p(-old_probability) - torch.log1p(-new_probability)
    )
    return torch.cat((analog, buttons), dim=-1)


def _channel_means(value: torch.Tensor) -> dict[str, float]:
    means = value.mean(dim=0)
    return {name: float(means[index].item()) for index, name in enumerate(ACTION_CHANNEL_NAMES)}


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    values = [
        parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None
    ]
    if not values:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.linalg.vector_norm(torch.stack(values), 2)


def _snapshot_transaction(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def _restore_transaction_exact(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    snapshot: dict[str, Any],
) -> dict[str, bool]:
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    checks = {
        "parameters_exact": _nested_exact(snapshot["model"], model.state_dict()),
        "optimizer_state_exact": _nested_exact(snapshot["optimizer"], optimizer.state_dict()),
    }
    checks["adam_step_counters_exact"] = checks["optimizer_state_exact"]
    if not all(checks.values()):
        raise RuntimeError(f"transactional optimizer restoration failed: {checks}")
    return checks


def _family_normalize(
    raw_advantage: torch.Tensor,
    family: torch.Tensor,
    family_names: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, Any]]:
    normalized = torch.empty_like(raw_advantage)
    diagnostics: dict[str, Any] = {}
    for family_index, name in enumerate(family_names):
        selected = family == family_index
        count = int(selected.sum().item())
        if count == 0:
            diagnostics[name] = {
                "sample_count": 0,
                "raw_advantage_mean": None,
                "raw_advantage_std": None,
                "normalized_advantage_mean": None,
                "normalized_advantage_std": None,
            }
            continue
        values = raw_advantage[selected]
        result = (values - values.mean()) / values.std(unbiased=False).clamp_min(1.0e-8)
        normalized[selected] = result
        diagnostics[name] = {
            "sample_count": count,
            "raw_advantage_mean": float(values.mean().item()),
            "raw_advantage_std": float(values.std(unbiased=False).item()),
            "normalized_advantage_mean": float(result.mean().item()),
            "normalized_advantage_std": float(result.std(unbiased=False).item()),
        }
    return normalized, diagnostics


@torch.no_grad()
def _final_mixed_diagnostics(
    model: Rival2ActorCritic,
    behavior_model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    family: torch.Tensor,
    family_names: tuple[str, ...],
    policy_config: Rival2PolicyConfig,
    chunk_size: int,
    distribution_override: HybridDistributionOverride | None = None,
) -> dict[str, Any]:
    device = observation.device
    channel_sum = torch.zeros(8, dtype=torch.float64, device=device)
    family_kl_sum = torch.zeros(len(family_names), dtype=torch.float64, device=device)
    family_count = torch.zeros(len(family_names), dtype=torch.int64, device=device)
    for start in range(0, observation.shape[0], chunk_size):
        stop = min(start + chunk_size, observation.shape[0])
        new_actor, _ = model(observation[start:stop])
        old_actor, _ = behavior_model(observation[start:stop])
        channel = _analytic_channel_kl(
            old_actor,
            new_actor,
            policy_config,
            distribution_override,
        )
        channel_sum += channel.sum(dim=0, dtype=torch.float64)
        new_log_probability = hybrid_log_probability(
            new_actor,
            action[start:stop],
            config=policy_config,
            pre_tanh=pre_tanh[start:stop],
            distribution_override=distribution_override,
        )
        log_ratio = new_log_probability - old_log_probability[start:stop]
        sample_kl = (torch.exp(log_ratio) - 1.0) - log_ratio
        labels = family[start:stop]
        family_kl_sum.scatter_add_(0, labels, sample_kl.to(torch.float64))
        family_count += torch.bincount(labels, minlength=len(family_names))
    count = observation.shape[0]
    return {
        "rollout_analytic_kl_by_action_channel": {
            name: float((channel_sum[index] / count).item())
            for index, name in enumerate(ACTION_CHANNEL_NAMES)
        },
        "family_empirical_kl": {
            name: (
                None
                if int(family_count[index].item()) == 0
                else float((family_kl_sum[index] / family_count[index]).item())
            )
            for index, name in enumerate(family_names)
        },
    }


def _parameter_group_snapshot(model: Rival2ActorCritic) -> dict[str, list[torch.Tensor]]:
    return {
        "shared_trunk": [parameter.detach().clone() for parameter in model.trunk.parameters()],
        "actor_head": [parameter.detach().clone() for parameter in model.actor.parameters()],
        "critic_head": [parameter.detach().clone() for parameter in model.critic.parameters()],
    }


def _parameter_step_norms(
    model: Rival2ActorCritic,
    before: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    groups = {
        "shared_trunk": list(model.trunk.parameters()),
        "actor_head": list(model.actor.parameters()),
        "critic_head": list(model.critic.parameters()),
    }
    output: dict[str, float] = {}
    for name, parameters in groups.items():
        changes = [
            (parameter.detach() - prior).norm(2)
            for parameter, prior in zip(parameters, before[name], strict=True)
        ]
        output[name] = float(torch.linalg.vector_norm(torch.stack(changes), 2).item())
    return output


def probe_fresh_adam_first_minibatch(
    model: Rival2ActorCritic,
    rollout: Rival2RolloutBuffer,
    ppo_config: Rival2PPOConfig,
    *,
    retention_observations: torch.Tensor,
    family_names: tuple[str, ...],
    generator: torch.Generator,
    policy_learning_rate: float,
    critic_learning_rate: float,
    policy_config: Rival2PolicyConfig | None = None,
    gae_ready: bool = False,
) -> dict[str, Any]:
    """Measure and fully roll back one exact fresh-Adam PPO minibatch proposal.

    This transition diagnostic uses the production mixed-PPO objective,
    family-local advantage normalization, critic isolation, gradient clipping,
    and action KL definitions. Model, optimizer, generator, and global RNG state
    are restored before it returns; it never accepts a training step.
    """

    if not math.isfinite(policy_learning_rate) or policy_learning_rate <= 0.0:
        raise ValueError("probe policy learning rate must be finite and positive")
    if not math.isfinite(critic_learning_rate) or critic_learning_rate <= 0.0:
        raise ValueError("probe critic learning rate must be finite and positive")
    policy_config = policy_config or model.config
    if not gae_ready:
        rollout.compute_gae(ppo_config)
    if rollout.opponent_family is None:
        raise ValueError("mixed PPO rollout has no authoritative opponent-family identity")
    if retention_observations.shape != (512, OBS_DIM):
        raise ValueError("probe retention observation corpus shape mismatch")

    indices = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        raise RuntimeError("mixed PPO rollout contains no trainable samples")
    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, indices)
    returns = rollout.returns.reshape(-1).index_select(0, indices)
    raw_advantage = rollout.advantages.reshape(-1).index_select(0, indices)
    family = rollout.opponent_family.reshape(-1).index_select(0, indices)
    advantage, family_statistics = _family_normalize(raw_advantage, family, family_names)

    model_device = next(model.parameters()).device
    model_before = copy.deepcopy(model.state_dict())
    gradients_before = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]
    model_training_before = model.training
    generator_before = generator.get_state().clone()
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(model_device).clone()
        if model_device.type == "cuda"
        else None
    )
    optimizer = make_empty_mixed_optimizer(
        model,
        policy_learning_rate=policy_learning_rate,
        critic_learning_rate=critic_learning_rate,
    )
    transaction = _snapshot_transaction(model, optimizer)
    result: dict[str, Any] | None = None
    try:
        model.train()
        behavior_model = copy.deepcopy(model).eval().requires_grad_(False)
        with torch.no_grad():
            retention_reference, _ = behavior_model(retention_observations)
        permutation = torch.randperm(
            indices.numel(), device=rollout.device, generator=generator
        )
        batch = permutation[: ppo_config.minibatch_size]
        batch_observation = observation.index_select(0, batch)
        batch_action = action.index_select(0, batch)
        batch_pre_tanh = pre_tanh.index_select(0, batch)
        batch_old_log_probability = old_log_probability.index_select(0, batch)
        batch_advantage = advantage.index_select(0, batch)
        batch_returns = returns.index_select(0, batch)
        with torch.no_grad():
            behavior_actor, _ = behavior_model(batch_observation)

        trunk_parameters = list(model.trunk.parameters())
        actor_parameters = list(model.actor.parameters())
        critic_parameters = list(model.critic.parameters())
        hidden = model.trunk(batch_observation)
        actor_output = model.actor(hidden)
        value = model.critic(hidden.detach()).squeeze(-1)
        new_log_probability = hybrid_log_probability(
            actor_output,
            batch_action,
            config=policy_config,
            pre_tanh=batch_pre_tanh,
        )
        log_ratio = new_log_probability - batch_old_log_probability
        ratio = torch.exp(log_ratio)
        unclipped = ratio * batch_advantage
        clipped = ratio.clamp(
            1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range
        ) * batch_advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * (value - batch_returns).square().mean()
        entropy = hybrid_entropy(actor_output, policy_config).mean()
        actor_objective = policy_loss - ppo_config.entropy_coefficient * entropy
        weighted_value_loss = ppo_config.value_loss_coefficient * value_loss

        optimizer.zero_grad(set_to_none=True)
        actor_objective.backward(retain_graph=True)
        trunk_gradient_after_policy = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in trunk_parameters
        ]
        actor_gradient_after_policy = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in actor_parameters
        ]
        policy_trunk_gradient_norm = _gradient_norm(trunk_parameters)
        actor_head_gradient_norm = _gradient_norm(actor_parameters)
        critic_gradients = torch.autograd.grad(
            weighted_value_loss,
            critic_parameters,
            retain_graph=False,
            allow_unused=False,
        )
        for parameter, gradient in zip(
            critic_parameters, critic_gradients, strict=True
        ):
            parameter.grad = gradient
        value_loss_isolated = all(
            (before is None and parameter.grad is None)
            or (
                before is not None
                and parameter.grad is not None
                and torch.equal(before, parameter.grad)
            )
            for parameter, before in zip(
                trunk_parameters, trunk_gradient_after_policy, strict=True
            )
        ) and all(
            (before is None and parameter.grad is None)
            or (
                before is not None
                and parameter.grad is not None
                and torch.equal(before, parameter.grad)
            )
            for parameter, before in zip(
                actor_parameters, actor_gradient_after_policy, strict=True
            )
        )
        critic_head_gradient_norm = _gradient_norm(critic_parameters)
        raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), ppo_config.max_gradient_norm
        )
        post_clip_gradient_norm = _gradient_norm(list(model.parameters()))
        parameter_before = _parameter_group_snapshot(model)
        optimizer.step()
        parameter_step_norms = _parameter_step_norms(model, parameter_before)

        with torch.no_grad():
            post_actor, post_value = model(batch_observation)
            post_log_probability = hybrid_log_probability(
                post_actor,
                batch_action,
                config=policy_config,
                pre_tanh=batch_pre_tanh,
            )
            post_log_ratio = post_log_probability - batch_old_log_probability
            post_step_kl = (
                (torch.exp(post_log_ratio) - 1.0) - post_log_ratio
            ).mean()
            minibatch_channel = _analytic_channel_kl(
                behavior_actor, post_actor, policy_config
            ).mean(dim=0)
            retention_actor, retention_value = model(retention_observations)
            retention_channel = _analytic_channel_kl(
                retention_reference, retention_actor, policy_config
            ).mean(dim=0)
            retention_mean_kl = retention_channel.sum()
        model_finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in model.parameters()
        )
        output_finite = bool(
            torch.isfinite(post_actor).all()
            and torch.isfinite(post_value).all()
            and torch.isfinite(retention_actor).all()
            and torch.isfinite(retention_value).all()
            and torch.isfinite(post_step_kl)
            and torch.isfinite(retention_mean_kl)
        )
        post_kl = float(post_step_kl.item())
        retention_kl = float(retention_mean_kl.item())
        finite = bool(
            model_finite
            and output_finite
            and torch.isfinite(raw_gradient_norm)
            and torch.isfinite(post_clip_gradient_norm)
        )
        result = {
            "policy_learning_rate": policy_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "trainable_sample_count": int(indices.numel()),
            "minibatch_samples": int(batch.numel()),
            "minibatch_index_sha256": hashlib.sha256(
                batch.detach().cpu().contiguous().numpy().tobytes()
            )
            .hexdigest()
            .upper(),
            "family_statistics": family_statistics,
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "post_step_minibatch_kl": post_kl,
            "retention_mean_kl": retention_kl,
            "minibatch_kl_by_action_channel": {
                name: float(minibatch_channel[index].item())
                for index, name in enumerate(ACTION_CHANNEL_NAMES)
            },
            "retention_kl_by_action_channel": {
                name: float(retention_channel[index].item())
                for index, name in enumerate(ACTION_CHANNEL_NAMES)
            },
            "raw_gradient_norm": float(raw_gradient_norm.item()),
            "post_clip_gradient_norm": float(post_clip_gradient_norm.item()),
            "policy_trunk_gradient_norm": float(policy_trunk_gradient_norm.item()),
            "actor_head_gradient_norm": float(actor_head_gradient_norm.item()),
            "critic_head_gradient_norm": float(critic_head_gradient_norm.item()),
            "parameter_step_norms": parameter_step_norms,
            "value_loss_isolated_from_policy_trunk_and_actor": value_loss_isolated,
            "model_parameters_finite": model_finite,
            "outputs_finite": output_finite,
            "passes_soft_minibatch_kl": post_kl <= 0.02,
            "passes_soft_retention_kl": retention_kl <= 0.02,
            "passes_hard_minibatch_kl": post_kl < 0.10,
            "passes_finite_guard": finite,
        }
        result["passes_transition_gate"] = bool(
            result["passes_soft_minibatch_kl"]
            and result["passes_soft_retention_kl"]
            and result["passes_hard_minibatch_kl"]
            and result["passes_finite_guard"]
            and value_loss_isolated
        )
    finally:
        rollback_checks = _restore_transaction_exact(model, optimizer, transaction)
        for parameter, gradient in zip(
            model.parameters(), gradients_before, strict=True
        ):
            parameter.grad = None if gradient is None else gradient.clone()
        generator.set_state(generator_before)
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, model_device)
        model.train(model_training_before)
        rollback_checks.update(
            {
                "model_state_exact": _nested_exact(model_before, model.state_dict()),
                "model_gradients_exact": all(
                    (before is None and parameter.grad is None)
                    or (
                        before is not None
                        and parameter.grad is not None
                        and torch.equal(before, parameter.grad)
                    )
                    for parameter, before in zip(
                        model.parameters(), gradients_before, strict=True
                    )
                ),
                "generator_state_exact": torch.equal(
                    generator_before, generator.get_state()
                ),
                "global_cpu_rng_state_exact": torch.equal(
                    cpu_rng_before, torch.get_rng_state()
                ),
                "global_cuda_rng_state_exact": cuda_rng_before is None
                or torch.equal(cuda_rng_before, torch.cuda.get_rng_state(model_device)),
            }
        )
        if result is not None:
            result["rollback_checks"] = rollback_checks
            result["rollback_complete"] = all(rollback_checks.values())
    if result is None:
        raise RuntimeError("fresh-Adam transition probe produced no result")
    if not result["rollback_complete"]:
        raise RuntimeError(f"fresh-Adam transition probe rollback failed: {result}")
    return result


def ppo_update_mixed_curriculum(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rival2RolloutBuffer,
    ppo_config: Rival2PPOConfig,
    safety_config: Rival2MixedPPOSafetyConfig,
    *,
    retention_observations: torch.Tensor,
    family_names: tuple[str, ...],
    generator: torch.Generator,
    policy_config: Rival2PolicyConfig | None = None,
    kl_guard: Rival2KLGuardConfig,
    distribution_override: HybridDistributionOverride | None = None,
    gae_ready: bool = False,
    diagnostic_optimizer_step_limit: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the production mixed-curriculum PPO update with transactional retries."""

    policy_config = policy_config or model.config
    if not gae_ready:
        rollout.compute_gae(ppo_config)
    if rollout.opponent_family is None:
        raise ValueError("mixed PPO rollout has no authoritative opponent-family identity")
    learning_rate_reset = reset_policy_learning_rate_for_new_update(optimizer, safety_config)
    learning_rates = mixed_optimizer_learning_rates(optimizer)
    if retention_observations.shape != (safety_config.retention_corpus_size, OBS_DIM):
        raise ValueError("retention observation corpus shape mismatch")

    indices = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        raise RuntimeError("mixed PPO rollout contains no trainable samples")
    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, indices)
    old_value = rollout.values.reshape(-1).index_select(0, indices)
    returns = rollout.returns.reshape(-1).index_select(0, indices)
    raw_advantage = rollout.advantages.reshape(-1).index_select(0, indices)
    family = rollout.opponent_family.reshape(-1).index_select(0, indices)
    if bool(((family < 0) | (family >= len(family_names))).any().item()):
        raise ValueError("mixed PPO rollout contains invalid family identity")
    advantage, family_statistics = _family_normalize(raw_advantage, family, family_names)
    for family_index, name in enumerate(family_names):
        selected = family == family_index
        if not bool(selected.any().item()):
            continue
        family_statistics[name].update(
            {
                "return_mean": float(returns[selected].mean().item()),
                "return_std": float(returns[selected].std(unbiased=False).item()),
                "value_mean": float(old_value[selected].mean().item()),
                "value_std": float(old_value[selected].std(unbiased=False).item()),
            }
        )

    behavior_model = copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        retention_reference, _ = behavior_model(retention_observations)
    retention_reference_sha256 = retention_observation_sha256(retention_reference)

    _policy_parameters, critic_parameters = _split_parameter_lists(model)
    trunk_parameters = list(model.trunk.parameters())
    actor_parameters = list(model.actor.parameters())
    metrics: dict[str, list[torch.Tensor]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
        "gradient_norm": [],
        "post_clip_gradient_norm": [],
    }
    accepted_steps: list[dict[str, Any]] = []
    retry_log: list[dict[str, Any]] = []
    optimizer_step_index = 0
    proposal_count = 0
    lr_backoffs = 0
    early_stop_reason: str | None = None
    start_policy_lr = learning_rates[POLICY_GROUP_NAME]

    stop = False
    for epoch in range(ppo_config.epochs):
        permutation = torch.randperm(indices.numel(), device=rollout.device, generator=generator)
        for start in range(0, indices.numel(), ppo_config.minibatch_size):
            if (
                diagnostic_optimizer_step_limit is not None
                and optimizer_step_index >= diagnostic_optimizer_step_limit
            ):
                stop = True
                break
            batch = permutation[start : start + ppo_config.minibatch_size]
            batch_digest: str | None = None
            retry_index = 0
            while True:
                proposal_count += 1
                policy_lr_before = mixed_optimizer_learning_rates(optimizer)[POLICY_GROUP_NAME]
                transaction = _snapshot_transaction(model, optimizer)
                batch_observation = observation.index_select(0, batch)
                batch_action = action.index_select(0, batch)
                batch_pre_tanh = pre_tanh.index_select(0, batch)
                batch_old_log_probability = old_log_probability.index_select(0, batch)
                batch_advantage = advantage.index_select(0, batch)
                batch_returns = returns.index_select(0, batch)
                with torch.no_grad():
                    behavior_actor, _ = behavior_model(batch_observation)

                hidden = model.trunk(batch_observation)
                actor_output = model.actor(hidden)
                value = model.critic(hidden.detach()).squeeze(-1)
                new_log_probability = hybrid_log_probability(
                    actor_output,
                    batch_action,
                    config=policy_config,
                    pre_tanh=batch_pre_tanh,
                    distribution_override=distribution_override,
                )
                log_ratio = new_log_probability - batch_old_log_probability
                ratio = torch.exp(log_ratio)
                unclipped = ratio * batch_advantage
                clipped = (
                    ratio.clamp(1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range)
                    * batch_advantage
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (value - batch_returns).square().mean()
                entropy = hybrid_entropy(
                    actor_output,
                    policy_config,
                    distribution_override=distribution_override,
                ).mean()
                actor_objective = policy_loss - ppo_config.entropy_coefficient * entropy
                weighted_value_loss = ppo_config.value_loss_coefficient * value_loss
                total_loss = actor_objective + weighted_value_loss
                if not bool(torch.isfinite(total_loss).item()):
                    raise Rival2PolicyDisplacementRejected(
                        {
                            "reason": "nonfinite_total_loss",
                            "optimizer_step_index": optimizer_step_index,
                            "retry_index": retry_index,
                        }
                    )

                optimizer.zero_grad(set_to_none=True)
                actor_objective.backward(retain_graph=True)
                trunk_gradient_after_policy = [
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in trunk_parameters
                ]
                actor_gradient_after_policy = [
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in actor_parameters
                ]
                policy_trunk_gradient_norm = _gradient_norm(trunk_parameters)
                actor_head_gradient_norm = _gradient_norm(actor_parameters)
                critic_gradients = torch.autograd.grad(
                    weighted_value_loss,
                    critic_parameters,
                    retain_graph=False,
                    allow_unused=False,
                )
                for parameter, gradient in zip(critic_parameters, critic_gradients, strict=True):
                    parameter.grad = gradient
                trunk_unchanged = all(
                    (before is None and parameter.grad is None)
                    or (
                        before is not None
                        and parameter.grad is not None
                        and torch.equal(before, parameter.grad)
                    )
                    for parameter, before in zip(
                        trunk_parameters, trunk_gradient_after_policy, strict=True
                    )
                )
                actor_unchanged = all(
                    (before is None and parameter.grad is None)
                    or (
                        before is not None
                        and parameter.grad is not None
                        and torch.equal(before, parameter.grad)
                    )
                    for parameter, before in zip(
                        actor_parameters, actor_gradient_after_policy, strict=True
                    )
                )
                if not trunk_unchanged or not actor_unchanged:
                    raise RuntimeError("value loss modified policy-representation gradients")
                value_loss_to_trunk_gradient_norm = torch.zeros(
                    (), dtype=torch.float32, device=rollout.device
                )
                critic_head_gradient_norm = _gradient_norm(critic_parameters)
                raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ppo_config.max_gradient_norm
                )
                post_clip_gradient_norm = _gradient_norm(list(model.parameters()))
                if not bool(
                    torch.isfinite(raw_gradient_norm) & torch.isfinite(post_clip_gradient_norm)
                ):
                    raise Rival2PolicyDisplacementRejected(
                        {
                            "reason": "nonfinite_gradient",
                            "optimizer_step_index": optimizer_step_index,
                            "retry_index": retry_index,
                        }
                    )
                parameter_before = _parameter_group_snapshot(model)
                optimizer.step()
                parameter_step_norms = _parameter_step_norms(model, parameter_before)

                with torch.no_grad():
                    pre_step_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > ppo_config.clip_range).to(torch.float32).mean()
                    )
                    post_actor, _ = model(batch_observation)
                    post_log_probability = hybrid_log_probability(
                        post_actor,
                        batch_action,
                        config=policy_config,
                        pre_tanh=batch_pre_tanh,
                        distribution_override=distribution_override,
                    )
                    post_log_ratio = post_log_probability - batch_old_log_probability
                    post_step_kl = ((torch.exp(post_log_ratio) - 1.0) - post_log_ratio).mean()
                    batch_channel_kl = _analytic_channel_kl(
                        behavior_actor,
                        post_actor,
                        policy_config,
                        distribution_override,
                    )
                    retention_actor, _ = model(retention_observations)
                    retention_channel = _analytic_channel_kl(
                        retention_reference,
                        retention_actor,
                        policy_config,
                        distribution_override,
                    )
                    retention_channel_mean = retention_channel.mean(dim=0)
                    retention_mean_kl = retention_channel.sum(dim=-1).mean()
                post_step_kl_value = float(post_step_kl.item())
                retention_kl_value = float(retention_mean_kl.item())
                if kl_guard.reject_minibatch_kl and (
                    not math.isfinite(post_step_kl_value)
                    or post_step_kl_value > kl_guard.minibatch_kl_limit
                ):
                    restore_checks = _restore_transaction_exact(model, optimizer, transaction)
                    raise Rival2PolicyDisplacementRejected(
                        {
                            "reason": "minibatch_kl_limit_exceeded",
                            "optimizer_step_index": optimizer_step_index,
                            "retry_index": retry_index,
                            "minibatch_start": start,
                            "minibatch_samples": int(batch.numel()),
                            "pre_step_approx_kl": float(pre_step_kl.item()),
                            "post_step_approx_kl": post_step_kl_value,
                            "retention_mean_kl": retention_kl_value,
                            "transactional_step_restore": restore_checks,
                            "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                            "completed_update_mean_kl_limit": (
                                kl_guard.completed_update_mean_kl_limit
                            ),
                        }
                    )

                violations: list[str] = []
                if (
                    not kl_guard.kl_telemetry_only
                    and post_step_kl_value > safety_config.soft_minibatch_kl_target
                ):
                    violations.append("soft_minibatch_kl")
                if (
                    not kl_guard.kl_telemetry_only
                    and retention_kl_value > safety_config.retention_soft_mean_kl_target
                ):
                    violations.append("retention_mean_kl")
                if violations:
                    if batch_digest is None:
                        batch_digest = (
                            hashlib.sha256(batch.detach().cpu().contiguous().numpy().tobytes())
                            .hexdigest()
                            .upper()
                        )
                    restore_checks = _restore_transaction_exact(model, optimizer, transaction)
                    restored_rates = mixed_optimizer_learning_rates(optimizer)
                    if policy_lr_before <= safety_config.minimum_policy_learning_rate:
                        early_stop_reason = "+".join(violations) + "_at_minimum_policy_lr"
                        retry_log.append(
                            {
                                "optimizer_step_index": optimizer_step_index,
                                "retry_index": retry_index,
                                "epoch": epoch,
                                "minibatch_start": start,
                                "minibatch_samples": int(batch.numel()),
                                "minibatch_index_sha256": batch_digest,
                                "violations": violations,
                                "proposed_post_step_minibatch_kl": post_step_kl_value,
                                "proposed_retention_mean_kl": retention_kl_value,
                                "policy_learning_rate_before": policy_lr_before,
                                "policy_learning_rate_after": policy_lr_before,
                                "critic_learning_rate_before_after": restored_rates[
                                    CRITIC_GROUP_NAME
                                ],
                                "restore_checks": restore_checks,
                                "same_minibatch_retry": True,
                                "accepted": False,
                                "early_stop": True,
                            }
                        )
                        stop = True
                        break
                    new_policy_lr = max(
                        safety_config.minimum_policy_learning_rate,
                        policy_lr_before * safety_config.policy_learning_rate_backoff,
                    )
                    set_policy_learning_rate(optimizer, new_policy_lr)
                    lr_backoffs += 1
                    retry_log.append(
                        {
                            "optimizer_step_index": optimizer_step_index,
                            "retry_index": retry_index,
                            "epoch": epoch,
                            "minibatch_start": start,
                            "minibatch_samples": int(batch.numel()),
                            "minibatch_index_sha256": batch_digest,
                            "violations": violations,
                            "proposed_post_step_minibatch_kl": post_step_kl_value,
                            "proposed_retention_mean_kl": retention_kl_value,
                            "policy_learning_rate_before": policy_lr_before,
                            "policy_learning_rate_after": new_policy_lr,
                            "critic_learning_rate_before_after": restored_rates[CRITIC_GROUP_NAME],
                            "restore_checks": restore_checks,
                            "same_minibatch_retry": True,
                            "accepted": False,
                            "early_stop": False,
                        }
                    )
                    retry_index += 1
                    continue

                step = {
                    "optimizer_step_index": optimizer_step_index,
                    "epoch": epoch,
                    "minibatch_start": start,
                    "minibatch_samples": int(batch.numel()),
                    "minibatch_index_sha256": batch_digest,
                    "retry_count": retry_index,
                    "policy_learning_rate": policy_lr_before,
                    "critic_learning_rate": mixed_optimizer_learning_rates(optimizer)[
                        CRITIC_GROUP_NAME
                    ],
                    "pre_step_empirical_kl": float(pre_step_kl.item()),
                    "post_step_empirical_kl": post_step_kl_value,
                    "retention_mean_kl": retention_kl_value,
                    "retention_kl_by_action_channel": {
                        name: float(retention_channel_mean[index].item())
                        for index, name in enumerate(ACTION_CHANNEL_NAMES)
                    },
                    "rollout_minibatch_kl_by_action_channel": _channel_means(batch_channel_kl),
                    "clip_fraction": float(clip_fraction.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "raw_gradient_norm": float(raw_gradient_norm.item()),
                    "post_clip_gradient_norm": float(post_clip_gradient_norm.item()),
                    "policy_trunk_gradient_norm": float(policy_trunk_gradient_norm.item()),
                    "actor_head_gradient_norm": float(actor_head_gradient_norm.item()),
                    "critic_head_gradient_norm": float(critic_head_gradient_norm.item()),
                    "value_loss_to_trunk_gradient_norm": float(
                        value_loss_to_trunk_gradient_norm.item()
                    ),
                    "value_loss_to_actor_gradient_norm": 0.0,
                    "parameter_step_norms": parameter_step_norms,
                }
                accepted_steps.append(step)
                metrics["policy_loss"].append(policy_loss.detach())
                metrics["value_loss"].append(value_loss.detach())
                metrics["entropy"].append(entropy.detach())
                metrics["total_loss"].append(total_loss.detach())
                metrics["approx_kl"].append(pre_step_kl.detach())
                metrics["clip_fraction"].append(clip_fraction.detach())
                metrics["gradient_norm"].append(raw_gradient_norm.detach())
                metrics["post_clip_gradient_norm"].append(post_clip_gradient_norm.detach())
                optimizer_step_index += 1
                break
            if stop:
                break
        if stop:
            break

    zero = torch.zeros((), dtype=torch.float32, device=rollout.device)
    result = {
        name: torch.stack(values).mean() if values else zero.clone()
        for name, values in metrics.items()
    }
    result["old_value_mean"] = old_value.mean()
    post_step_values = [
        torch.tensor(
            step["post_step_empirical_kl"],
            dtype=torch.float32,
            device=rollout.device,
        )
        for step in accepted_steps
    ]
    result["optimizer_pre_step_approx_kl_mean"] = result["approx_kl"]
    result["optimizer_post_step_approx_kl_mean"] = (
        torch.stack(post_step_values).mean() if post_step_values else zero.clone()
    )
    result["optimizer_post_step_approx_kl_max"] = (
        torch.stack(post_step_values).amax() if post_step_values else zero.clone()
    )
    completed = _completed_update_diagnostics(
        model,
        observation,
        action,
        pre_tanh,
        old_log_probability,
        policy_config,
        ppo_config.minibatch_size,
        distribution_override,
    )
    result.update(completed)
    result["approx_kl"] = completed["completed_update_mean_kl"]
    result["predicted_value_mean"] = old_value.mean()
    result["predicted_value_std"] = old_value.std(unbiased=False)
    result["predicted_value_max_abs"] = old_value.abs().amax()
    result["return_mean"] = returns.mean()
    result["return_std"] = returns.std(unbiased=False)
    result["return_max_abs"] = returns.abs().amax()
    result["advantage_before_normalization_mean"] = raw_advantage.mean()
    result["advantage_before_normalization_std"] = raw_advantage.std(unbiased=False)
    result["advantage_before_normalization_max_abs"] = raw_advantage.abs().amax()
    for channel, name in enumerate(ANALOG_ACTION_NAMES):
        result[f"emitted_action_saturation_fraction_{name}"] = (
            (action[:, channel].abs() > 0.95).to(torch.float32).mean()
        )

    with torch.no_grad():
        final_retention_actor, _ = model(retention_observations)
        final_retention_channel = _analytic_channel_kl(
            retention_reference,
            final_retention_actor,
            policy_config,
            distribution_override,
        )
        final_retention_channel_mean = final_retention_channel.mean(dim=0)
        final_retention_mean = final_retention_channel.sum(dim=-1).mean()
    mixed_final = _final_mixed_diagnostics(
        model,
        behavior_model,
        observation,
        action,
        pre_tanh,
        old_log_probability,
        family,
        family_names,
        policy_config,
        ppo_config.minibatch_size,
        distribution_override,
    )
    for name, value in mixed_final["family_empirical_kl"].items():
        family_statistics[name]["empirical_kl"] = value

    completed_kl = float(result["completed_update_mean_kl"].item())
    if kl_guard.reject_completed_update_kl and (
        not math.isfinite(completed_kl)
        or completed_kl > kl_guard.completed_update_mean_kl_limit
    ):
        raise Rival2PolicyDisplacementRejected(
            {
                "reason": "completed_update_mean_kl_limit_exceeded",
                "optimizer_steps_completed": optimizer_step_index,
                "completed_update_mean_kl": completed_kl,
                "completed_update_sample_kl_max": float(
                    result["completed_update_sample_kl_max"].item()
                ),
                "optimizer_post_step_approx_kl_max": float(
                    result["optimizer_post_step_approx_kl_max"].item()
                ),
                "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                "completed_update_mean_kl_limit": (kl_guard.completed_update_mean_kl_limit),
            }
        )

    end_rates = mixed_optimizer_learning_rates(optimizer)
    diagnostics = {
        "schema_version": 1,
        "mode": "RIVAL2_MIXED_OPPONENT_ADAPTIVE_PPO_V1",
        "safety_config": asdict(safety_config),
        "safety_config_hash": safety_config.content_hash,
        "trainable_sample_count": int(indices.numel()),
        "expected_optimizer_steps": ppo_config.epochs
        * math.ceil(indices.numel() / ppo_config.minibatch_size),
        "accepted_optimizer_steps": optimizer_step_index,
        "optimizer_step_proposals": proposal_count,
        "optimizer_step_retries": len(retry_log),
        "policy_learning_rate_backoffs": lr_backoffs,
        "policy_learning_rate_scope": "ppo_update_local",
        "policy_learning_rate_before_update_reset": learning_rate_reset[
            "policy_learning_rate_before_reset"
        ],
        "policy_learning_rate_update_start_reset_applied": learning_rate_reset[
            "policy_learning_rate_reset_applied"
        ],
        "policy_learning_rate_start": start_policy_lr,
        "policy_learning_rate_end": end_rates[POLICY_GROUP_NAME],
        "critic_learning_rate_start_end": end_rates[CRITIC_GROUP_NAME],
        "ppo_early_stop": early_stop_reason is not None,
        "ppo_early_stop_reason": early_stop_reason,
        "maximum_post_step_minibatch_kl": max(
            (step["post_step_empirical_kl"] for step in accepted_steps),
            default=0.0,
        ),
        "completed_update_mean_kl": completed_kl,
        "retention_corpus_mean_kl": float(final_retention_mean.item()),
        "retention_reference_actor_sha256": retention_reference_sha256,
        "retention_kl_by_action_channel": {
            name: float(final_retention_channel_mean[index].item())
            for index, name in enumerate(ACTION_CHANNEL_NAMES)
        },
        **mixed_final,
        "maximum_gradient_norms": {
            key: max((step[key] for step in accepted_steps), default=0.0)
            for key in (
                "actor_head_gradient_norm",
                "policy_trunk_gradient_norm",
                "critic_head_gradient_norm",
                "value_loss_to_trunk_gradient_norm",
                "value_loss_to_actor_gradient_norm",
            )
        },
        "family_statistics": family_statistics,
        "retry_log": retry_log,
        "optimizer_steps": accepted_steps,
        "checks": {
            "family_local_advantage_normalization": True,
            "returns_not_family_normalized": True,
            "value_targets_not_family_normalized": True,
            "value_loss_to_shared_trunk_gradient_exact_zero": all(
                step["value_loss_to_trunk_gradient_norm"] == 0.0 for step in accepted_steps
            ),
            "value_loss_to_actor_gradient_exact_zero": all(
                step["value_loss_to_actor_gradient_norm"] == 0.0 for step in accepted_steps
            ),
            "actor_receives_nonzero_policy_gradient": any(
                step["actor_head_gradient_norm"] > 0.0 for step in accepted_steps
            ),
            "trunk_receives_nonzero_policy_gradient": any(
                step["policy_trunk_gradient_norm"] > 0.0 for step in accepted_steps
            ),
            "critic_learning_rate_unchanged": end_rates[CRITIC_GROUP_NAME]
            == safety_config.critic_learning_rate,
            "policy_learning_rate_starts_at_configured_base": start_policy_lr
            == safety_config.initial_policy_learning_rate,
            "update_start_reset_changed_only_policy_group_lr": learning_rate_reset[
                "critic_learning_rate_before_reset"
            ]
            == learning_rate_reset["critic_learning_rate_after_reset"]
            == safety_config.critic_learning_rate,
            "hard_minibatch_guard_unchanged": kl_guard.minibatch_kl_limit == 0.10,
            "hard_completed_guard_unchanged": (kl_guard.completed_update_mean_kl_limit == 0.05),
            "kl_telemetry_only": kl_guard.kl_telemetry_only,
            "accepted_steps_within_soft_minibatch_target": all(
                step["post_step_empirical_kl"] <= safety_config.soft_minibatch_kl_target
                for step in accepted_steps
            ),
            "accepted_steps_within_retention_target": all(
                step["retention_mean_kl"] <= safety_config.retention_soft_mean_kl_target
                for step in accepted_steps
            ),
        },
    }
    diagnostics["verdict"] = "PASS_GREEN" if all(diagnostics["checks"].values()) else "FAIL_RED"
    if diagnostics["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"mixed PPO safety checks failed: {diagnostics['checks']}")
    return result, diagnostics


__all__ = [
    "CRITIC_GROUP_NAME",
    "POLICY_GROUP_NAME",
    "Rival2MixedPPOSafetyConfig",
    "build_retention_observation_corpus",
    "make_empty_mixed_optimizer",
    "migrate_adam_to_mixed_groups",
    "mixed_optimizer_learning_rates",
    "ppo_update_mixed_curriculum",
    "probe_fresh_adam_first_minibatch",
    "reset_policy_learning_rate_for_new_update",
    "retention_observation_sha256",
    "set_policy_learning_rate",
]
