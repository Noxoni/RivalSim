"""CPU reduction of the frozen +550 diagnostic; no model/optimizer or simulator."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.diagnose_rival2_direct_skills_learning_signal import distribution, independent_gae
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, write_json
from rivalsim.direct_skills_v1 import NAMES
from rivalsim.fresh_ground_30hz import GAE_LAMBDA, GAMMA, PHYSICS_GAMMA, WEIGHTS

DIRECTORY = ROOT / "results/rival2/direct_skills_v1/advantage_diagnostic_000550"


def affine_fit(x, y):
    """Descriptive two-parameter calibration, never applied to a checkpoint."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def reduce():
    summary = json.loads((DIRECTORY / "summary.json").read_text())
    keep = (
        "train_mask",
        "opponent_family",
        "roles",
        "entropy",
        "maximum_probability",
        "rewards",
        "values",
        "returns",
        "advantages",
        "normalized_advantage",
        "terminated",
        "truncated",
        "scoring_team",
    )
    buffers, checks, counts = [], [], []
    for i, item in enumerate(summary["rollouts"]):
        path = DIRECTORY / f"rollout_{i:02d}.npz"
        assert sha(path) == item["archive_sha256"]
        with np.load(path) as archive:
            a = {k: archive[k] for k in keep}
            terminal = a["terminated"]
            win = a["scoring_team"][..., None] == np.arange(2)
            expected_goal = np.where(win, 10.0, -10.0) * PHYSICS_GAMMA ** np.maximum(
                archive["first_goal_tick"][..., None], 0
            )
            expected_goal *= terminal
            goal_error = float(np.abs(expected_goal - archive["component.terminal_goal"]).max())
            natural = sum(archive["component." + name] for name in WEIGHTS)
            reward_error = float(
                np.abs(
                    a["rewards"] - (archive["component.terminal_goal"] + natural + archive["bonus"])
                ).max()
            )
            gae = independent_gae(
                a["rewards"],
                a["values"],
                archive["next_values"],
                terminal,
                a["truncated"],
                GAMMA,
                GAE_LAMBDA,
            )
            assert goal_error < 2e-6 and reward_error < 2e-6
            assert np.array_equal(gae, a["advantages"])
            assert item["same_batch_stepwise_replay"]["collected_log_probability_exactly_equal"]
            assert item["gradients"]["value_loss_actor_gradient_exactly_zero"]
            checks.append(
                dict(
                    rollout=i,
                    terminal_goal_reward_error=goal_error,
                    total_reward_accounting_error=reward_error,
                    independent_gae_exact=True,
                    same_batch_replay_exact=True,
                    value_loss_actor_gradient_exactly_zero=True,
                )
            )
            counts.append(
                dict(
                    rollout=i,
                    true_terminals=int(terminal.sum()),
                    truncations=int(a["truncated"].sum()),
                    trainable_samples=int(a["train_mask"].sum()),
                )
            )
        a["rollout"] = np.full_like(a["roles"], i)
        buffers.append(a)
    all_data = {k: np.concatenate([a[k] for a in buffers]) for k in buffers[0]}
    a, train = all_data, all_data["train_mask"]
    roles, health, terminal_groups = {}, {}, {}
    terminal = train & a["terminated"]
    win = a["scoring_team"][..., None] == np.arange(2)
    for i, name in enumerate(NAMES):
        for family, label in enumerate(("selfplay", "nexto")):
            mask = train & (a["roles"] == i) & (a["opponent_family"] == family)
            roles[f"{name}_{label}"] = dict(
                samples=int(mask.sum()),
                entropy=distribution(a["entropy"][mask]),
                maximum_probability=distribution(a["maximum_probability"][mask]),
                critic_rmse=float(
                    np.sqrt(np.square(a["values"][mask] - a["returns"][mask]).mean())
                ),
                normalized_advantage=distribution(a["normalized_advantage"][mask]),
            )
    for name, mask in (
        ("all", train),
        ("nexto", train & (a["opponent_family"] == 1)),
        ("selfplay", train & (a["opponent_family"] == 0)),
    ):
        health[name] = dict(
            samples=int(mask.sum()),
            entropy=distribution(a["entropy"][mask]),
            mean_maximum_probability=float(a["maximum_probability"][mask].mean()),
            maximum_probability_above_99_fraction=float(
                (a["maximum_probability"][mask] > 0.99).mean()
            ),
            maximum_probability_above_95_fraction=float(
                (a["maximum_probability"][mask] > 0.95).mean()
            ),
        )
    for name, lane in (("natural", a["roles"] == 0), ("skills", a["roles"] != 0)):
        for outcome, winning in (("goal", win), ("concede", ~win)):
            mask = terminal & lane & winning
            terminal_groups[f"{name}_{outcome}"] = dict(
                reward=distribution(a["rewards"][mask]),
                value=distribution(a["values"][mask]),
                advantage=distribution(a["advantages"][mask]),
                normalized_advantage=distribution(a["normalized_advantage"][mask]),
            )

    # Descriptive held-out calibration only: first6seconds fit, later6seconds score.
    # These are diagnostic buffers, not the frozen human/simulator test corpora.
    fit = train & (a["rollout"] < 2)
    test = train & ~fit
    common = affine_fit(a["values"][fit], a["returns"][fit])
    per_role = {}
    prediction = np.zeros_like(a["returns"], dtype=np.float64)
    for role, name in enumerate(NAMES):
        m = fit & (a["roles"] == role)
        coefficients = affine_fit(a["values"][m], a["returns"][m])
        m = a["roles"] == role
        prediction[m] = coefficients[0] * a["values"][m] + coefficients[1]
        per_role[name] = coefficients.tolist()
    calibration = dict(
        fit_rollouts=[0, 1],
        held_out_rollouts=[2, 3],
        fitted_parameters_per_role=2,
        pooled_coefficients=common.tolist(),
        role_coefficients=per_role,
        original_held_out_mse=float(np.square(a["values"][test] - a["returns"][test]).mean()),
        pooled_held_out_mse=float(
            np.square(common[0] * a["values"][test] + common[1] - a["returns"][test]).mean()
        ),
        role_conditioned_held_out_mse=float(
            np.square(prediction[test] - a["returns"][test]).mean()
        ),
        applied_to_policy=False,
        optimizer_steps=0,
        caveat="Small temporal diagnostic holdout, not randomized independent games. "
        "No role-conditioned neural critic was trained. This is descriptive error structure, "
        "not a counterfactual actor or proof a PPO change will improve play.",
    )
    return dict(
        schema="RIVAL2_DIRECT_SKILLS_SIGNAL_REDUCTION_V1",
        source_sha256=sha(Path(__file__)),
        diagnostic_summary_sha256=sha(DIRECTORY / "summary.json"),
        checks=checks,
        collection_counts=counts,
        health=health,
        roles=roles,
        terminal_groups=terminal_groups,
        descriptive_calibration=calibration,
        maximum_batched_replay_logp_error=max(
            r["gradients"]["collection_replay_max_log_probability_error"]
            for r in summary["rollouts"]
        ),
        batched_replay_clip_exceedances=sum(
            r["gradients"]["replay_comparison"]["ratio_outside_production_clip_count"]
            for r in summary["rollouts"]
        ),
        no_optimizer_or_simulator=True,
        interpretation="Negative advantage on a scoring tick or positive advantage on a "
        "conceding tick is a baseline-relative residual, not causal proof that PPO rewards "
        "conceding. Here the consistent natural/skill differences motivate critic calibration "
        "investigation; reward signs, goal bootstrapping and GAE are verified separately.",
    )


if __name__ == "__main__":
    result = reduce()
    write_json(DIRECTORY / "reduction.json", result)
    print(
        json.dumps({k: result[k] for k in ("health", "terminal_groups", "descriptive_calibration")})
    )
