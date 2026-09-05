"""Read-only, count-weighted training telemetry; not deterministic evaluation."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_fresh_ground_30hz_v1 import write_json
from benchmarks.run_rival2_ssl_entity_joint_control import EXTERNAL, RESULTS, verify
from third_party.nexto.adapter import build_action_table


def summarize(rows, table):
    if not rows:
        raise ValueError("empty window")
    counts = [0] * 90
    for row in rows:
        tr, ppo = row["training"], row["ppo"]
        count = tr["action_index_counts"]
        assert len(count) == 90 and all(isinstance(c, int) and c >= 0 for c in count)
        assert sum(count) == tr["trainable_agent_samples"] == 5898240
        assert tr["current_selfplay_only"] and tr["kl_telemetry_only"]
        assert ppo["optimizer_steps"] == 182 and ppo["kl_rejections"] == 0
        assert all(math.isfinite(x) for x in ppo.values() if isinstance(x, (int, float)))
        for j, key in enumerate(("jump", "boost", "handbrake"), 5):
            assert sum(count[i] * table[i][j] for i in range(90)) == tr[key]
        counts = [a + b for a, b in zip(counts, count, strict=True)]

    def total(key):
        return sum(r["training"][key] for r in rows)

    def divide(a, b):
        return None if b == 0 else a / b

    samples = total("trainable_agent_samples")
    player_minutes = samples / 1800
    world_minutes = player_minutes / 2
    return dict(
        first_update=rows[0]["accepted_updates"],
        last_update=rows[-1]["accepted_updates"],
        updates=len(rows),
        player_decision_samples=samples,
        simulated_world_minutes=world_minutes,
        simulated_player_minutes=player_minutes,
        touches=total("touches"),
        touches_per_player_minute=total("touches") / player_minutes,
        goals=total("goals"),
        goals_per_world_minute=total("goals") / world_minutes,
        world_resets=total("resets"),
        no_touch_resets=total("no_touch"),
        no_touch_fraction_of_world_resets=divide(total("no_touch"), total("resets")),
        ended_player_episodes=total("ended_player_episodes"),
        ended_player_episode_touch_fraction=divide(
            total("episodes_with_touch"), total("ended_player_episodes")
        ),
        first_touch_seconds_conditional=divide(total("first_touch_age"), total("first_touches")),
        goalward_touch_fraction=divide(total("goalward_touches"), total("touches")),
        mean_speed_uu_per_second=total("speed") / samples,
        categorical_entropy_nats=total("entropy") / samples,
        jump_fraction=total("jump") / samples,
        boost_fraction=total("boost") / samples,
        handbrake_fraction=total("handbrake") / samples,
        action_index_counts=counts,
        max_completed_update_mean_kl=max(r["ppo"]["completed_update_mean_kl"] for r in rows),
        max_completed_update_sample_kl=max(
            r["ppo"]["completed_update_sample_kl_max"] for r in rows
        ),
        kl_rejections=0,
        optimizer_steps=182 * len(rows),
        mean_rollout_seconds=sum(r["rollout_seconds"] for r in rows) / len(rows),
        mean_ppo_seconds=sum(r["ppo_seconds"] for r in rows) / len(rows),
        potential_signed_sums={
            k: sum(r["training"]["potential_reward_components"][k] for r in rows)
            for k in rows[0]["training"]["potential_reward_components"]
        },
    )


def contiguous_prefix(lines, boundary):
    result = []
    for line in lines:
        if not line.endswith(b"\n"):
            break  # A writer may still be appending its last line.
        row = json.loads(line)
        if row["accepted_updates"] > boundary:
            break
        result.append(row)
    if [r["accepted_updates"] for r in result] != list(range(1, boundary + 1)):
        raise ValueError("training prefix must contain every accepted update exactly once")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, required=True)
    args = parser.parse_args()
    if args.through not in range(10, 101, 10):
        raise ValueError("report fixed complete ten-update blocks only")
    verify(published=True)
    lines = (EXTERNAL / "training_curve.jsonl").read_bytes().splitlines(keepends=True)
    rows = contiguous_prefix(lines, args.through)
    table = build_action_table("cpu").tolist()
    blocks = [summarize(rows[i : i + 10], table) for i in range(0, len(rows), 10)]
    report = dict(
        through_update=args.through,
        source_prefix_sha256=hashlib.sha256(b"".join(lines[: args.through])).hexdigest().upper(),
        source_rows=rows,
        blocks=blocks,
        complete_prefix=summarize(rows, table),
        optimizer_or_policy_changed=False,
        interpretation=[
            "Stochastic current-self-play curriculum experience, "
            "not fixed-case evaluation or match strength.",
            "Ratios are recomputed from raw counts, not averaged per-update ratios.",
            "Initial reset synchronization and changing episode composition "
            "confound short-window trends.",
            "No-touch ages are per world; an earlier-touch player episode can later time out. "
            "Counts are not complements.",
            "Both sides train: each world goal is one player goal and one concede, "
            "not two physical goals.",
            "Antisymmetric potentials and terminal reward cancel in signed two-player sums; "
            "zero sum is not missing reward.",
            "Conditional first-touch time excludes episodes without any first touch.",
            "KL magnitude is telemetry, never an acceptance gate.",
        ],
    )
    write_json(RESULTS / f"training_summary_{args.through:03d}.json", report)
    for block in blocks:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in block.items()
                    if k not in ("action_index_counts", "potential_signed_sums")
                }
            )
        )


if __name__ == "__main__":
    main()
