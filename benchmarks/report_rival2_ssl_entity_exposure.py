"""CPU-only reduction of frozen mixed-opponent training logs; no policy imports.

Recover goal-ending counts by opponent family from conservation identities.
These are stochastic curriculum episodes, NOT regulation-match win rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/ssl_entity_continuation_v1"
CAPACITY = 32768 * 90 * 2


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid count: {name}")
    if value < 0 or not float(value).is_integer():
        raise ValueError(f"invalid count: {name}")
    return int(value)


def decompose(training):
    """Self-play trains two sides; Nexto games train exactly one current side.

    P=S+W+L, G=S+W, C=S+L, where P is physical goals, G/C are
    learner goals/concedes, S self-play goals, W/L current-vs-Nexto outcomes.
    """
    p, g, c = [count(training[k], k) for k in ("physical_goals", "goals", "concedes")]
    s, w, loss = g + c - p, p - c, p - g
    if min(s, w, loss) < 0 or s + w + loss != p:
        raise ValueError("inconsistent physical/learner goal ledger")
    samples = count(training["trainable_agent_samples"], "samples")
    nexto = count(training["nexto_training_sample_count"], "nexto samples")
    if samples + nexto != CAPACITY or not 0 <= nexto <= samples:
        raise ValueError("inconsistent learner/opponent sample ledger")
    if p > count(training["resets"], "resets"):
        raise ValueError("goals exceed physical episode endings")
    if not nexto and (w or loss):
        raise ValueError("Nexto goal endings without Nexto exposure")
    return dict(
        trainable_decisions=samples,
        nexto_learner_decisions=nexto,
        selfplay_physical_goal_endings=s,
        nexto_learner_goal_endings=w,
        nexto_learner_concede_endings=loss,
        physical_goal_endings=p,
    )


def summarize(rows):
    if not rows:
        raise ValueError("empty window")
    entries = [decompose(row["training"]) for row in rows]
    totals = {k: sum(e[k] for e in entries) for k in entries[0]}
    nexto_goals = totals["nexto_learner_goal_endings"] + totals["nexto_learner_concede_endings"]
    return dict(
        first_update=rows[0]["accepted_updates"],
        last_update=rows[-1]["accepted_updates"],
        **totals,
        nexto_decision_fraction=totals["nexto_learner_decisions"] / totals["trainable_decisions"],
        nexto_fraction_of_goal_endings_scored_by_learner=(
            totals["nexto_learner_goal_endings"] / nexto_goals if nexto_goals else None
        ),
        regulation_match_win_rate=None,
    )


def report(prefix, through):
    if through < 150 or through % 50:
        raise ValueError("use a completed scheduled boundary >=150")
    rows = [json.loads(line) for line in prefix.splitlines()]
    if [r["accepted_updates"] for r in rows] != list(range(101, through + 1)):
        raise ValueError("noncontiguous or wrong frozen prefix")
    return dict(
        version="RIVAL2_ENTITY_EXPOSURE_REDUCTION_V1",
        through_update=through,
        source_sha256=hashlib.sha256(prefix).hexdigest().upper(),
        blocks=[summarize(rows[i : i + 50]) for i in range(0, len(rows), 50)],
        total=summarize(rows),
        derivation={
            "selfplay_goals": "learner_goals + learner_concedes - physical_goals",
            "nexto_learner_goals": "physical_goals - learner_concedes",
            "nexto_learner_concedes": "physical_goals - learner_goals",
        },
        semantics=[
            "Derived exact counts from existing counters, not newly observed goal events.",
            "Requires both self-play agents train; only current agent trains against Nexto.",
            "Training uses sampled actions and changing curriculum states; not fixed evaluation.",
            "20 percent episode assignment is not 20 percent learner sample exposure.",
            "Goal-ending fractions exclude no-touch and time-limit episodes, and are not match wins.",  # noqa: E501
            "No scenario-family attribution or cause of natural-match regression is inferred.",
        ],
        optimizer_steps_performed=0,
        runtime_changed=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, required=True)
    args = parser.parse_args()
    source = RESULTS / f"training_curve_through_{args.through:06d}.jsonl"
    result = report(source.read_bytes().replace(b"\r\n", b"\n"), args.through)
    destination = RESULTS / f"training_exposure_{args.through:06d}.json"
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("existing report differs; preserve prior evidence")
    else:
        destination.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
