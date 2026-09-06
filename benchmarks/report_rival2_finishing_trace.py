"""Physical description of parity-verified finishing replays; CPU only.

No new reward predicate or mechanic detector. The goal-plane ray is explicitly
constant-velocity geometry, not a claim about a shot's eventual success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def describe(path):
    manifest_path, archive = path / "manifest.json", path / "finishing.npz"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["verdict"] == "PASS" and manifest["exact_saved_evaluation_parity"]
    assert manifest["model_and_checkpoint_unchanged"] and manifest["optimizer_steps"] == 0
    assert digest(archive) == manifest["trace_sha256"]
    with np.load(archive, allow_pickle=False) as stored:
        f = {key: stored[key] for key in stored.files}
    sides, active = f["focal_side"], f["decision.alive"]
    decisions, worlds = active.shape
    cumulative = f["physics.post.interval_touch_count"].reshape(decisions, 4, worlds, 2)
    prior = np.concatenate((np.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), 1)
    contacts = np.maximum(cumulative - prior, 0).reshape(decisions * 4, worlds, 2)
    contacts *= f["physics.valid"][..., None]
    raw = manifest["replay"]["finishing"]["raw"]
    cases = []
    for case, side in enumerate(sides):
        side = int(side)
        own_ticks = np.flatnonzero(contacts[:, case, side])
        opponent_ticks = np.flatnonzero(contacts[:, case, 1 - side])
        assert len(own_ticks) == raw["touches"][case] and len(own_ticks)
        first, last = int(own_ticks[0]), int(own_ticks[0])
        # Final tick of the first uninterrupted native collision, not merely
        # first impulse; raw hit flags and debounced touch events differ.
        while (
            last + 1 < decisions * 4
            and f["physics.valid"][last + 1, case]
            and f["physics.post.hit_this_tick"][last + 1, case, side]
        ):
            last += 1
        sign = np.array([1 if side == 0 else -1, 1 if side == 0 else -1, 1])
        ball = f["physics.post.ball_pos"][last, case].astype(float) * sign
        velocity = f["physics.post.ball_vel"][last, case].astype(float) * sign
        car = f["physics.pre.car_pos"][first, case, side].astype(float) * sign
        pre_ball = f["physics.pre.ball_pos"][first, case].astype(float) * sign
        car_velocity = f["physics.pre.car_vel"][first, case, side].astype(float) * sign
        goal_time = float((5120 - ball[1]) / velocity[1]) if velocity[1] > 0 else None
        ray_x = float(ball[0] + velocity[0] * goal_time) if goal_time is not None else None
        after = opponent_ticks[opponent_ticks > first]
        first_opponent = int(after[0]) if len(after) else None
        end_decision = int(np.flatnonzero(active[:, case])[-1])
        chosen = np.flatnonzero(f["physics.valid"][:, case])
        controls = f["physics.pre.applied_action"][chosen, case, side]
        outcome = (
            "goal" if raw["goals"][case] else "concede" if raw["concedes"][case] else "timeout"
        )
        cases.append(
            dict(
                case=case,
                side=side,
                outcome=outcome,
                end_seconds=raw["seconds"][case],
                first_touch_seconds=(first + 1) / 120,
                contact_exit_seconds=(last + 1) / 120,
                contiguous_contact_ticks=last - first + 1,
                rival_touch_seconds=((own_ticks + 1) / 120).tolist(),
                nexto_touch_seconds=((opponent_ticks + 1) / 120).tolist(),
                first_nexto_after_rival_seconds=(first_opponent + 1) / 120
                if first_opponent is not None
                else None,
                rival_followup_before_nexto=bool(
                    len(own_ticks) > 1 and (first_opponent is None or own_ticks[1] < first_opponent)
                ),
                canonical_first_car_position=car.tolist(),
                canonical_pre_contact_ball_position=pre_ball.tolist(),
                canonical_first_car_velocity=car_velocity.tolist(),
                canonical_ball_minus_car_at_contact=(pre_ball - car).tolist(),
                canonical_contact_exit_ball_position=ball.tolist(),
                canonical_contact_exit_ball_velocity=velocity.tolist(),
                constant_velocity_ray_goal_x=ray_x,
                constant_velocity_ray_goal_seconds=goal_time,
                canonical_nexto_position_at_first_contact=(
                    f["physics.pre.car_pos"][first, case, 1 - side] * sign
                ).tolist(),
                first_contact_rival_action=f["physics.pre.applied_action"][
                    first, case, side
                ].tolist(),
                first_contact_car_quaternion_xyzw=f["physics.pre.car_quat"][
                    first, case, side
                ].tolist(),
                first_contact_car_angular_velocity=f["physics.pre.car_ang_vel"][
                    first, case, side
                ].tolist(),
                first_contact_wheels=f["physics.pre.wheel_contact"][first, case, side].tolist(),
                first_contact_has_flipped=int(f["physics.pre.has_flipped"][first, case, side]),
                first_contact_is_flipping=int(f["physics.pre.is_flipping"][first, case, side]),
                boost_input_fraction=float((controls[:, 6] != 0).mean()),
                jump_input_fraction=float((controls[:, 5] != 0).mean()),
                final_decision_reward=float(f["decision.reward"][end_decision, case, side]),
                reward_sum_undiscounted=float(
                    f["decision.reward"][active[:, case], case, side].sum()
                ),
            )
        )
    assert sum(c["outcome"] == "goal" for c in cases) == sum(raw["goals"])
    assert sum(c["outcome"] == "concede" for c in cases) == sum(raw["concedes"])
    summary = dict(
        cases=len(cases),
        goals=sum(c["outcome"] == "goal" for c in cases),
        concedes=sum(c["outcome"] == "concede" for c in cases),
        timeouts=sum(c["outcome"] == "timeout" for c in cases),
        boost_input_at_first_contact=sum(c["first_contact_rival_action"][6] != 0 for c in cases),
        jump_input_at_first_contact=sum(c["first_contact_rival_action"][5] != 0 for c in cases),
        native_flip_active_at_first_contact=sum(c["first_contact_is_flipping"] != 0 for c in cases),
        subsequent_nexto_contact=sum(
            c["first_nexto_after_rival_seconds"] is not None for c in cases
        ),
        rival_followup_before_nexto=sum(c["rival_followup_before_nexto"] for c in cases),
        scored_without_later_nexto_contact=sum(
            c["outcome"] == "goal" and c["first_nexto_after_rival_seconds"] is None for c in cases
        ),
        conceded_final_reward_range=[
            min(c["final_decision_reward"] for c in cases if c["outcome"] == "concede"),
            max(c["final_decision_reward"] for c in cases if c["outcome"] == "concede"),
        ],
    )
    return dict(
        accepted_updates=manifest["accepted_updates"],
        source=str(path),
        manifest_sha256=digest(manifest_path),
        archive_sha256=digest(archive),
        checkpoint_sha256=manifest["checkpoint_sha256"],
        summary=summary,
        cases=cases,
    )


def report(paths, output):
    reports = [describe(path) for path in paths]
    assert len(reports) == 2 and reports[0]["accepted_updates"] < reports[1]["accepted_updates"]
    changes = []
    for before, after in zip(reports[0]["cases"], reports[1]["cases"], strict=True):
        assert (before["case"], before["side"]) == (after["case"], after["side"])
        if before["outcome"] != after["outcome"]:
            changes.append(
                dict(case=before["case"], before=before["outcome"], after=after["outcome"])
            )
    result = dict(
        version="DIRECT_SKILLS_PHYSICAL_FINISHING_COMPARISON_V1",
        replays=reports,
        outcome_changes=changes,
        optimizer_steps=0,
        extra_policy_runs=0,
        semantics={
            "frame": "Orange positions/velocities rotate180deg; quaternion/angular velocity native",
            "contact": "Native event; final state of first uninterrupted collision episode",
            "ray": "Constant-velocity xy goal-plane projection, not bounce/goalkeeper prediction",
            "followup": "Next distinct car-ball contact identity, not duration of possession",
            "action": "Exact emitted controls, not inferred intention or a mechanic label",
            "outcome": "Native goal/concession or existing skill timeout; no new success detector",
        },
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists():
        assert output.read_text() == text, "refusing to overwrite changed diagnosis"
    else:
        output.write_text(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report((args.baseline, args.candidate), args.output)
    print(
        json.dumps(
            dict(
                summaries=[r["summary"] for r in result["replays"]],
                outcome_changes=result["outcome_changes"],
            ),
            indent=2,
        )
    )
