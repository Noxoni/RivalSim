"""GPU-native five-minute 1v1 training lifecycle and reward accounting."""

from __future__ import annotations

import warp as wp

from rivalsim.kernels.rival2 import (
    GOAL_PROGRESS_SCALE_Y,
    NO_TOUCH_TIMEOUT_TICKS,
)
from rivalsim.rival2_contracts import (
    SCORING_DEMOLITION_REWARD,
    SCORING_PROGRESS_COEFFICIENT,
    SCORING_TOUCH_REWARD,
)

REGULATION_TICKS = 5 * 60 * 120
REWARD_BASE = 0
REWARD_GOAL_ONLY = 1
REWARD_SCORING = 2


@wp.kernel(enable_backward=False)
def rival2_full_match_accumulate_tick(
    reward_mode: int,
    physics_ticks_per_decision: int,
    ball_pos: wp.array(dtype=wp.vec3),
    goal_scored: wp.array(dtype=wp.int32),
    scoring_team: wp.array(dtype=wp.int32),
    car_a_hit_this_tick: wp.array(dtype=wp.int32),
    car_b_hit_this_tick: wp.array(dtype=wp.int32),
    bump_event_count: wp.array(dtype=wp.int32),
    bump_event_bumper: wp.array(dtype=wp.int32),
    bump_event_victim: wp.array(dtype=wp.int32),
    bump_event_is_demo: wp.array(dtype=wp.int32),
    interval_tick: wp.array(dtype=wp.int32),
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    ball_y_before: wp.array(dtype=wp.float32),
    ball_y_after: wp.array(dtype=wp.float32),
    touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    demo_by_count: wp.array(dtype=wp.int32),
    demoed_event: wp.array(dtype=wp.int32),
    goal_latched: wp.array(dtype=wp.int32),
    scoring_team_latched: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    physics_reset_mask: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    regulation_ticks_remaining: wp.array(dtype=wp.int32),
    blue_score: wp.array(dtype=wp.int32),
    orange_score: wp.array(dtype=wp.int32),
    overtime: wp.array(dtype=wp.int32),
    match_done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    pending_kickoff_reset: wp.array(dtype=wp.int32),
    match_goal_count: wp.array(dtype=wp.int32),
    match_blue_touches: wp.array(dtype=wp.int32),
    match_orange_touches: wp.array(dtype=wp.int32),
    kickoff_segment_active: wp.array(dtype=wp.int32),
    kickoff_segment_ticks: wp.array(dtype=wp.int32),
    kickoff_segments_total: wp.array(dtype=wp.int32),
    no_touch_segments_total: wp.array(dtype=wp.int32),
    completed_matches: wp.array(dtype=wp.int32),
    completed_blue_wins: wp.array(dtype=wp.int32),
    completed_orange_wins: wp.array(dtype=wp.int32),
    completed_overtime_matches: wp.array(dtype=wp.int32),
    completed_blue_goals: wp.array(dtype=wp.int32),
    completed_orange_goals: wp.array(dtype=wp.int32),
    completed_blue_touches: wp.array(dtype=wp.int32),
    completed_orange_touches: wp.array(dtype=wp.int32),
    completed_match_goals: wp.array(dtype=wp.int32),
    completed_match_ticks: wp.array(dtype=wp.int32),
):
    """Accumulate one physics tick without ever truncating for no-touch age."""

    env = wp.tid()
    car_base = env * 2
    active = match_done[env] == 0
    touched_a = wp.int32(0)
    touched_b = wp.int32(0)

    if active:
        reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
        reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
        touched_a = wp.int32(
            reports_a != 0 and touch_contact_latched[car_base] == 0
        )
        touched_b = wp.int32(
            reports_b != 0 and touch_contact_latched[car_base + 1] == 0
        )
        touch_contact_latched[car_base] = reports_a
        touch_contact_latched[car_base + 1] = reports_b
        if touched_a != 0:
            touch_count[car_base] = touch_count[car_base] + 1
            match_blue_touches[env] = match_blue_touches[env] + 1
        if touched_b != 0:
            touch_count[car_base + 1] = touch_count[car_base + 1] + 1
            match_orange_touches[env] = match_orange_touches[env] + 1
        touched = touched_a != 0 or touched_b != 0
        if touched:
            no_touch_ticks[env] = 0
        else:
            no_touch_ticks[env] = no_touch_ticks[env] + 1

        if kickoff_segment_active[env] != 0:
            next_segment_tick = kickoff_segment_ticks[env] + 1
            kickoff_segment_ticks[env] = next_segment_tick
            if touched:
                kickoff_segments_total[env] = kickoff_segments_total[env] + 1
                kickoff_segment_active[env] = 0
            elif next_segment_tick >= NO_TOUCH_TIMEOUT_TICKS:
                kickoff_segments_total[env] = kickoff_segments_total[env] + 1
                no_touch_segments_total[env] = no_touch_segments_total[env] + 1
                kickoff_segment_active[env] = 0

        event_base = env * 4
        count = bump_event_count[env]
        for relative in range(4):
            if relative < count:
                event = event_base + relative
                if bump_event_is_demo[event] != 0:
                    bumper = bump_event_bumper[event]
                    victim = bump_event_victim[event]
                    if bumper >= 0 and bumper < 2:
                        demo_by_count[car_base + bumper] = (
                            demo_by_count[car_base + bumper] + 1
                        )
                    if victim >= 0 and victim < 2:
                        demoed_event[car_base + victim] = 1

        if goal_scored[env] != 0 and goal_latched[env] == 0:
            scorer = scoring_team[env]
            goal_latched[env] = 1
            scoring_team_latched[env] = scorer
            match_goal_count[env] = match_goal_count[env] + 1
            if scorer == 0:
                blue_score[env] = blue_score[env] + 1
            else:
                orange_score[env] = orange_score[env] + 1
            if overtime[env] != 0:
                match_done[env] = 1
                winner[env] = scorer
            else:
                pending_kickoff_reset[env] = 1

        episode_ticks[env] = episode_ticks[env] + 1
        ball_y_after[env] = ball_pos[env][1]

        if overtime[env] == 0:
            remaining = regulation_ticks_remaining[env] - 1
            regulation_ticks_remaining[env] = remaining
            if remaining == 0:
                if blue_score[env] == orange_score[env]:
                    overtime[env] = 1
                    pending_kickoff_reset[env] = 1
                else:
                    match_done[env] = 1
                    winner[env] = wp.int32(
                        blue_score[env] < orange_score[env]
                    )
                    pending_kickoff_reset[env] = 0

        if match_done[env] != 0:
            completed_matches[env] = completed_matches[env] + 1
            if winner[env] == 0:
                completed_blue_wins[env] = completed_blue_wins[env] + 1
            else:
                completed_orange_wins[env] = completed_orange_wins[env] + 1
            if overtime[env] != 0:
                completed_overtime_matches[env] = (
                    completed_overtime_matches[env] + 1
                )
            completed_blue_goals[env] = completed_blue_goals[env] + blue_score[env]
            completed_orange_goals[env] = (
                completed_orange_goals[env] + orange_score[env]
            )
            completed_blue_touches[env] = (
                completed_blue_touches[env] + match_blue_touches[env]
            )
            completed_orange_touches[env] = (
                completed_orange_touches[env] + match_orange_touches[env]
            )
            completed_match_goals[env] = (
                completed_match_goals[env] + match_goal_count[env]
            )
            completed_match_ticks[env] = (
                completed_match_ticks[env] + episode_ticks[env]
            )

    next_interval_tick = interval_tick[env] + 1
    interval_tick[env] = next_interval_tick
    if next_interval_tick == physics_ticks_per_decision:
        terminal = match_done[env]
        terminated[env] = terminal
        truncated[env] = 0
        reset_mask[env] = terminal
        physics_reset_mask[env] = wp.int32(
            terminal != 0 or pending_kickoff_reset[env] != 0
        )

        blue_reward = 0.0
        orange_reward = 0.0
        if reward_mode == REWARD_BASE:
            blue_reward = (
                0.5
                * (ball_y_after[env] - ball_y_before[env])
                / GOAL_PROGRESS_SCALE_Y
            )
            blue_reward = blue_reward + 0.05 * float(
                touch_count[car_base] - touch_count[car_base + 1]
            )
            blue_reward = blue_reward + 0.10 * float(
                demo_by_count[car_base] - demo_by_count[car_base + 1]
            )
            if goal_latched[env] != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                else:
                    blue_reward = blue_reward - 10.0
            orange_reward = -blue_reward
        elif reward_mode == REWARD_SCORING:
            progress_reward = (
                SCORING_PROGRESS_COEFFICIENT
                * (ball_y_after[env] - ball_y_before[env])
                / GOAL_PROGRESS_SCALE_Y
            )
            blue_reward = progress_reward
            orange_reward = -progress_reward
            blue_reward = blue_reward + SCORING_TOUCH_REWARD * float(
                touch_count[car_base]
            )
            orange_reward = orange_reward + SCORING_TOUCH_REWARD * float(
                touch_count[car_base + 1]
            )
            demo_reward = SCORING_DEMOLITION_REWARD * float(
                demo_by_count[car_base] - demo_by_count[car_base + 1]
            )
            blue_reward = blue_reward + demo_reward
            orange_reward = orange_reward - demo_reward
            if goal_latched[env] != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                    orange_reward = orange_reward - 10.0
                else:
                    blue_reward = blue_reward - 10.0
                    orange_reward = orange_reward + 10.0
        else:
            if goal_latched[env] != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                    orange_reward = orange_reward - 10.0
                else:
                    blue_reward = blue_reward - 10.0
                    orange_reward = orange_reward + 10.0
        reward[car_base] = blue_reward
        reward[car_base + 1] = orange_reward


@wp.kernel(enable_backward=False)
def rival2_full_match_after_reset(
    physics_reset_mask: wp.array(dtype=wp.int32),
    match_reset_mask: wp.array(dtype=wp.int32),
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    kickoff_indicator: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    demo_by_count: wp.array(dtype=wp.int32),
    demoed_event: wp.array(dtype=wp.int32),
    previous_action: wp.array(dtype=wp.float32),
    regulation_ticks_remaining: wp.array(dtype=wp.int32),
    blue_score: wp.array(dtype=wp.int32),
    orange_score: wp.array(dtype=wp.int32),
    overtime: wp.array(dtype=wp.int32),
    match_done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    pending_kickoff_reset: wp.array(dtype=wp.int32),
    match_goal_count: wp.array(dtype=wp.int32),
    match_blue_touches: wp.array(dtype=wp.int32),
    match_orange_touches: wp.array(dtype=wp.int32),
    kickoff_segment_active: wp.array(dtype=wp.int32),
    kickoff_segment_ticks: wp.array(dtype=wp.int32),
    lifecycle_blue_score: wp.array(dtype=wp.int32),
    lifecycle_orange_score: wp.array(dtype=wp.int32),
):
    """Finish a goal kickoff or start the next complete match."""

    env = wp.tid()
    if physics_reset_mask[env] == 0:
        return

    no_touch_ticks[env] = 0
    kickoff_indicator[env] = 1
    pending_kickoff_reset[env] = 0
    kickoff_segment_active[env] = 1
    kickoff_segment_ticks[env] = 0
    car_base = env * 2
    for local_car in range(2):
        car = car_base + local_car
        touch_count[car] = 0
        touch_contact_latched[car] = 0
        demo_by_count[car] = 0
        demoed_event[car] = 0
        action_base = car * 8
        for channel in range(8):
            previous_action[action_base + channel] = 0.0

    if match_reset_mask[env] != 0:
        episode_ticks[env] = 0
        regulation_ticks_remaining[env] = REGULATION_TICKS
        blue_score[env] = 0
        orange_score[env] = 0
        overtime[env] = 0
        match_done[env] = 0
        winner[env] = -1
        match_goal_count[env] = 0
        match_blue_touches[env] = 0
        match_orange_touches[env] = 0
        lifecycle_blue_score[env] = 0
        lifecycle_orange_score[env] = 0

    physics_reset_mask[env] = 0


__all__ = [
    "REGULATION_TICKS",
    "REWARD_BASE",
    "REWARD_GOAL_ONLY",
    "REWARD_SCORING",
    "rival2_full_match_accumulate_tick",
    "rival2_full_match_after_reset",
]
