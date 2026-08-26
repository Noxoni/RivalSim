# Rival 2.0 Final-Checkpoint Behavioral Telemetry

Verdict: **PASS_GREEN**.

The single authorized evaluation completed 16,384 first episodes under ordinary stochastic final-policy self-play. The unchanged final checkpoint controlled both cars; no training or historical opponent was used. It recorded 530,968 unique accepted touch entries and 15,140 goals.

## Immediate touch direction

Using the declared symmetric ±100 uu/s canonical longitudinal threshold, forward touches were 404,277/530,968 (0.761396), lateral-neutral touches were 32,692/530,968 (0.061571), and backward touches were 93,999/530,968 (0.177033). These are descriptive directions, not quality labels.

The raw continuous canonical post-touch Y velocity, longitudinal velocity delta, goal-center heading angles, actual net Y displacement, and 0.25/0.5/1/2-second continuation distributions are in `touch_trajectory_summary.json`. Instantaneous goal-plane intersections are explicitly projections only; an outside-mouth projection is not called a missed shot.

## Possession and continuation

The same player made the next accepted touch after 368,220/530,968 touches (0.693488); the opponent did so after 146,364/530,968 (0.275655). Primitive possession chains are maximal consecutive accepted touches by one player.

Arena-contact continuation rates per touch were ground 0.240514, side wall 0.037812, backboard/goal structure 0.028303, and ceiling 0.020920. Categories come from the dominant axis of retained source collision normals. Wall and backboard paths are not treated as inaccurate.

Touch locations were defensive 0.270775, middle 0.409237, and attacking 0.319987 in the toucher's canonical frame.

## Goals and goal mouth

An interpolated scoring-plane crossing was available for 15,140/15,140 goals. Of valid crossings, 15,140/15,140 were inside the declared ±892.755 by 0..642.775 uu mouth.

`goal_entry_summary.json` contains scoring side, last toucher and timing, event position/velocity, entry speed/angle, final-possession touch count, and exact arena-contact sequences. `goal_mouth_histogram.json` contains absolute and normalized X/Z distributions, exact descriptive third-bin boundaries, and the 2D heatmap counts.

## Authority and non-interference

- Authorized starting HEAD: `df295da1bcaec07170465f22fdc512b66fdd7538`.
- Checkpoint SHA-256: `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`.
- Reward contract: `RIVAL2_REWARD_V1` (unchanged).
- Evaluation seed: `920260826`; worlds: `16,384`.
- Policy/physics rate: 30 Hz / 120 Hz; first completed episode per world.
- Telemetry is a post-tick read-only GPU launch and has no controller, observation, reward, policy, reset, or dynamics output.
- One technical recorder attempt was rejected before publication; the published replay used the identical frozen checkpoint, seeds, policy, and simulator, with only the read-only recorder corrected.
- Raw artifact: `results/rival2/behavioral_telemetry/raw_events_representative_sample.npz`; SHA-256 `CD2D37FB87C25C8ED4F4F6C49E1B1D806850C9F8D7B732B55FAEF1F50047404C`; 2,304,130 bytes.

All percentages in the JSON evidence include their explicit integer numerator and denominator. This is descriptive evidence only; no reward or behavior recommendation was implemented.
