# Rival 2.0 vs pinned public Nexto — kickoff-free open play

Verdict: **PASS_GREEN**.

This benchmark begins from physically continuous mid-play states and contains no kickoff or reset anywhere in the measured duel. Every base state is replayed four ways to balance physical role, Blue/Orange assignment, and an exact 180-degree team mirror.

## Headline outcome

- Overall: Rival 8,786, Nexto 7,255, draws 343; decisive Rival win rate 54.772%, all-duel Rival win fraction 53.625%.
- Rival as **Blue**: Rival 3,753, Nexto 4,241, draws 198; decisive Rival win rate 46.948%, all-duel Rival win fraction 45.813%.
- Rival as **Orange**: Rival 5,033, Nexto 3,014, draws 145; decisive Rival win rate 62.545%, all-duel Rival win fraction 61.438%.

Blue and Orange are reported separately because the prior full-match benchmark exposed a material team-side scoring asymmetry; the overall figure never replaces these side-specific results.

| Dimension | Stratum | Duels | Rival wins | Nexto wins | Draws | Decisive Rival win rate |
|---|---|---:|---:|---:|---:|---:|
| state_orientation | original | 8,192 | 4,438 | 3,608 | 146 | 55.158% |
| state_orientation | mirrored | 8,192 | 4,348 | 3,647 | 197 | 54.384% |
| source_policy | rival_stochastic_self_play | 8,192 | 4,115 | 3,906 | 171 | 51.303% |
| source_policy | nexto_deterministic_self_play | 8,192 | 4,671 | 3,349 | 172 | 58.242% |
| initial_physical_role_inherited_by_rival | original_blue_car | 8,192 | 4,377 | 3,628 | 187 | 54.678% |
| initial_physical_role_inherited_by_rival | original_orange_car | 8,192 | 4,409 | 3,627 | 156 | 54.866% |
| field_third | blue_defensive_third | 3,356 | 1,674 | 1,630 | 52 | 50.666% |
| field_third | midfield_third | 9,672 | 5,425 | 4,025 | 222 | 57.407% |
| field_third | orange_defensive_third | 3,356 | 1,687 | 1,600 | 69 | 51.323% |
| height_bin | high_1000_plus_uu | 2,048 | 1,190 | 809 | 49 | 59.530% |
| height_bin | low_0_to_300_uu | 7,780 | 3,981 | 3,652 | 147 | 52.155% |
| height_bin | middle_300_to_1000_uu | 6,556 | 3,615 | 2,794 | 147 | 56.405% |
| initial_closest_policy | Nexto | 8,192 | 4,212 | 3,799 | 181 | 52.578% |
| initial_closest_policy | Rival | 8,192 | 4,574 | 3,456 | 162 | 56.961% |
| boost_bin | Rival_leads_by_more_than_25 | 908 | 569 | 317 | 22 | 64.221% |
| boost_bin | Rival_trails_by_more_than_25 | 908 | 500 | 399 | 9 | 55.617% |
| boost_bin | within_25 | 14,568 | 7,717 | 6,539 | 312 | 54.132% |

## Four-way paired-state control

- Complete families without a draw: `3,770/4,096`.
- Families where draws prevent a complete four-way decision: `326`.
- Rival wins 4 of 4: `480`.
- Rival wins 3 of 4: `986`.
- Rival wins 2 of 4: `1,362`.
- Rival wins 1 of 4: `657`.
- Rival wins 0 of 4: `285`.

## Open-play behavior

| Policy | Touches | Touch share | First touch share | Same next touch | Opponent handoff | Wall continuations | Backboard continuations | Demos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rival | 359,316 | 0.638768 | 0.584936 | 0.888366 | 0.111634 | 0.067397 | 0.021334 | 1,304 |
| Nexto | 203,198 | 0.361232 | 0.415064 | 0.805679 | 0.194321 | 0.025415 | 0.030612 | 522 |

The machine-readable telemetry also contains forward/neutral/backward immediate direction and net displacement, goal-entry X/Z histograms, final-touch-to-goal distributions, scorer/last-toucher agreement, and the same policy-separated metrics for Rival-as-Blue and Rival-as-Orange. These are descriptive categories; backward, wall, and backboard play are not labeled inherently bad.

## State bank and integrity

- Base states: `4,096`: `2,048` stochastic final-45B Rival self-play and `2,048` deterministic pinned-Nexto self-play.
- Capture rule: `For each seeded source world and episode, prospectively choose age 600 + ((world_index*1103515245 + reset_count*12345 + seed) & 0x7fffffff) % 601; capture the first subsequent 30 Hz boundary satisfying every eligibility condition. The rule never observes policy advantage or duel outcome.`.
- Full continuation fields: `295`; compressed bank SHA-256 `E11BBE510F3C9C2709DA56EB1371733C605278C7C969B53FBFF278F2B8EA3C62`.
- Each capture is at least 600 active physics ticks old, follows an accepted touch, has two active cars, is inside the scoring plane, and has no pending reset.
- The one-decision policy-memory boundary is neutral: Rival and Nexto previous actions are all zeros; physical boost/demo/jump/flip/pad/lifecycle timers are preserved.
- Mirror involution max error: `0.0`; failed fields `0`.
- Capture/restore exact-field audit: `270` fields, max float error `0.0`, failed fields `0`.
- Duel-loop profiled H2D/D2H events: `0`; runtime `2,445,565.93` world-ticks/s; peak CUDA `1.656` GiB.
- Actual kickoff/reset events after restored start: `0` / `0`.

## Frozen identities

- Rival checkpoint SHA-256: `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`; policy version `5403`, cumulative samples `45,323,649,024`.
- Nexto upstream commit: `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`; model SHA-256 `BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`.
- Pinned Nexto material remains under CC BY-NC-SA 4.0 and is unchanged by this benchmark.

## Evidence

- `results/rival2/nexto_open_play/summary.json`
- `results/rival2/nexto_open_play/outcomes.json`
- `results/rival2/nexto_open_play/behavioral_telemetry.json`
- `results/rival2/nexto_open_play/state_bank_description.json`
- `results/rival2/nexto_open_play/state_bank.npz`
- `results/rival2/nexto_open_play/per_duel_ledger.csv`
- `results/rival2/nexto_open_play/paired_family_ledger.csv`
- `results/rival2/nexto_open_play/paired_summary.json`
- `results/rival2/nexto_open_play/evidence_manifest.json`

## Explicitly deferred

Fake-kickoff curriculum work—including retreat/backflip-to-boost opponents that intentionally concede first contact—is recorded as future work only. No training, reward/PPO/model/physics/controller change, viewer work, v0.6 work, or fake-kickoff implementation occurred here.
