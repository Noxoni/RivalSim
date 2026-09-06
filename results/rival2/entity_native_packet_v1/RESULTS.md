# Native entity-policy comparison: severe partial-game failure

Date: 2026-09-06. Protocol published at
`c6c4f6d08276a80f16e558b33aa322a21a715edb`, before the native launch.
Authority file SHA-256:
`A4683A18B62BAD4280854ACD42731367BA0083EE024D9A78F6E51597DA171EDD`.

## Outcome, not an inference-parity success claim

The actual Rocket League local game put reference600 on Blue against the
installed native Nexto on Orange. Rival trailed **0-37** when the prospectively
fixed 600-second wall-clock limit stopped the comparison. The last recorded
decision still had 79.866699 seconds of regulation remaining. The launcher's
coarser progress snapshot had 82.358490 seconds remaining. Neither is a completed
five-minute match result. Goal/countdown overhead consumes wall time without
equivalent regulation-clock progress.

No match completed under this protocol. Only `reference600_blue` ran; neither
finishing650 nor the other side comparison ran. Do not rank the two candidates
from this partial result. Do not describe the earlier RivalSim 8/10 wins as
demonstrated native Rocket League competence, or either candidate as SSL.

The campaign stopped with `RuntimeError: Native case wall-clock timeout`.
The runner closed its match/connections and the bot closed all three gzip
streams. No bot `failure.json`, nonfinite inference failure, or optimizer step
occurred. The native runner, Rival, and Nexto workers are no longer running;
Rocket League and its RLBot server remain open after match teardown.

## Actual observed activity

All aggregates below use the 9,171 recorded policy-decision packets, including
actionable kickoff frames; they are not a regulation-only or ranked statistic.

| Measurement | Result |
| --- | ---: |
| Rival / Nexto distinct touch timestamps observed | 23 / 97 |
| Mean Rival speed | 1,217.316 uu/s |
| 95th percentile speed | 2,015.146 uu/s |
| Speed below 100 uu/s | 1.428% |
| Grounded | 70.210% |
| Car above 300 uu | 2.900% |
| Forward / reverse throttle requested | 88.714% / 2.573% |
| Boost / jump requested | 35.972% / 4.721% |
| Mean distance to ball | 2,782.071 uu |
| Ball in opponent half | 14.742% |

Touch counts are lower bounds from per-player `latest_touch` changes observed
at decisions. Multiple contacts between recorded decisions can be missed.
These are not possession, aerial, dash, or other mechanic classifications.
Rival was moving and using boost, not idle; this activity did not produce goals.

## What has actually been ruled out

- The exact bound reference600 checkpoint and exported actor are unchanged.
- All 9,171 recorded observations reproduce their original eight-channel
  controller outputs through the exact exported recurrent actor: zero action
  differences, finite hidden states and logits, with recorded recurrent resets.
- All 68,553 logged delivered-packet records have consistent decision/held
  outputs. Scheduler reports 9,171 decisions, zero skipped decisions, zero
  duplicates, zero out-of-order packets, three missed physics packets and
  38 resets. Delivered packets include nonactionable phases.
- At all **9,132 uninterrupted nonboundary decision endpoints**, actual native
  `PlayerInfo.last_input` equals all eight recorded `previous_action` values,
  and those equal the preceding command. Zero mismatches in every channel.
  Exclusions: 38 reset boundaries and one interval with a packet gap.
- Seven focused reducer/delivery tests pass, including detection of deliberately
  mismatched inputs, truncated streams, and incorrect previous-action history.

Endpoint delivery verification does not prove every intervening applied physics
tick. Action replay does not prove observation semantics or simulation fidelity.
The input audit still explicitly classifies 39 fields direct, 104 derivable,
29 approximate and 10 unavailable. Individual-wheel and sticky-state proxies
have not suddenly become native measurements.

## Shooting coverage and the next useful investigation

Shooting/finishing already occupies 20% of episode starts in the direct-skills
mixture, alongside 30% natural play, 20% challenges, 15% defense and 15% kickoff.
The last finishing25 block retained real competing opponents and graduated its
finishing reward to actual simulator goals (+10), concedes (-10), and an
unsuccessful timeout (-1). First-touch, projected-shot and approach bonuses are
zero in that finishing role. This is documented and tested in
`results/rival2/direct_skills_finishing_goal_v2/authority.json` and
`tests/test_direct_skills_finishing_goal_v2.py`. It did not establish native
shooting capability; finishing650 has not yet played a native game here.

The current evidence supports a **large simulator-to-native performance gap**,
not a proved root cause. Do not tune rewards or launch another PPO extension
merely to address this observation. Preserve both candidate policies.

Next, use the already captured native packets and issued controls for a bounded,
no-learning discrepancy audit. Prioritize early kickoff/approach segments before
closed-loop divergence: verify physical observations and phase semantics, and
compare short native versus simulator motion/contact traces from sufficiently
specified states. Separately verify the simulator's Nexto integration against
the installed native opponent's actual inputs and identity. Unknown simulator
internal state must remain qualified, not fabricated as exact. Do not presume
which of policy weakness, observation proxies, simulator dynamics or opponent
integration explains the gap until measured. Do not rerun the already completed
export/cadence/loader audits, extend this expired comparison's deadline, or
silently resume training.

## Evidence and reproduction

- `reference600_blue/audit.json`: partial outcome, activity, execution parity.
- `reference600_blue/delivery_audit.json`: actual consumed-input comparison.
- `reference600_blue/decisions.bin.gz`: 967-byte records of observations, quality,
  exact actions, frame/epoch/reset and packet-gap metadata.
- `reference600_blue/ticks.bin.gz`: 42-byte delivered-packet/command records.
- `reference600_blue/decision_packets.bin.gz`: uint32 little-endian size followed
  by exact RLBot `GamePacket.pack()` bytes, one packet per policy decision.
- `ready.json`, `bot_state.json`, `partial_campaign_state.json` under that case.
- `campaign.stderr.log`: bounded timeout and teardown context.
- `report_tests.xml`: seven passing focused tests.

The reducer intentionally refuses to overwrite an existing reduction. Commands
used, with the native RLBot Python environment, were:

```powershell
& G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B benchmarks/report_entity_native_comparison.py reference600_blue
& G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B benchmarks/report_entity_native_comparison.py reference600_blue --delivery-only
& G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B -m pytest -q tests/test_entity_native_report.py --basetemp .tmp/native-report-delivery-20260906 --junitxml results/rival2/entity_native_packet_v1/report_tests.xml
```

No policy, optimizer, reward, physics, curriculum, installed legacy bot or
existing comparison authority was changed during this reduction and audit.
