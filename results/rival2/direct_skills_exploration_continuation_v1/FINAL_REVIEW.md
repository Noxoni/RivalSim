# Exploration continuation complete at750; competitive capability not established

The frozen70-update continuation finished normally. Worker30656 exited after
the final evaluation at2026-09-06T17:29:47.141758Z. No update beyond750 was
accepted. The most recent durable rolling checkpoint and permanent750 snapshot
are byte-identical. This is a completed experiment, not completion of the
user's SSL-development goal and not a deployment/promotion decision.

## Full deterministic Nexto matches

Each boundary used the same ten complete300-second development matches, five
standard layouts across both sides, native-v5 Nexto, raw-argmax Rival at30Hz,
120Hz physics, and unassisted standing kickoffs. No match was rerun or excluded.

| Metric | Original650 | Parent680 | 700 | 725 | Final750 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wins / matches | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| Scored | 7 | 2 | 1 | 6 | 2 |
| Conceded | 196 | 228 | 222 | 209 | 194 |
| Goal difference | -189 | -226 | -221 | -203 | -192 |
| Rival contacts | 456 | 444 | 515 | 508 | 717 |
| Contacts/minute | 9.12 | 8.88 | 10.30 | 10.16 | 14.34 |
| Kickoff first contacts | 0 | 0 | 0 | 0 | 0 |

Versus the immediate680 parent, final goal difference improves in9/10 paired
worlds and worsens in1/10. Versus original650 it improves in6/10 and worsens
in4/10, yet aggregate goal difference is still three goals worse. Versus725,
six improve, one is unchanged and three worsen. Final750 concedes15 fewer but
scores four fewer than725. The late recovery is not a monotonic scoring gain.

The final contact rate is57.2% above650. Next-contact identity is Rival for
347/708 resolved followups (49.0113%), versus123/444 (27.7027%) at650 and125/498
(25.1004%) at725. This is a real change in the recorded contact sequence, not
proof of deliberate control or continuous possession: repeated nearby
collisions can also produce same-player contacts. It does not establish
dribbling, successful challenges, mechanics, or good finishing.

Final contact detail: Rival717, Nexto1342; forward ball-velocity change481/717;
forward ball displacement to the next contact/goal340/712 resolved records.
All ten matches had Rival contact. Zero no-touch truncations is fixed protocol,
not an acquired skill. The standing-kickoff failure remains unresolved.

These are descriptive fixed-development results, not a new untouched test or
statistical proof of general improvement/native Rocket League parity. The
policy remains very far from beating Nexto. No SSL claim is supported.

## Are the direct tasks learned?

TASK_PROGRESS_750.md and task_progress_750.json retain all100 exploration
updates, ten nonoverlapping windows, raw role/opponent counts and denominators.
They reduce saved records only; no extra training or evaluation occurred.

- Nexto shooting goals/observed endings:10.7901% in651-660,8.0642% in691-700,
  9.5789% in741-750. Some late recovery, still below the first window.
- Self-play shooting:32.0542% to15.0383% over the first/last windows. Self-play
  opposition and visitation evolve; this is not a controlled ability estimate.
- Natural Nexto touches/minute:7.6445 to6.5779; natural goals/minute:0.1219 to
  0.1147. Both recover somewhat from the early post-intervention low.
- Challenge control-gain events/minute:0.9225 to0.8479; defensive clear
  events/minute:3.4243 to3.1309. These are existing task predicates, not
  independent possession or save adjudication.
- Training kickoff race-first-contact events/minute:0.6745 to1.1549, but
  control-gain events/minute:0.2736 to0.1335. Momentum-assisted training starts
  and standing evaluation starts must not be conflated.

Thus neither complete task mastery nor merely a transfer-only problem has been
demonstrated. Episode-window censoring, fresh physical entry at681, changing
on-policy states and evolving self-play limit causal interpretation.

## Integrity and resumability

- Continuation:70 accepted updates,10,052 Adam steps,309,657,600 learner
  decisions,825,753,600 physical world ticks.
- Including the preceding30-update exploration arm:100 updates and442,368,000
  learner decisions. This exposure did not establish competitive superiority.
- Final cumulative counters:750 updates,158,982 Adam steps,3,317,760,000 learner
  decisions and8,847,360,000 physical world ticks.
- All nine terminal checks pass. All four preserved audit boundaries pass28
  core and six exploration-objective checks each. The three match boundaries
  pass all11 match-integrity checks each.
- Parent/model/Adam lineage, reward, PPO, corpus, architecture, controller and
  exploration authority remain unchanged. Model/Adam are finite. RNG and
  opponent state are preserved with the documented fresh-physical-episode
  resume semantics; this is not exact physical-world replay.
- Maximum completed-update mean KL in this continuation:0.005023520667472223;
  maximum sample KL:3.6825273036956787. KL remains telemetry only; zero KL
  rejections. No numerical/runtime failure is recorded; stderr is empty.
- CPU closeout revalidated the source package, final rolling/permanent hash,
  counter/curve prefix, all evaluations, closed process and stable logs. It
  performed zero optimizer steps and zero new matches. Imported Warp device
  discovery is not a simulation rollout.
- Task reduction rebuilt byte-identically twice, SHA
  1994E321ADFF8BF3186D391E646F80702370E696070A4A812604706F621BCBF2.
  Its20 focused tests and the earlier35 closeout/adjacent tests are recorded.

Final checkpoint:
checkpoints/rival2/direct_skills_exploration_continuation_v1/child_000070.pt

SHA-256:
517A2217CBDF4124612B518E10F30BC4D5F00F5C1DC77A5119D10D2B1DD89613

Rolling counterpart:
G:/dev/RivalSim-runs/direct-skills-exploration-continuation-v1/rolling_0.pt

## Next decision under the broad goal

Do not extend this arm automatically, discard its evidence, restart random
weights, or declare the exploration intervention a complete success/failure
from one metric. It improved late contact sequencing and conceding relative
to680, while scoring and standing kickoffs remain poor. The constant barrier
is a plausible influence, not a proven cause of the regression.

The next useful learning experiment is a bounded matched comparison from the
same preserved entry checkpoint: retain the current exploration term in one
arm and withdraw only the added uniform barrier in the other. Freeze the
parent, budget, evaluation points and comparison before either arm learns;
preserve model/Adam/RNG, temperature, ordinary entropy, rewards, curriculum and
architecture. Consider750 as the common parent so its observed repeat-contact
change is retained rather than assuming680 is still the right intervention
point. This is a next-experiment recommendation, not a frozen authority or a
launched campaign. Do not substitute a gradient-norm probe for that comparison.

The completed-run heartbeat monitor is paused; its existing prompt, schedule,
and target were preserved through the supported automation-update tool after
following OpenAI Docs. The broad SSL goal remains active. Any next campaign
requires a separate prospective authority and updated monitor context.
[Official scheduling documentation](https://learn.chatgpt.com/docs/automations?surface=app).
