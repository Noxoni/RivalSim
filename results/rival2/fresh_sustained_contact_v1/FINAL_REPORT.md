# Final full-scenario contact-reward diagnostic — completed, not promoted

## Outcome

Training completed exactly 150 accepted updates and stopped by the frozen budget.
No extension, restart, reward retuning, added opponent, changed contract or model
promotion occurred. The process reported completion at 2026-09-08T20:17:28.884537+00:00;
worker 50200 and launcher 17276 were absent on the final read-only process check.
The operating-system exit code was not separately captured. There is no failure
or STOP file and stderr is empty. The completion state and exited processes,
not absence of stderr alone, establish the bounded stop.

**Execution/integrity PASS; behavioral result NOT PROMOTED.** Final easy
acquisition is 35/512 (6.835938%),
close to the untrained 31/512. Varied is 71/512
(13.867188%), above the untrained 11/512 but below its
115/512 best. Both regressed substantially from update 140. Scoring against
Nexto during training remained extremely weak. Successful payout accounting
does not mean successful learning.

The full run used 663,552,000 trainable 30 Hz decisions,
1,769,472,000 world physics ticks and
2,654,208,000 learner physics-tick exposure, with
21,210 actual Adam steps. Elapsed time from the sole
fresh entry through bounded completion was 8,838.989848 seconds
(approximately 2.455 hours, including scheduled
evaluation/checkpoint overhead). There was one process entry and no resume.

## Frozen experiment and boundaries

Fresh random construction, seed 2026090810; no trained parent or previous Adam.
Original initialization SHA: `DACF9FC9995CBA6E2C8E6E65DBF0175AB71286DADB4F4CBAC34037398D703A93`.
Authority SHA: `63B4B5D911957322D9358D3BE9EA3E0DFF5E0F17552241D74C0CC2ED185F71BD`.
All 42 frozen source hashes, original test evidence, preflight and initialized
checkpoint still match. No active source was edited during this final audit.

All six reset families remained enabled: approximately 50% acquisition, 15%
natural, 10% challenge, 10% finishing, 7.5% defense and 7.5% kickoff. Native
kickoff starts are stationary, including natural-family kickoff rows.
Half the worlds use current self-play and half active native-v5 Nexto. Both
current sides learn in self-play; only Rival learns against Nexto.

The +1 own-first-contact and +0.25 off-ground contact bonuses were retained,
alongside all original goal, seven potential, sustained-control and inactivity
terms. Off-ground requires pre-contact sphere clearance above radius +2 uu;
low bounces qualify and the car need not be airborne. The air budget is ten
awards per player per episode. No named mechanic or inferred aerial success is
reported. Contact does not end an episode: goals or 45 seconds without either
car touching do. All reward and PPO settings remained frozen.

The reward and scenario mix changed together versus the earlier acquisition-only
experiment. This run is **not a single-factor causal test of contact rewards**.

## Complete deterministic acquisition curve

| Update | Easy own contact within 5 seconds | Varied own contact within 8 seconds |
| --- | ---: | ---: |
| 0 | 31/512 (6.054688%) | 11/512 (2.148438%) |
| 10 | 247/512 (48.242188%) | 48/512 (9.375000%) |
| 20 | 247/512 (48.242188%) | 48/512 (9.375000%) |
| 30 | 218/512 (42.578125%) | 42/512 (8.203125%) |
| 40 | 203/512 (39.648438%) | 77/512 (15.039063%) |
| 50 | 39/512 (7.617188%) | 27/512 (5.273438%) |
| 60 | 25/512 (4.882813%) | 20/512 (3.906250%) |
| 70 | 68/512 (13.281250%) | 52/512 (10.156250%) |
| 80 | 49/512 (9.570313%) | 64/512 (12.500000%) |
| 90 | 76/512 (14.843750%) | 97/512 (18.945313%) |
| 100 | 93/512 (18.164063%) | 93/512 (18.164063%) |
| 110 | 68/512 (13.281250%) | 115/512 (22.460938%) |
| 120 | 71/512 (13.867188%) | 89/512 (17.382813%) |
| 130 | 78/512 (15.234375%) | 88/512 (17.187500%) |
| 140 | 97/512 (18.945313%) | 110/512 (21.484375%) |
| 150 | 35/512 (6.835938%) | 71/512 (13.867188%) |

Each scheduled evaluation used the same 1,024 starts: 512 easy and 512 varied,
deterministic Rival against active Nexto, both sides. Own contact must occur
during the original episode; a goal before own contact is failure and contact
in a replacement episode does not count. This is not first-to-ball, possession,
full-match scoring or a win-rate benchmark. Evaluation was read-only and zero
optimizer steps; no extra GPU evaluation was launched during monitoring.

Final conditional successful-contact lower medians: 1.1916666666666667
seconds easy, 2.033333333333333 seconds varied. Final full
8-second no-own-contact fractions: 92.382813% easy,
86.132813% varied. Easy's 5-second success and
8-second no-contact fractions are not complements. Every raw tick list, its
failures and start receipt is retained and independently recomputed.

## Preserved descriptive and final checkpoints

All 16 scheduled checkpoints remain stored and published; no duplicate weight
files are necessary. Descriptive bests do not authorize promotion or continuation.

- Best easy: update 10, 247/512; update 20 ties and has the same full raw contact
  timing list. Earliest-tie convention chooses 10.
- Best varied: update 110, 115/512.
- Best pooled success count: update 10, 295/1,024, tied at 20. This is descriptive
  only because cohort deadlines differ; it is not a prospective selection gate.
- Final: update 150, 106/1,024 pooled successes.

- best_easy: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000010.pt`
  SHA-256: `56D0467D29788AFB72109CF4B60CA41377C7C6F61CCD6323970002E6ABA126F7`
- best_varied: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000110.pt`
  SHA-256: `805AE7A4ED6297D5F7D764451A2E293B561738EDBB781C266385C4FDF3A8D4C1`
- best_combined: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000010.pt`
  SHA-256: `56D0467D29788AFB72109CF4B60CA41377C7C6F61CCD6323970002E6ABA126F7`
- final: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000150.pt`
  SHA-256: `B51C3BA6CBB455B9F902BEFEE34CDF7765541F2CB083C87940D45E66822A1552`

The final evaluated snapshot records last_probe_update 140 because it was saved
before evaluation. The latest rolling checkpoint records 150 afterward.
They have byte-equal model and optimizer tensors/values. Their serialized file
hashes differ due to checkpoint metadata. The final evaluated snapshot is the
permanent final artifact, not a newly selected model.

Rolling checkpoint: `G:\dev\RivalSim-runs\fresh-sustained-contact-v1\rolling_0.pt`
SHA-256: `CFA12BEF4E825A95D60A844D0ED3405A194DB9191DCF72D2FA017BB327BB97F4`

Checkpoint contains model, fresh optimizer, counters, contract hashes, RNG and
opponent state. The existing resume semantics create fresh physical episodes
and clear recurrent/payout state; no physics-state-exact continuation claim is made.

## Ten-update gameplay and contact-award trend

These are stochastic training scenario aggregates, **not match scores or win
rates**. Goals/concessions/touches/inactivity below are from Nexto worlds only.
Speed is the mean of the all-learner update speed metric. First/air awards cover
all learner samples, including self-play; they are not Nexto-only counts.

| Updates | Rival goals | Nexto goals | Rival touches | Inactivity resets | Mean learner speed uu/s | First awards | Off-ground awards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-10 | 87 | 82,464 | 21,283 | 0 | 743.271739 | 22,233 | 5,995 |
| 11-20 | 91 | 80,192 | 24,290 | 1 | 758.579885 | 25,232 | 6,854 |
| 21-30 | 68 | 72,954 | 23,342 | 0 | 675.850795 | 20,008 | 6,426 |
| 31-40 | 58 | 67,709 | 24,294 | 1 | 701.590908 | 25,495 | 7,940 |
| 41-50 | 44 | 57,436 | 23,766 | 0 | 611.46769 | 26,985 | 7,585 |
| 51-60 | 11 | 34,087 | 14,565 | 186 | 473.183612 | 16,537 | 4,488 |
| 61-70 | 9 | 22,931 | 12,210 | 5,223 | 478.419935 | 20,296 | 5,478 |
| 71-80 | 8 | 21,322 | 11,260 | 4,961 | 475.644099 | 18,295 | 5,335 |
| 81-90 | 12 | 19,108 | 11,201 | 5,346 | 458.399568 | 17,189 | 5,384 |
| 91-100 | 7 | 17,819 | 10,218 | 5,138 | 498.22838 | 18,852 | 6,319 |
| 101-110 | 4 | 15,999 | 9,269 | 5,494 | 487.997556 | 17,698 | 5,979 |
| 111-120 | 6 | 15,332 | 7,844 | 5,855 | 466.319315 | 17,801 | 6,434 |
| 121-130 | 6 | 13,504 | 7,412 | 6,613 | 445.353026 | 17,867 | 7,309 |
| 131-140 | 7 | 12,552 | 7,398 | 7,171 | 423.887829 | 17,556 | 8,841 |
| 141-150 | 4 | 11,796 | 7,825 | 7,512 | 421.14301 | 17,066 | 11,159 |

Final window 141-150: 4 Rival goals versus
11,796 Nexto goals, 7,825 Rival touches and
7,512 inactivity resets over 491,519.999996
Nexto-world seconds. This does not constitute competitive gameplay.

The late increase in global contact awards did not translate into robust
deterministic acquisition or scoring. In updates 141-150, self-play worlds had
36,648
current-policy touches across both sides versus 7,825 Rival
touches against Nexto. This is a descriptive split, not proof of a particular
reward exploit or causal explanation. No mechanism investigation or retuning was
performed under this frozen run.

## Whole-run family/opponent evidence

Focal means the configured focal side; in self-play both sides are the same
evolving policy. These totals retain starting-family identity throughout the
episode, including play after the first contact.

| Starting family/opponent | World seconds | Focal goals | Opponent goals | Focal touches | Inactivity resets |
| --- | ---: | ---: | ---: | ---: | ---: |
| ball_acquisition_nexto | 5,285,001.499948 | 51 | 242,008 | 117,849 | 52,532 |
| ball_acquisition_selfplay | 3,910,385.033302 | 8,371 | 6,408 | 70,580 | 60,707 |
| challenge_nexto | 354,479.066667 | 6 | 60,713 | 30,004 | 377 |
| challenge_selfplay | 797,540.133335 | 1,385 | 1,360 | 14,089 | 12,201 |
| defense_nexto | 190,463.133333 | 1 | 45,765 | 12,534 | 6 |
| defense_selfplay | 228,909.633333 | 729 | 8,596 | 6,056 | 2,492 |
| finishing_nexto | 519,246.233334 | 256 | 60,046 | 46,443 | 469 |
| finishing_selfplay | 677,140.833334 | 4,845 | 800 | 18,848 | 9,689 |
| kickoff_nexto | 339,826.866666 | 8 | 45,639 | 3,671 | 0 |
| kickoff_selfplay | 588,343.566666 | 1,212 | 1,147 | 9,275 | 9,000 |
| natural_nexto | 683,783.200002 | 100 | 91,034 | 5,676 | 117 |
| natural_selfplay | 1,170,480.800007 | 2,513 | 2,499 | 14,216 | 17,634 |

Reset-source percentages are **not** guaranteed to equal time occupancy:
episodes last until goal or no-contact timeout. In the final ten-update window,
acquisition accounts for 81.847%
of Nexto-world time. That is measured occupancy, not a changed reset mixture.
All six families had nonzero measured exposure in every update.

The frozen telemetry does not contain a per-family decomposition of the contact
bonuses or every reward component. Those allocations cannot be reconstructed
exactly from aggregate logs and are not fabricated. Complete available
per-family gameplay plus global reward/contact ledgers are in FINAL_STATISTICS.json
and all per-update raw values in through_000150.json.

## Reward and payout evidence

The following are accumulated learner reward contributions, already incorporating
within-decision discount where applicable. Signed terminal totals are dominated
by Nexto losses; self-play terminal gains/losses largely cancel across learners.
These are not absolute-return ratios or standalone skill scores.

| Component | Whole run | Updates 141-150 |
| --- | ---: | ---: |
| access | -311,537.001241 | -14,074.536914 |
| air_touch | 25,374.150753 | 2,788.941481 |
| alignment | -125,968.773064 | -4,443.837348 |
| boost | -115,157.163779 | -4,097.677125 |
| control | -12,408.135044 | -199.22659 |
| defense | 10,318.758677 | 243.179951 |
| field | 3,555.147145 | 176.443885 |
| first_touch | 299,023.733596 | 17,061.071983 |
| goal_velocity | 9,243.441691 | 263.456703 |
| inactivity | -276,787.077011 | -12,460.800356 |
| sustained_control | 4,552.98247 | 741.578223 |
| terminal_goal | -5,446,256.673929 | -117,885.981522 |
| total | -5,936,046.609131 | -131,887.387626 |

Whole-run first awards: 299,110, discounted payment
299023.7335958481. Off-ground awards: 101,526,
discounted payment 25374.150753200054; eligible events:
101,707. The difference of
181 events was unpaid
under the frozen air budget. Capped agent-decisions: 39,954;
that is not a unique-player or unpaid-event count. Repeated contacts without
another first award: 140,061.

Excluded Nexto first awards: 542,797; excluded Nexto
off-ground awards: 902,343. These never enter the learner
bonus total. Final-window exact totals and all 15 window ledgers are preserved.

For all 150 updates independently verified:
- component first_touch == learner_first_reward, tolerance 1e-5;
- component air_touch == learner_air_reward, tolerance 1e-5;
- count * nominal_payment * physics_gamma^3 <= discounted payment
  <= count * nominal_payment, tolerance 1e-5.

The total reward sums can differ from the sum of individually accumulated
components by small floating-point accumulation error; no unrecorded reward
term is inferred from such numerical residuals.

## Integrity and PPO safety

All 17 final CPU audit checks PASS. All 16 probe/checkpoint hashes and receipts,
baseline scenario/spec identities, raw counts, lower medians and no-contact
fractions independently verified. The complete closed prefix exactly matches
the now-stopped external curve and has offsets 1 through 150 without gaps.
Every update has contact-accounting checks and every family has exposure.

Max completed-update mean KL: 0.006568310661950145.
Max recorded completed-update sample KL: 1.3069863319396973.
KL rejections: 0. These are telemetry, not safety thresholds.
Final entropy: 3.5187082290649414. Finite model/Adam checks pass;
all final Adam step counters agree at 21,210.
No numerical, corruption or reward-accounting failure was recorded.

Preflight remains the original 25 focused tests and exact 32,768-world
zero-campaign-step audit. Finalization changes evidence only, not training code.
It does not add tests of gameplay skill or reinterpret acquisition as skill.

## Reproduction and follow-up

Existing command: project .venv Python
`benchmarks/run_fresh_sustained_contact_v1.py report --offset 150`.
It is CPU-only; rerunning refreshes the audit timestamp, not the checkpoint.

FINAL_STATISTICS.json sums numeric per-update family, reward and bonus fields
over contiguous windows [1,10], [11,20], ..., [141,150] and the whole run. The
reported speed/entropy/touches-per-minute are arithmetic means of their update
metrics. Per-probe first-contact success uses positive ticks <=600 or <=960 on
alternating easy/varied rows; conditional medians use the lower middle tick.
Source identities use LF-normalized text hashes; checkpoint hashes use raw bytes.
All precision needed to reproduce the displayed rounded tables is retained.

Recommendation: keep this run stopped and the final model unpromoted. Investigate
why rewarded stochastic contacts do not persist as deterministic acquisition,
and compare the preserved update 10/110/final policy behavior before spending
more compute. This is a recommended next diagnostic, **not a new campaign or
authorization to retune rewards**. No claim of beating Nexto, aerial offense,
possession competence or SSL is supported.
