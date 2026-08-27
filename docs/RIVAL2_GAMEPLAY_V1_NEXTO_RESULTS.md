# Rival 2.0 Gameplay V1 vs pinned Nexto

Source HEAD: `bf03aaad90e6d44a04adfd8d7d4d74f42ede974e`.

This is evaluation only: standard Soccar kickoff, first goal ends the episode, 15 seconds without a touch truncates, and 45 seconds is the hard limit. Rival is deterministic at 30 Hz; frozen Nexto is deterministic at 15 Hz with its stock 120 Hz kickoff controller. Reward is not used to select an outcome.

## Identities

- Nexto upstream commit: `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`.
- Nexto model SHA-256: `BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`.
- `plus_180`: iteration `300`, SHA-256 `FEC1C289E7F7EB8D69876FB75C5325D56063A7A674A46F6FD20C5C270542511B`.
- `plus_239`: iteration `359`, SHA-256 `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`.

## Episode design

Each checkpoint uses 1024 primary episodes: 512 with Rival Blue and 512 with Rival Orange. Within each side, kickoff layouts 0-4 receive 103, 103, 102, 102, and 102 episodes. Checkpoints use the identical prospective simulator seed `2026082701`, world index, physical side, and layout.

Because both deployed policies and standard kickoff states are deterministic, repeated rows for a given side/layout are paired episode outcomes rather than claims of 1,024 statistically independent randomized physical starts.

## Primary deterministic result

| Checkpoint / Rival side | Episodes | Rival wins | Nexto wins | No goal | Decisive win rate | GF-GA | GD | Goal-term | No-touch | Hard-time | Rival touches | Nexto touches | Touch diff | First-touch share | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plus_180 / overall | 1024 | 0 | 1024 | 0 | 0.000% | 0-1024 | -1024 | 100.000% | 0.000% | 0.000% | 2816 | 9315 | -6499 | 100.000% | 18.893 |
| plus_180 / Rival Blue | 512 | 0 | 512 | 0 | 0.000% | 0-512 | -512 | 100.000% | 0.000% | 0.000% | 1588 | 3280 | -1692 | 100.000% | 17.279 |
| plus_180 / Rival Orange | 512 | 0 | 512 | 0 | 0.000% | 0-512 | -512 | 100.000% | 0.000% | 0.000% | 1228 | 6035 | -4807 | 100.000% | 20.507 |
| plus_239 / overall | 1024 | 0 | 1024 | 0 | 0.000% | 0-1024 | -1024 | 100.000% | 0.000% | 0.000% | 2250 | 6958 | -4708 | 100.000% | 14.455 |
| plus_239 / Rival Blue | 512 | 0 | 512 | 0 | 0.000% | 0-512 | -512 | 100.000% | 0.000% | 0.000% | 1125 | 3684 | -2559 | 100.000% | 13.614 |
| plus_239 / Rival Orange | 512 | 0 | 512 | 0 | 0.000% | 0-512 | -512 | 100.000% | 0.000% | 0.000% | 1125 | 3274 | -2149 | 100.000% | 15.296 |

## Secondary stochastic Rival result

This optional suite samples Rival's learned hybrid action distribution while Nexto remains deterministic. It uses 256 episodes per Rival side and is secondary to the deterministic deployment-policy result above.

| Checkpoint / Rival side | Episodes | Rival wins | Nexto wins | No goal | Decisive win rate | GF-GA | GD | Goal-term | No-touch | Hard-time | Rival touches | Nexto touches | Touch diff | First-touch share | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plus_180 / overall | 512 | 37 | 474 | 1 | 7.241% | 37-474 | -437 | 99.805% | 0.000% | 0.195% | 1369 | 4370 | -3001 | 90.234% | 13.856 |
| plus_180 / Rival Blue | 256 | 22 | 233 | 1 | 8.627% | 22-233 | -211 | 99.609% | 0.000% | 0.391% | 670 | 2074 | -1404 | 89.453% | 13.572 |
| plus_180 / Rival Orange | 256 | 15 | 241 | 0 | 5.859% | 15-241 | -226 | 100.000% | 0.000% | 0.000% | 699 | 2296 | -1597 | 91.016% | 14.139 |
| plus_239 / overall | 512 | 45 | 464 | 3 | 8.841% | 45-464 | -419 | 99.414% | 0.000% | 0.586% | 1435 | 4433 | -2998 | 100.000% | 13.732 |
| plus_239 / Rival Blue | 256 | 23 | 232 | 1 | 9.020% | 23-232 | -209 | 99.609% | 0.000% | 0.391% | 690 | 1972 | -1282 | 100.000% | 13.251 |
| plus_239 / Rival Orange | 256 | 22 | 232 | 2 | 8.661% | 22-232 | -210 | 99.219% | 0.000% | 0.781% | 745 | 2461 | -1716 | 100.000% | 14.212 |

## Rival movement and controls

| Checkpoint / side | Mean speed | Supersonic | Boost active | Pickups | Grounded | Airborne | Jump active | Flips/min | Analog saturation | Saves |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plus_180 / overall | 1155.632 | 4.999% | 12.716% | 3380 | 47.997% | 52.003% | 34.088% | 27.965 | 0.778% | 0 |
| plus_180 / Blue | 1176.198 | 5.840% | 12.601% | 1640 | 44.226% | 55.774% | 34.271% | 29.889 | 0.743% | 0 |
| plus_180 / Orange | 1138.302 | 4.290% | 12.812% | 1740 | 51.175% | 48.825% | 33.934% | 26.344 | 0.807% | 0 |
| plus_239 / overall | 1243.190 | 6.571% | 13.144% | 2456 | 55.346% | 44.654% | 40.297% | 22.213 | 3.693% | 408 |
| plus_239 / Blue | 1246.370 | 7.281% | 12.178% | 1023 | 52.368% | 47.632% | 44.171% | 22.905 | 4.452% | 204 |
| plus_239 / Orange | 1240.359 | 5.938% | 14.003% | 1433 | 57.997% | 42.003% | 36.849% | 21.597 | 3.016% | 204 |

## Dash-mechanics telemetry

Named mechanics have no authoritative simulator event flag. The evaluator therefore retains actual flip onset, all four wheel contacts, first-jump and landing timing, suspension length/velocity, orientation, contact normal, controller input, and speed before/after landing. Labels are prospective candidates rather than intent claims.

A wavedash candidate is an actual airborne dodge from zero wheel contact that reaches first wheel contact within 0.20 seconds with no more than 0.35 seconds of pre-flip air time. A speed-increasing candidate must also retain strictly higher velocity tangent to the contacted surface after landing. Zapdash candidates additionally require a front-wheel-only first landing, a non-flat three-wheel first-jump onset, and the subsequent landing dodge; double-dash candidates require two wavedash candidates with intervening contact. Raw evidence for every named candidate is published beside each checkpoint result.

Sources used to define and qualify the telemetry: [Psyonix GDC vehicle-physics presentation](https://media.gdcvault.com/gdc2018/presentations/Cone_Jared_It_Is_Rocket.pdf), [RLBot useful game values](https://wiki.rlbot.org/v5/botmaking/useful-game-values/), [Rocket Science dodge measurements](https://www.s543778567.website-start.de/know/videos/dodges), [Rocket Science landing-wavedash analysis](https://www.youtube.com/watch?v=baNsqFEfRMY), and the [community mechanics database](https://0byte-coding.github.io/rocket_league_mechanics/) for zapdash/double-dash naming.

| Mode | Checkpoint | Policy | Wavedash candidates | Speed-increasing | Zapdash | Double-dash event rows | Wall dash | Curved-surface dash |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deterministic | plus_180 | Rival | 103 | 103 | 0 | 0 | 0 | 0 |
| deterministic | plus_180 | Nexto | 0 | 0 | 0 | 0 | 0 | 0 |
| deterministic | plus_239 | Rival | 306 | 306 | 0 | 0 | 204 | 0 |
| deterministic | plus_239 | Nexto | 204 | 204 | 0 | 0 | 204 | 0 |
| stochastic | plus_180 | Rival | 103 | 66 | 1 | 12 | 24 | 9 |
| stochastic | plus_180 | Nexto | 46 | 25 | 0 | 0 | 17 | 22 |
| stochastic | plus_239 | Rival | 82 | 63 | 0 | 12 | 19 | 17 |
| stochastic | plus_239 | Nexto | 100 | 77 | 0 | 0 | 60 | 34 |

### Representative measured zapdash sequence

The sole strict zapdash candidate occurred for Rival `plus_180` in the secondary_stochastic suite (world 279, layout 3). Front-only contact began at tick 1406 on front_left; 11 ticks later, the first jump began with front_left, front_right, back_left in contact; the directional `right` flip began 8 ticks later and landed after 1 tick on front_right. Contact-surface-tangent speed changed by +361.061 uu/s. It is also one half of a measured double-dash candidate. This is strong transition evidence, but not a claim about learned intent.

## Checkpoint selection

**Recommendation: continue from `plus_239`.**

`plus_239` recorded 0 Rival wins and 1024 Nexto wins, versus 0 and 1024 for `plus_180`. Its goal differential was -1024 versus -1024; no-touch fraction was 0.000000 versus 0.000000. Side-consistency gaps were 0.000000 and 0.000000. With the primary outcome tied, `plus_239` supplied 408 measured saves versus 0, 9.120 versus 8.733 Rival touches/min, and allowed 28.204 versus 28.889 Nexto touches/min. Its shorter mean survival (14.455 versus 18.893 seconds) is an honest negative. The secondary sampled-policy suite also favored `plus_239`: 45 wins and 8.841% decisive win rate, versus 37 and 7.241% for `plus_180`; this remains corroborating rather than primary evidence.

This is a tiebreaking recommendation: neither checkpoint showed competitive deterministic performance against Nexto.

The selection is based first on the fixed Nexto opponent, then side consistency, goal differential/defensive outcome, acquisition retention, and finally controller/mechanical sanity. Self-play goals/min and reward totals were not used.

## Evidence boundary

No Rival or Nexto training ran. No policy, reward, PPO, observation, action, physics, or adapter behavior changed. The stochastic suite is explicitly secondary and did not influence the primary deterministic measurement.
