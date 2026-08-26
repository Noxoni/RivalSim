# Rival 2.0 RocketSim reciprocal cross-validation

Verdict: **PASS_GREEN**.

## Adapter fidelity

The 2,048-state gate passed with `3.5762786865234375e-07` maximum observation error and 100% deterministic action agreement for both Blue and Orange. Exact team/mirror action agreement was also 100%.

## Normal five-minute RocketSim matches

The required ten deterministic layout/side matches and the user-authorized 128 stochastic matches completed (64 Rival Blue, 64 Rival Orange). The published throughput probe still records the original 4,096-match projection; the controlling sample-size authority is now exactly 128.

- Rival Blue: 0-64, 0.000% wins, goals 3-2608.
- Rival Orange: 0-64, 0.000% wins, goals 5-2678.

The ten exact deterministic scorelines are in `canonical_normal_matches.json`; Blue and Orange are never hidden in an aggregate rate.

Cross-simulator normal-match classification: **DISAGREEMENT**.

## Kickoff-free RocketSim open play

All 4,096 continuous states and 16,384 four-way duels completed. Rival Blue: 746-7434 with 12 draws and 9.120% decisive wins. Rival Orange: 753-7419 with 20 draws and 9.214% decisive wins.

## Rival movement mechanics

Read-only telemetry was collected from controller and RocketSim state transitions. It includes jump rising edges/holds, actual first/double-jump and flip onsets, direction, wheel/ground context, wheel-contact-to-jump-to-flip timing, landing timing, air time, speed deltas, unavailable jump presses, and conservative wavedash/zapdash/double-dash candidates. Candidate labels are not inferred from button frequency; every classified event retains its raw timing/state evidence.

Normal stochastic jump-active decisions: Blue `0.587812`, Orange `0.581387`; prior RivalSim comparison `0.77516`.
Open-play jump-active decisions: Blue `0.768373`, Orange `0.770505`.

## Comprehensive behavioral dataset

The same single-pass runs retain per-match accumulators and authoritative touch, possession-chain, pad, demo, native shot/save, goal, and classified-mechanic events for both policies. Aggregates cover controller magnitudes, boost economy, speed/travel, ground/air/demo occupancy, distances, field occupancy, touch regions and ball response, goal timing/entry, kickoff versus established play, and all requested side/suite dimensions.

Ball wall/backboard/ceiling continuation and challenge/50 outcomes are explicitly omitted: this public RocketSim binding exposes neither authoritative ball-surface contacts nor an objective challenge event. No positional or hindsight-tuned proxy was substituted.

## Side-asymmetry localization

RocketSim open play substantially reduces the RivalSim side split while adapter symmetry is exact; RivalSim simulator/lifecycle asymmetry becomes more likely, without proving causality.

No policy, reward, PPO, observation/action contract, controller semantics, or simulator physics was changed. No training was run.

## Evidence

Machine-readable evidence is under `results/rival2/rocketsim_crosscheck/`, including adapter fidelity, provenance, throughput, all match scorelines, side-separated summaries, cross-simulator deltas, the state bank, all duel outcomes, paired families, mechanic summaries/classified-event evidence, and artifact hashes. The complete open-play event stream is retained losslessly as `open_play_behavior_raw.jsonl.gz`; its manifest records both the compressed artifact hash and the original 418,790,602-byte JSONL hash so the published evidence can be checked after decompression.
