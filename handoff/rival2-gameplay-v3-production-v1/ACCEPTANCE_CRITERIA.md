# Gameplay V3 Acceptance Criteria

Gameplay V3 is accepted for **review** only when every required item below is backed by committed machine-readable evidence.

## A. Contract / source compatibility

- [ ] `RIVAL2_REWARD_GAMEPLAY_V3` exists as a new immutable identity.
- [ ] canonical V3 SHA-256 is recorded.
- [ ] Gameplay V1 hash remains `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`.
- [ ] Gameplay V2 hash remains `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`.
- [ ] observation/action/episode hashes remain unchanged.
- [ ] V3 unconditional touch reward is exactly `0.0`.
- [ ] no generic jump or flip penalty exists.
- [ ] Gameplay V2 standalone double-dash reward is absent from V3 composition.

## B. V3 reward reconstruction

Tests must prove exact/within-documented-float-tolerance reconstruction of:

- [ ] goal;
- [ ] progress;
- [ ] touch=`0`;
- [ ] demo;
- [ ] speed;
- [ ] supersonic;
- [ ] boost use;
- [ ] boost pickups;
- [ ] save;
- [ ] mechanics;
- [ ] unnecessary flip-through penalty;
- [ ] total;
- [ ] `Orange == -Blue`.

Historical V1/V2 reward tests must still pass unchanged.

## C. Mechanics production parity

For the committed calibration trace set or a bounded deterministic parity subset:

- [ ] speedflip completion identity/tick matches final calibrated detector;
- [ ] half-flip matches;
- [ ] Musty matches targeted correction;
- [ ] Breezi matches targeted correction;
- [ ] redirect matches targeted correction;
- [ ] pinch matches;
- [ ] pogo matches.

Source-exact tests:

- [ ] successful dash requires calibrated tangent-speed gain > `1.0 uu/s`;
- [ ] dash air/landing windows remain `42/24` ticks;
- [ ] zap windows remain `12/30` ticks;
- [ ] Rival double-dash window remains `90` ticks;
- [ ] no fresh-jump prohibition added;
- [ ] double-dash label adds no third reward;
- [ ] ball reset requires >=3 ball-supporting wheels plus real untimed resource acquisition;
- [ ] car reset uses other-car support identity;
- [ ] chain reset pays only on a distinct later reacquisition;
- [ ] preflip reset subtype does not stack.

## D. Mechanics budget/de-duplication

- [ ] canonical mechanics payout exactly `0.005` each;
- [ ] max paid events/player/episode exactly 10;
- [ ] contract budget exactly `0.05`;
- [ ] event 11+ is suppressed rather than paid;
- [ ] budgets independent by player;
- [ ] budget resets only on true episode reset;
- [ ] same accomplishment cannot stack subtype labels;
- [ ] Breezi terminal event does not also pay Musty;
- [ ] a reset followed by a later distinct Musty may pay both;
- [ ] dash1 + dash2 may both pay, double-dash label pays zero additional.

## E. Explicitly disabled mechanics

Machine-readable V3 contract/evidence must show no reward for:

- [ ] possession;
- [ ] ground carry/dribble;
- [ ] generic controlled-flick exemption event;
- [ ] air-dribble milestones;
- [ ] pop reset beyond reset acquisition;
- [ ] double tap/rebound;
- [ ] bare stall;
- [ ] recovery;
- [ ] generic jump/flip/aerial.

## F. Bad flip candidate tests

Positive penalty:

- [ ] uncontested loose ball + active directional flip + new contact => exactly one `UNNECESSARY_FLIP_THROUGH_CONTACT`.

No candidate:

- [ ] drive-through touch without active flip;
- [ ] jump/aerial touch without directional dodge;
- [ ] `has_flipped` stale state with `is_flipping == 0`;
- [ ] active directional flip without ball contact;
- [ ] flip occurs near contact but not on the same physical contact tick/eligible bounded sequence.

Exactly-once:

- [ ] persistent contact chatter cannot retrigger;
- [ ] genuine separation/recontact can create a new candidate;
- [ ] pending candidate crossing a 30 Hz boundary resolves once;
- [ ] reset cancels pending candidate completely.

## G. Contest/50 exemption calibration

Evidence package must include:

- [ ] calibration corpus IDs and labels;
- [ ] source commit/physics hashes;
- [ ] derived contest window/continuous boundaries;
- [ ] prospective held-out split frozen before evaluation;
- [ ] held-out FP=0 and FN=0 for intended calibrated cases, or explicit BLOCKED result;
- [ ] simultaneous 50 accepted;
- [ ] narrowly adjacent opponent contact accepted;
- [ ] genuine convergence accepted;
- [ ] distant opponent rejected;
- [ ] nearby non-converging opponent rejected;
- [ ] opponent moving away rejected;
- [ ] uncontested loose-ball flip rejected as exemption.

No large hand-chosen proximity exemption is allowed.

## H. Power-contact exemption calibration

Evidence package must include:

- [ ] actual contact-point velocity including `omega x r`;
- [ ] rotational/dodge vs translational contribution;
- [ ] ball delta-v / impulse;
- [ ] offensive power-shot positives;
- [ ] defensive power-clear positives;
- [ ] weak-but-real dodge-powered positive;
- [ ] weak ordinary flip-touch negative;
- [ ] translation-dominated high-speed negative;
- [ ] already-fast ball / insignificant dodge negative;
- [ ] prospective held-out FP=0/FN=0 on intended cases, or BLOCKED.

## I. Controlled-flick exemption calibration

This is exemption-only.

Evidence package must prove:

- [ ] controlled pre-dodge relation exists in positive cases;
- [ ] actual directional dodge occurs;
- [ ] dodge contact produces release;
- [ ] ball exits prior controlled relation;
- [ ] front/diagonal/side or varied real controlled releases represented;
- [ ] loose-ball flip-through negative represented;
- [ ] kickoff/50 negative represented;
- [ ] brief near-car/no-control negative represented;
- [ ] chase contact negative represented;
- [ ] prospective held-out FP=0/FN=0 on intended cases, or BLOCKED;
- [ ] this detector contributes exactly zero positive mechanics reward.

## J. Recognized-mechanic exemption

- [ ] Musty same terminal contact suppresses bad-flip penalty;
- [ ] Breezi same terminal contact suppresses bad-flip penalty;
- [ ] authorized preflip/reset same-contact sequence suppresses when applicable;
- [ ] unrelated nearby mechanic does not suppress;
- [ ] redirect/pinch/pogo/dash/speedflip/half-flip are not blanket exemption flags;
- [ ] same physical contact never gets both mechanics reward and bad-flip penalty when its recognized mechanic requires the contact.

## K. Reset / lifetime / ABI tests

- [ ] every V3 Warp kernel launch compiles on CUDA;
- [ ] every launch argument count/order/dtype matches kernel signature;
- [ ] V1/V2 launch path unchanged;
- [ ] V3 post-physics detector placement proven;
- [ ] interval counters reset correctly;
- [ ] persistent state survives decision boundary;
- [ ] persistent state clears on episode reset;
- [ ] history initialized after reset;
- [ ] V3 bridge views alias Warp arrays;
- [ ] V1/V2 bridge still constructs without V3 state;
- [ ] observations remain `(N,2,182)`.

## L. Hot-path / GPU memory tests

- [ ] no per-world CPU classifier;
- [ ] no `.numpy()`/`.cpu()` in production per-tick path;
- [ ] no full-world host transfer during normal V3 stepping;
- [ ] calibration evidence buffers absent from production V3 state;
- [ ] added logical V3 bytes reported at 131,072 worlds;
- [ ] exact 131,072-world V3 environment allocation succeeds;
- [ ] one 4-tick V3 decision succeeds at full scale;
- [ ] one horizon-32 mixed-opponent rollout collection succeeds at full scale with **no update**;
- [ ] no OOM, CUDA illegal access, Warp launch failure, or unrecoverable allocator warning.

## M. Checkpoint transition tests

Known candidate source published in repository evidence:

`plus_120 / iter 479 / policy 479 / SHA 3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`

Required:

- [ ] intended source checkpoint selected explicitly;
- [ ] exact file SHA verified;
- [ ] ordinary strict V2 checkpoint load into V3 fails as expected;
- [ ] explicit V2->V3 transition succeeds;
- [ ] model exact after transition;
- [ ] optimizer exact;
- [ ] counters exact;
- [ ] RNG/generator states preserved according to transition record;
- [ ] historical pool exact;
- [ ] mixed optimizer/adaptive PPO/retention state exact;
- [ ] opponent family/side assignments exact;
- [ ] simulator/V3 detector state fresh;
- [ ] Nexto/Wisp temporal state reset to fresh episode consistent with restored assignments;
- [ ] source checkpoint remains byte-identical after test.

## N. 256-episode no-learning shadow gate

From the exact selected source policy:

- [ ] policy frozen;
- [ ] opponent learning disabled;
- [ ] no PPO update;
- [ ] full V3 reward reconstructed but not used to learn;
- [ ] event rates reported;
- [ ] exemption rates reported;
- [ ] mechanics budget-hit fraction reported;
- [ ] reward scale ratios reported;
- [ ] bounded raw event evidence inspected;
- [ ] impossible/jitter count reported;
- [ ] no hidden reward term;
- [ ] policy/model/optimizer identity unchanged after gate.

## O. Focused test command set

Codex should create a focused V3 test module rather than modifying broad acceptance scope. At minimum run:

```text
pytest -q tests/test_rival2_gameplay_v2.py
pytest -q tests/test_rival2_mechanics_calibration.py
pytest -q <new Gameplay V3 focused tests>
```

Also run adjacent reward-transition/checkpoint tests if existing tests cover those paths.

Run Ruff/static checks only on changed Python files.

Do not broaden to the full simulator acceptance suite unless a specific failure makes it necessary.

## P. Required committed evidence

Publish under a dedicated V3 result directory, suggested:

`results/rival2/gameplay_v3_validation/`

Required machine-readable artifacts:

- `contract.json` or contract manifest;
- `classifier_calibration.json`;
- `detector_parity.json`;
- `deterministic_cases.json`;
- `kernel_abi_smoke.json`;
- `memory_smoke.json`;
- `checkpoint_transition.json`;
- `reward_reconstruction.json`;
- `shadow_gate_summary.json`;
- `shadow_event_evidence.json`;
- `artifact_manifest.json` with hashes bound to committed blobs/content.

Write a compact reviewer document under `docs/`.

## Q. Required return package from Codex

Return all of:

1. final commit SHA;
2. starting/ending HEAD;
3. files changed;
4. V3 contract hash;
5. exact reward arithmetic;
6. V1/V2 hash proof;
7. production mechanics list;
8. disabled/telemetry-only list;
9. V3 dash/reset semantics;
10. bad-flip candidate definition;
11. contest exemption thresholds/window and confusion counts;
12. power exemption thresholds and confusion counts;
13. controlled-flick exemption thresholds and confusion counts;
14. recognized-mechanic allowlist/association rule;
15. deterministic test results;
16. kernel ABI/reset tests;
17. V3 state memory footprint;
18. 131,072-world one-decision smoke result;
19. 131,072-world horizon-32 rollout-only smoke result;
20. source checkpoint selected and SHA;
21. checkpoint transition preservation proof;
22. 256-episode no-learning shadow results;
23. mechanics/bad-flip reward scale;
24. confirmation no PPO update/training occurred;
25. confirmation no historical checkpoint changed;
26. final verdict `GAMEPLAY_V3_READY_FOR_REVIEW` or `BLOCKED`.
