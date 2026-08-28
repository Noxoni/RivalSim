# Codex Start Prompt — Rival 2.0 Gameplay V3 Production

Work in `Noxoni/RivalSim` on the latest `origin/main`.

This task is a release-gated implementation and validation task. It is **not** a training task.

## 1. Source integrity first

Before editing anything:

1. `git fetch origin` and inspect current `origin/main`.
2. Verify audited baseline commit
   `0228a833e90d2db2715f8b79b65f6cbdc59fefbc`
   is an ancestor of the current HEAD.
3. Preserve all concurrent work. Do not reset, discard, rebase away, or overwrite unrelated changes.
4. Record exact starting HEAD and worktree status.
5. If current source materially changed any file/path described by this handoff after the audited baseline, reconcile the handoff against the newer code before implementation and document the drift. Do not blindly apply stale assumptions.

## 2. Read the entire package before implementation

Read every file under:

`handoff/rival2-gameplay-v3-production-v1/`

in this order:

1. `README.md`
2. `AUDIT_FINDINGS.md`
3. `IMPLEMENTATION_SPEC.md`
4. `TRAINING_SAFETY_GATES.md`
5. `ACCEPTANCE_CRITERIA.md`
6. this file

Then read every mechanics authority/calibration document and machine-readable artifact referenced by `README.md`.

The package is the implementation authority. Do not reduce it to the brief objective below.

## 3. Objective

Implement one immutable reward transition:

`RIVAL2_REWARD_GAMEPLAY_V2` -> `RIVAL2_REWARD_GAMEPLAY_V3`

Gameplay V3 combines:

- retained healthy Gameplay V1 terms;
- unconditional unique-touch reward removed (`0.0` in V3 only);
- approved general mechanics reward (`+0.005` canonical completion, max 10 paid events/player/short episode = `0.05` contract budget);
- targeted `UNNECESSARY_FLIP_THROUGH_CONTACT` penalty (`-0.01` offending player before zero-sum composition);
- physically calibrated exemptions for genuine contest/50, dodge-powered shot/clear, controlled flick, and explicitly same-contact recognized mechanics.

No generic jump penalty. No generic flip penalty. No mechanic failure penalty.

## 4. Architecture requirements

Do **not** turn `MechanicsShadowObserver.attach()` into production training architecture.

Build native GPU-resident V3 state integrated into `Rival2WorldSim` at the authoritative:

`post-physics -> V3 detector/state-machine -> reward accumulation/composition`

boundary.

Allocate V3 detector state only for Gameplay V3 worlds. Do not bloat or alter historical V1/V2 runtime behavior.

Do not copy calibration evidence buffers into production state.

Preserve historical Gameplay V2 strict-double-dash code exactly. Gameplay V3 needs its own successful-dash detector because the old V2 tracker does not enforce the calibrated tangent-speed success requirement.

## 5. Reward-ready scope is frozen for this task

Reward in V3 only:

Continuous calibrated:
- speedflip
- half-flip
- Musty
- Breezi
- Redirect
- Pinch
- Pogo

Source-exact/topology-backed:
- successful dash/wavedash
- Zapdash subtype/sequence via terminal successful dash
- Rival double-dash label over two distinct successful dashes, no third payout
- ball reset acquisition
- distinct chain reset reacquisition
- pre-flip reset subtype, no extra payout
- car reset acquisition

Do not add reward for possession, ground carry, generic flicks, air-dribble milestones, pop reset beyond reset acquisition, double taps, stalls, recovery, generic jump/flip/aerial, or any OBSERVE-FIRST/telemetry-only catalog item.

The controlled-flick classifier built here is **exemption-only** and pays zero reward.

## 6. Historical invariants

Gameplay V1 and V2 must remain immutable.

Before and after implementation prove these exact hashes remain unchanged:

- Gameplay V1: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`
- Gameplay V2: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`
- Obs V1: `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF`
- Action V1: `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B`
- Episode V1: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`

Run existing V2 tests unchanged.

## 7. New exemption calibration

The contest/50, dodge-powered contact, and controlled-flick exemption boundaries are not pre-authorized numerical constants.

Generate the compact deterministic calibration/held-out evidence required by the package.

Freeze thresholds/state topology before held-out evaluation.

Do not loosen or tune against held-out failures merely to pass.

If a legitimate class cannot be protected cleanly, return `BLOCKED` rather than activating a broad penalty.

## 8. Checkpoint transition safety

The known latest accepted checkpoint in committed repository evidence is the `plus_120` mixed-opponent boundary:

- iteration/policy: `479`
- decision samples: `3,655,854,038`
- SHA-256: `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`
- recorded path:
  `G:/dev/RivalSim-runs/opponent-curriculum-v1-safe-20260827-b2af03d/checkpoints/rival2_opponent_curriculum_plus_120_resume.pt`

Before using it, check whether a newer accepted local checkpoint exists. If so, prove its audit/identity before choosing it. If source selection is ambiguous, return `BLOCKED`. Do not silently fall back to Gameplay V1 +239.

Use a fresh Gameplay V3 environment and explicit V2->V3 checkpoint curriculum transition. Do not use ordinary strict checkpoint loading and do not live-switch the reward inside an existing campaign process.

Preserve model/optimizer/RNG/counters/historical pool/mixed-PPO safety/retention/curriculum family+side assignment exactly as specified by the package, while explicitly reinitializing fresh simulator/V3 detector state and external-opponent temporal episode state consistently with the fresh kickoff world.

Never overwrite the source checkpoint.

## 9. No training — hard boundary

Do not call any PPO optimizer/update path under Gameplay V3.

Forbidden in this task:

- `trainer.update(...)`
- `train_iteration(...)`
- `ppo_update(...)`
- mixed curriculum PPO update
- any optimizer `.step()` associated with V3 validation

Frozen inference and rollout collection are allowed only for validation.

Any full-shape rollout smoke must use a **disposable validation trainer/process**. It is acceptable for that disposable object to advance environment/RNG/sample-accounting state while collecting a rollout; do not save or treat that state as campaign continuation. The source checkpoint and its persisted counters must remain byte-identical and authoritative.

The 256-episode shadow gate must likewise be no-learning and disposable.

## 10. Production-scale crash gate

Do not declare READY based only on small tests.

On the intended training GPU, run the exact-scale gates from `TRAINING_SAFETY_GATES.md`:

- construct Gameplay V3 at `131,072` worlds;
- execute one complete four-physics-tick / 30 Hz decision;
- report logical and actual CUDA memory impact;
- run one horizon-32 mixed-opponent rollout collection with the real production world count/configuration and **no PPO update**;
- destroy the disposable validation state cleanly.

Do not silently reduce world count if this fails. Return `BLOCKED` on OOM, illegal access, Warp ABI failure, or unrecoverable allocator failure.

## 11. Shadow reward gate

Run the package's approximately 256 short-episode frozen-policy shadow evaluation from the exact selected source policy.

Inspect raw bounded evidence, not only aggregate counts.

Report reward-scale ratios and classifier frequencies. The gate must prove the anti-flip classifier is neither swallowing legitimate flip contacts through broad exemptions nor penalizing legitimate challenges/power/flicks.

## 12. Evidence / commit discipline

Publish the dedicated Gameplay V3 implementation/evidence package requested in `ACCEPTANCE_CRITERIA.md`.

Artifact manifests must bind to committed content hashes. Do not claim final hashes before content is committed/stable.

Before push:

1. fetch `origin/main` again;
2. reconcile any concurrent advancement without force push;
3. rerun affected focused validation;
4. `git diff --check`;
5. commit and push normally to `main`.

## 13. Stop conditions

Fail closed.

Return `BLOCKED` if any required gate fails, including:

- exemption classifier cannot separate legitimate/non-legitimate cases;
- V3 production detector parity fails;
- historical hashes change;
- V3 double-pays a physical accomplishment;
- budget/reset semantics leak;
- V3 hot path requires host per-world classification;
- exact 131,072-world smoke fails;
- checkpoint source/transition cannot be proven safe;
- reward reconstruction fails;
- a live training process cannot be safely isolated from the implementation work.

Do not work around a safety gate by weakening it.

## 14. Final return

Return the complete reviewer package enumerated in `ACCEPTANCE_CRITERIA.md`, including final commit SHA and one final verdict only:

`GAMEPLAY_V3_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

Stop there.

**Do not begin Gameplay V3 PPO training.**
