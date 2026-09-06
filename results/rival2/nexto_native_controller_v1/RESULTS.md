# Controller correction validated; no training or capability promotion

Prospective implementation/protocol: `bbafdd99fcac865b879822e3c11e9ba459537a1b`.
All 12 files were retrieved as GitHub blobs and their complete Git object hashes
verified before GPU execution. The worker exited normally. No training is live.

The new opt-in adapter and collector passed bounded CPU and real CUDA validation.
Neither Rival nor Nexto weights were modified. No optimizer was constructed;
there were zero optimizer steps, zero backward calls and no policy gradients.
Rewards, scenarios, observations and physics remain unchanged. Legacy runners
and their old evidence remain untouched and are not silently upgraded.

## What was verified

- Actual installed native controller methods provide the independent oracle:
  512 packet steps x eight worlds, uneven activation/suspension, both sides,
  kickoff admission/denial, full script, phase exit and saved-state restoration.
  CPU and CUDA match emitted controls, pending actions, prediction timing,
  previous-action inputs, neural clock and kickoff index exactly.
- Eleven focused CPU tests passed, including sampling modes, nonfinite logits,
  corrupt-state rejection and the collector's explicit fresh-episode resume.
  Initial fixture failures and their correction are preserved, not hidden.
- Both real pinned CUDA model modes passed serialized mid-cadence restoration:
  17 subsequent calls reproduce controls, sampled action indices, all controller
  tensors, inference counters and RNG state exactly despite uneven active masks.
- A real 32,768-world collector completed all 90 decisions / 360 physics ticks
  per world in **14.959 seconds** after initialization. It produced **4,423,680
  learner decisions** and **11,796,480 physics world-ticks**. **1,474,560 learner
  decisions** faced Nexto, exactly one third, with opponent controls excluded
  from optimization masks. Self-play-only worlds kept zero Nexto controller
  caches. Real goals/resets were observed: 1,785 each. No nonfinite failure.
- Peak Torch allocated memory: 12,764,209,152 bytes; reserved: 15,898,509,312.
  These are allocator measurements, not total device memory or PPO peak memory.
  This was rollout-only; no claim about backward/optimizer performance is made.

Full numerical data, raw short-match telemetry and hashes are in
`gpu_results.json`, `gpu_match_smoke.npz` and `gpu_protocol.json`.
The final combined CPU/evidence/adjacent-regression suite passed **27 tests**
in 5.78 seconds; see `tests_final.xml`. This also verifies the earlier causal
and full-match artifacts remain bound to their unchanged historical sources.

## Unchanged-policy integration smoke

The deterministic variant played ten fixed 20-second worlds: **2 goals for,
10 against, 23 Rival touches; one world without a Rival touch**. All ten matches
are deliberately unresolved, not ten complete losses. This reproduces the
aggregate outcome of the earlier corrected-timing diagnostic. It is a wiring
check, not evidence of learning or native competence. The real native-v5 sampling
mode was tested in the full-scale collector; the two modes are not mislabeled.

The reference600 checkpoint remains:

`checkpoints/rival2/direct_skills_v1/plus_000600.pt`

SHA-256: `8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2`

Model tensor SHA-256:
`1219DDFFD91F1E6DDDBE6C3AAACCAEC622CE9EA8D69168BBB4B9588D064F888B`.

## Remaining scope and next action

The controller correction is validated for the tested batched path. It does not
prove exact Rocket League physics or all-native Nexto observation parity.
Simulation does not fabricate native countdown ticks. Full-match kickoff phase
and the existing collector's centered-ball kickoff inference remain distinct;
that limitation is explicit. Same-device RNG restoration is exact, not native
CPU/GPU samplewise RNG equivalence. A fresh physical-episode resume discards
controller caches explicitly; it is not exact physical-world continuation.

Before another learning block, freeze a corrected campaign configuration with
this collector, explicit mode/seed, exact parent and checkpoint-resume semantics.
Use corrected opponent evaluation consistently; old 8/10 simulator wins must
not be treated as the new baseline or proof of native strength. Do not repeat
the completed native games, causal comparisons or controller tests as a new
investigation. Preserve both reference600 and finishing650 and their negative
evidence. The SSL goal is active and unachieved.

Shooting already occupies 20% of direct-skills starts. The latest finishing
reward authority requires actual goals rather than rewarding a projected shot.
This task did not add another shooting reward or restart any learning campaign.
