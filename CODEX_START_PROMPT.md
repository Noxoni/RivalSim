# Active Codex Handoff — RivalSim v0.5 / Rival 2.0 GPU-Native Training

RivalSim v0.5 is **AUTHORIZED**.

Start from current `origin/main`. The completed v0.4 release immediately below this handoff is:

`8a422a86c69f16f0d62073992e515575f88733b5`

The v0.4 implementation commit is:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

v0.4 is `PASS_GREEN`. All published v0.1 through v0.4 evidence is frozen and must remain byte-for-byte unchanged.

This milestone creates **Rival 2.0** as a clean-slate policy/training system built directly around RivalSim. Do not import, wrap, preserve, or target compatibility with the existing `Noxoni/Rival` training stack, its Wisp lineage, its 432-value observation, its 90/158-action tables, its policy weights, or its training schemas. The old Rival repository is not a v0.5 implementation dependency.

Before changing runtime behavior, read in full:

- `handoff/v0.5/README.md`;
- `handoff/v0.5/ACCEPTANCE.md`;
- `handoff/v0.5/RIVAL2_CONTRACT.md`;
- `docs/V0_4_RESULTS.md`;
- `docs/V0_4_AUTHORITY.md`;
- `results/v0.4/manifest.json`.

Mission order:

1. establish zero-copy RivalSim <-> PyTorch CUDA tensor interoperability and freeze `RIVAL2_OBS_V1`;
2. implement the fixed Rival 2.0 hybrid controller: five tanh-squashed Gaussian analog channels plus three Bernoulli buttons, held for four physics ticks (30 Hz decisions);
3. implement GPU-native reward, termination/truncation, reset, and rollout-buffer writes;
4. implement GPU-native GAE and PPO actor/critic learning, checkpoint/save/resume, and deterministic evaluation;
5. add GPU-resident self-play and bounded historical-policy opponent support without CPU inference in the rollout loop;
6. run end-to-end learning sanity gates, world-count/performance sweeps, residency checks, and all inherited simulator regressions;
7. publish compact v0.5 evidence and stop at the v0.5 boundary.

The normal training path must remain device resident:

`RivalSim GPU -> RIVAL2_OBS_V1 -> Rival 2.0 actor/critic -> 8 controls -> RivalSim x4 ticks -> GPU reward/dones -> GPU rollout -> GPU GAE/PPO -> updated policy`

CPU work is allowed for startup/configuration, logging, metrics snapshots, checkpoint files, and explicit offline evaluation/reporting. It must not become a per-decision or per-world participant in ordinary rollout or PPO data movement.

The action standard is settled. Do not run an action-space comparison study and do not reintroduce a fixed lookup table. The actor produces five continuous controls (`throttle`, `steer`, `pitch`, `yaw`, `roll`) and three binary controls (`jump`, `boost`, `handbrake`).

Do not begin v0.6 RocketSim/RLBot transfer work in this run. Complete and push v0.5 only if every gate in the v0.5 handoff passes; otherwise preserve all work and stop at the first genuine architectural, correctness, or performance boundary with explicit evidence.
