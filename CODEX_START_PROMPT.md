# RivalSim Completed Boundary — v0.5 / Rival 2.0

RivalSim v0.5 is **COMPLETE / PASS_GREEN**.

The completed milestone provides the bounded Rival 2.0 GPU-native training stack on top of the
frozen v0.4 standard-Soccar 1v1 transition engine:

- zero-copy Warp/PyTorch CUDA state views;
- frozen `RIVAL2_OBS_V1`, hybrid native-action, reward, and episode contracts;
- exact four-tick/30-Hz control cadence;
- GPU-resident rewards, resets, rollout storage, GAE, and PPO;
- exact checkpoint/resume and deterministic evaluation;
- current-policy self-play and bounded GPU historical opponents;
- fixed-seed learning sanity and practical world-count performance evidence;
- all inherited v0.4/v0.3/v0.2.2/v0.1 regressions green.

Read the completed result package before proposing further work:

- `VERSION.md`;
- `docs/V0_5_RESULTS.md`;
- `docs/REPRODUCING_V0_5.md`;
- `docs/RIVAL2_TRAINING_CONTRACT.md`;
- `results/v0.5/manifest.json`.

The controlling implementation handoff remains archived under `handoff/v0.5/`. Published
`results/v0.1/` through `results/v0.4/` are frozen and byte-for-byte unchanged.

There is no active next milestone authorization. Do not begin v0.6 deployment, CPU RocketSim or
RLBot integration, Rocket League transfer evaluation, curricula, legacy Rival/Wisp compatibility,
or broader simulator work without a new explicit controlling handoff.
