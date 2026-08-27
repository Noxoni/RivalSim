# Active Codex Handoff - Rival 2.0 Gameplay V1

The active objective is the bounded acquisition-to-gameplay continuation described
in:

`handoff/rival2-gameplay-v1/README.md`

Use only the acquisition-complete source checkpoint:

`checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt`

Expected SHA-256:

`4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A`

The source commit is:

`61307571d86508f3026402c4948f759f310ff36c`

Do not resume any full-match Scoring V1 checkpoint. Preserve the acquisition
checkpoint's learned and training state, create fresh original short-lifecycle
world state, use only `RIVAL2_REWARD_GAMEPLAY_V1`, and run no more than 239
additional PPO updates. A policy-displacement rejection is a mandatory immediate
stop. Otherwise stop after the +239 checkpoint and held-out evaluation. Do not
begin an overnight continuation, train against Nexto, use five-minute training
matches, change simulator mechanics, or begin v0.6.
