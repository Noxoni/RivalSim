# Closed Codex Boundary — Rival 2.0 Campaign 03

Rival 2.0 Campaign 03 is complete. The bounded direct reward-density run implemented
`RIVAL2_REWARD_V2`, passed its sole authorized targeted GPU reward smoke, trained through the
first completed PPO update crossing 100,000,000 agent decision samples, published the final
checkpoint and one final evaluation, and stopped.

## Completed result

- reward: preserved `RIVAL2_REWARD_V1` plus the Campaign 03 per-agent approach delta in
  `RIVAL2_REWARD_V2`;
- Reward V2 content SHA-256:
  `54CD5AC582133D9BA77CF7DF7976C549B3E659920BA407C9ACCE8A9FD5F50B32`;
- targeted reward sign/reset-leakage smoke: `PASS_GREEN`;
- training: 131,072 worlds, horizon 32, Campaign 02 entropy-off PPO baseline;
- stop: update 12 / 100,663,296 agent decision samples;
- all training-update integrity records: 12 / 12 `PASS_GREEN`;
- final checkpoint reload/continuation: `PASS_GREEN`;
- single final 4,096-world stochastic self-play evaluation: `PASS_GREEN`;
- final checkpoint SHA-256:
  `A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`.

The final comparison with Campaign 02 is published in
`docs/RIVAL2_CAMPAIGN03_RESULTS.md` and `results/rival2/campaign03/`.

## Boundary

There is no active follow-on authorization in this file. Do not continue training, add another
reward term or curriculum, tune Campaign 03, begin v0.6, or start RocketSim/RLBot transfer work
without a new explicit handoff.
