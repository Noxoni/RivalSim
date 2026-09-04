# SSL Foundation V2: update-600 PPO and independent-critic amendment

The user requested policy LR `1e-4`, two PPO passes, 30-second evaluations,
more Nexto than frozen V5, and an independent multilayer critic. This continues
the accepted corrected V2 update-600 model, not the invalid V1 SSL lineage.

## Settings and training lineage

- Actor/shared-policy LR: `1e-4`; epochs: `2`.
- Critic LR: `3e-4`; independent `182 -> 512 -> 512 -> 512 -> 1` SiLU MLP.
- The value network starts from copies of the update-600 policy trunk and value
  head. Initial actor, recurrent state output, and value predictions are exact.
- Actor and original value-head Adam moments and step counters are preserved.
  Only the new independent critic feature parameters have fresh Adam state.
- The actor's architecture is unchanged. The augmented model has a distinct
  policy-config hash and `RIVAL2_INDEPENDENT_CRITIC_MLP_3X512_V1` identity.
- Opponent assignment probabilities: 40% current Rival, 40% Nexto, 20% original
  frozen Unified V5. These are episode assignment probabilities, not guaranteed
  sample shares; the existing telemetry records actual sample shares.
- Physics/policy: 120 Hz; 32,768 worlds; rollout: 128 ticks. Recurrent sequence
  handling, exploration, gamma/lambda, clipping and the six-potential reward
  are unchanged. KL remains telemetry only; nonfinite rollback remains active.
- Training episode semantics remain goal terminal, 15-second no-touch
  truncation, and 45-second episode cap. The requested 30 seconds changes the
  evaluation duration; it does not force every training episode to last 30 s.

## Evaluation comparability

The original update-0 through update-600 evaluations lasted only 1,200 ticks
(10 seconds). Their zero no-touch-reset counts could not test the 15-second
timeout. Do not use those zeros as evidence that no-touch behavior was solved.

This amendment starts a separate manifest with a 3,600-tick evaluation at the
unchanged update-600 actor. Subsequent evaluations use that same 30-second
method and fixed seed. Never compare raw goal counts between the old 10-second
evaluations and these 30-second evaluations. Snapshots/evaluations remain every
50 cumulative updates: next learned checkpoint evaluation is update 650.

## Resume and interruption

```powershell
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_foundation_ppo_v2_amended.py
```

The default starts from the migrated update-600 checkpoint at
`G:/dev/RivalSim-runs/ssl-foundation-ppo-v2-amendment-v1/amended_start_u0600.pt`.
For operational recovery use the same script with `--resume` pointing to this
amendment's `rolling.pt`. Continuation is user-authorized beyond update 600.
Creating `STOP_REQUESTED` in the amendment run directory requests a stop at the
next accepted update boundary. Do not run two campaign processes concurrently.

As with prior V2 resumes, physics episodes and recurrent hidden state restart;
the checkpoint does not serialize live simulator state. Model, optimizer,
training counters and RNG state carry forward. This limitation is explicit in
the authority and checkpoint rather than claiming exact live-world replay.

## Validation and artifacts

`test_rival2_ssl_foundation_amendment.py` checks real-checkpoint actor/value/hidden
parity, independence of actor and critic gradients, trainability of every critic
layer, preserved Adam slots and RNG tensors, and rejection of incompatible
parents. Adjacent recurrent PPO and corrected-V2 tests also pass.

The 32,768-world preflight passed, including exact actor/value parity on native
simulator observations (maximum absolute differences both zero). Authority,
launch binding, migrated checkpoint metadata and preflight are adjacent JSON
artifacts. The original update-600 checkpoint and evidence remain preserved.
The migrated startup is also archived in Git under
`checkpoints/rival2/ssl_foundation_ppo_v2/amendment_v1/start_u0600.pt`.
