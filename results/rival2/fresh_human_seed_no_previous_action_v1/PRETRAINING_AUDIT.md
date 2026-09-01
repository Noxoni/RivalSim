# Fresh Human Seed, no-previous-action Stage-1 pre-training audit

Verdict: **PASS — optimizer steps remain forbidden until this authority commit is
pushed and read back from `origin/main`.**

## Frozen scope

- Authorization parent: `38BDD501BECC0F6BA0867EABB205F25D05E0F31C`
- Implementation commit: `B4FE38365FE86EAE7E046C16A78487F12D9597D3`
- New lineage name: `RIVAL2_FRESH_HUMAN_SEED_NO_PREVIOUS_ACTION_V1`
- This is not BC V6 and accepts no model checkpoint as a training input.
- Stage 1 is fresh-random gameplay imitation only. PPO, reward optimization,
  mechanic-practice data, retention objectives, and prior Rival weights are forbidden.

## Human source and split

- Reviewed session: `CD6E7DB1-2761-4B8B-BD37-F21C7F135722`
- Total frames: 58,306
- Chronological train / validation / test: 46,644 / 5,831 / 5,831
- Source/split manifest SHA-256:
  `35C3AB6A88415E810D49BD801508BBACF34897D30068F607DED395F02A946392`
- Observation Adapter V2 SHA-256:
  `EDEDC9CCDE3269B393FB4C944F641CF4D34A78AB5944662F9019009BBA914C99`

All 58,306 samples were rematerialized from the native recording. For every sample,
indices 167–174 were set to zero and marked unavailable before Adapter V2, then set to
zero again after Adapter V2 and native pad overlay. The manifest records 58,306 verified
input masks and 58,306 verified output masks.

## RivalSim policy boundary

The new policy config sets `zero_previous_action_inputs=true`. `Rival2ActorCritic.forward`
hard-zeros indices 167–174 immediately before the shared trunk. Unit testing proves actor
and critic outputs are bit-identical when only those eight caller-supplied values change.
The new policy contract hash is
`43C2A0850E1F1D04671C6DCE564784DF103656F1194B0E915FA1B89164CA9BE1`.
The legacy default policy hash remains
`58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136`.

## Fresh initialization and selection

- Fresh initialization seed: 2,026,090,106
- Initial model tensor SHA-256:
  `D30C918CAF2CAE62567176E04C17813129D0B8B91FAEEAFC065786CE6C4C0F1E`
- Shared trunk and action-producing actor rows train.
- Critic and actor log-standard-deviation rows receive no Stage-1 gradient.
- Selection uses only the lowest held-out validation complete-action RMSE.
- Test remains untouched until the selected validation checkpoint is fixed.
- After selection, only deterministic 120 Hz evaluation against Nexto is authorized.

The machine-readable authority is `authority.json`; its SHA-256 at preparation is
`6AE0B6D6CCC49724B2D25E8DC09B9F83AFC98B316B58D61EED83D363E8CF0AEB`.
