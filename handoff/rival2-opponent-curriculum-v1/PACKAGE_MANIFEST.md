# Rival 2.0 Opponent Curriculum V1 — Package Manifest

## Package purpose

This package authorizes implementation and one bounded 120-update training phase
from the healthy Gameplay V1 +239 checkpoint using a mixed frozen-opponent/self-
play curriculum plus a tiny successful-double-dash reward.

Creating this package does **not** itself run training.

## Package files

- `handoff/rival2-opponent-curriculum-v1/README.md`
  - authoritative curriculum, reward, opponent mix, boundaries, evaluation, and
    return requirements.
- `handoff/rival2-opponent-curriculum-v1/WISP_SOURCE.md`
  - pinned Wisp v2-75B source/model identities, source-faithful port rules, and
    the required targeted fidelity gate.
- `handoff/rival2-opponent-curriculum-v1/PACKAGE_MANIFEST.md`
  - this manifest.
- root `CODEX_START_PROMPT.md`
  - active entrypoint directing Codex to this package.

## Required experiment source

The package was authored against RivalSim source commit:

`1a4437fe92fa7ab66efd0e4100d74bb90302ea46`

That commit must remain an ancestor of the implementation head. The implementation
must start from the active package head, not detach or reset `main` back to the
source commit.

Rival parent checkpoint:

`checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt`

SHA-256:

`77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`

## Frozen external opponents

### Nexto

- upstream commit:
  `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`
- model SHA-256:
  `BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`
- use existing verified RivalSim integration.

### Wisp v2-75B

- upstream repository: `NicEastvillage/RLBot-Wisp-v2-py`
- upstream commit:
  `58d4ab18fd0c92529b5ae6582ecf1713a6b1887a`
- policy Git blob:
  `2262095fbeb41846f0c68934aeefac5c56984f4a`
  - size: `7,689,613` bytes
- shared-head Git blob:
  `201917fac35badb7c9846688416b706082a3002a`
  - size: `5,995,907` bytes
- implementation must compute and freeze byte SHA-256 identities before
  training.

## Existing authoritative evidence reused

- Gameplay V1 health/transition evidence:
  `results/rival2/gameplay_v1/`
- Gameplay V1 versus Nexto checkpoint-selection evidence:
  `results/rival2/gameplay_v1_nexto/`
- dash-mechanics contract:
  `results/rival2/gameplay_v1_nexto/dash_mechanics_contract.json`
- dash contract content SHA-256:
  `F8FCF018A1D013D84DEF0F16D5033CE86F56B3F82CB95C1022F32722D1FB7510`

Do not regenerate these historical artifacts merely to satisfy a new package.

## Authorized delta

Only the following curriculum changes are authorized relative to Gameplay V1:

1. new mixed opponent selection:
   - 35% Nexto;
   - 35% Wisp v2-75B;
   - 20% current Rival self-play;
   - 10% historical Rival;
2. new immutable `RIVAL2_REWARD_GAMEPLAY_V2`, equal to Gameplay V1 plus
   `+0.005` for the current player's strict successful double-dash completion and
   the corresponding zero-sum opponent subtraction;
3. the state/provenance/adapter machinery required to run frozen Wisp and the
   new opponent selector;
4. online strict-double-dash state required for the reward, proven equivalent
   to the existing offline classifier;
5. opponent-specific/aerial/dash telemetry required by the README.

Everything else remains bounded by the README.

## Stop boundary

Training stops after exactly +120 PPO updates from source iteration 359, or at
the first rejected KL-safety update if earlier.

No automatic continuation is authorized.
