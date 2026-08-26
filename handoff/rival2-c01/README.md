# Rival 2.0 Campaign 01

This handoff authorizes one bounded first-play training campaign using the completed RivalSim v0.5 / Rival 2.0 stack.

## Goal

Produce the first real trained Rival 2.0 checkpoints and measure what emerges without changing the frozen v0.5 contracts.

The campaign is intentionally conservative in scope:

- fresh Rival 2.0 initialization;
- normal v0.5 PPO defaults;
- 32-decision rollouts;
- 100M agent decision samples;
- fixed checkpoints/evaluations at 0/10M/25M/50M/100M thresholds;
- no curriculum, reward tuning, action changes, or deployment work.

## Controlling files

1. `CODEX_START_PROMPT.md`
2. `handoff/rival2-c01/CAMPAIGN.md`
3. `handoff/rival2-c01/ACCEPTANCE.md`
4. `docs/RIVAL2_TRAINING_CONTRACT.md`
5. `results/v0.5/manifest.json`

Where there is conflict, the order above controls, except that no campaign document may alter the frozen v0.5 contract hashes.

## Output location

Use a dedicated campaign namespace, for example:

- `results/rival2/campaign01/` for compact committed evidence;
- `checkpoints/rival2/campaign01/` for committed checkpoint artifacts that satisfy the size policy;
- a documented ignored local artifact directory for any larger resumable checkpoint.

Do not modify `results/v0.1/` through `results/v0.5/`.

## Stop condition

Stop after the first completed PPO update crossing 100M agent decision samples, the corresponding frozen evaluation, checkpoint custody, evidence publication, and final campaign report.

No v0.6 work is authorized.