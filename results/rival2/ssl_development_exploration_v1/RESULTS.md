# Controlled exploration pilot: negative/inconclusive

No checkpoint from this pilot is promoted as an SSL-capable model. Thirty
accepted PPO updates per arm completed without nonfinite/corruption failure.
Both arms began with the identical preserved update-597 model and Adam state;
only effective analog standard deviation differed. All snapshots remain saved.

| Independent original-episode cases (256 each) | Parent | Control +30 | Half sigma +30 |
|---|---:|---:|---:|
| Acquisition: cases touched | 53 | 50 | 57 |
| Acquisition: no-touch truncations | 239 | 241 | 236 |
| Finishing: cases touched | 190 | 191 | 192 |
| Finishing: goals for | 75 | 71 | 74 |
| Ongoing vs Nexto: goals for / against | 1 / 253 | 1 / 255 | 0 / 254 |
| Kickoff vs Nexto: cases touched | 52 | 103 | 0 |
| Kickoff vs Nexto: goals for / against | 0 / 256 | 0 / 256 | 0 / 256 |

These are scenario outcomes, not full-match wins. A world can make a touch and
later reach the no-touch timeout; those counts are not complements. Conditional
first-touch time excludes all failures. The frozen pilot decision remains
negative/inconclusive: half sigma did not beat control by the required four of
64 acquisition cases at both +20 and +30. The independent cases do not overturn
that decision. In particular, its kickoff result regressed despite a small
independent acquisition increase. Do not infer that all exploration changes
are useless or that no further PPO learning is possible from this short probe.

## Focused diagnosis

The preserved parent's native trajectories initially accelerated and boosted
toward the ball, then frequently passed/missed it and failed to return. During
seconds 1-3, 89.3% of the sampled active focal states were receding from the ball
at more than 100uu/s. This is measured approach/turning failure, not a claim of
learned aerial mechanics from airborne time.

A **scripted diagnostic-only** steering reference using the same final 182-field
observations reached the ball in 256/256 native cases, conditional median 1.175s.
It used no future state and was never deployed, added to training, converted to
human data, or credited as Rival's capability. It demonstrates that the current
observations and native controls suffice for these elementary ground approaches;
it is not a complete simulator-fidelity proof.

On three fixed first-rollout minibatches per final model, PPO mean-head gradient
norms were nonzero. Entropy-gradient norms were about 5-8% of policy-gradient
norms on log-standard-deviation heads, 4-7% on buttons, and 4-13% on actor features.
Entropy did not overwhelm the sampled gradients. Roughly half the raw log-std
outputs hit existing clamp bounds (mostly floor, partly ceiling); clamp gradients
are therefore absent there. This is a potential future exploration-design
limitation, not a proven cause or an implementation change. All checked stored
and recomputed log likelihoods matched exactly. No diagnostic optimizer step
occurred and all source checkpoints stayed unchanged.

## Next experiment

Test reset difficulty without changing rewards, exploration, model, optimizer,
or PPO settings: more attainable short ground approaches/finishes, still with
off-angle heading, freely controlled opponents, and broad original-state
coverage. A new prospective authority bounds that test. Do not restart random
weights, silently resume the stalled old campaign, add behavior rewards, or
substitute the scripted reference for learned gameplay.

Machine-readable detail, per-case data, exact hashes, full training curves and
gradient statistics are in `final_pilot_report.json`, `independent/`,
`diagnostics/`, and the two arm directories. The SSL development goal is ongoing.
