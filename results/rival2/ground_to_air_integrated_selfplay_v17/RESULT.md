# Integrated aerial self-play V17 result

## Verdict

**NOT PROMOTED.** The prospectively frozen 240-update campaign completed with
no hard safety failure, but no checkpoint passed the deterministic natural
self-play transfer gates. The untouched natural test was therefore not opened.

The external snapshots remain inspection/rollback evidence only. The
selection rule ranked update 210 highest, with SHA-256
`3EDBFCFA653F6C16A7956558BC618F58661C9DE6D09F2FCC01E23E1E2140C1CC1`.
It is not a deployable promotion.

## What trained successfully

Across 240 accepted PPO updates, the option experienced:

- 163,938 airborne entry contacts;
- 6,212 separated second airborne contacts;
- 11,449 goals within the six-contact route budget;
- 51,541,905 option-policy samples by the selected update 210;
- no KL rejection (KL was telemetry only) and no nonfinite/corruption failure.

Compared with the first update-30 boundary, update 210 improved the original
controlled V11 macro second-contact fraction from `0.2421875` to `0.2548828125`
and its goal fraction from `0.1640625` to `0.1669921875`. On the independent
high-speed validation it improved airborne entry from `0.904296875` to
`0.9150390625`, in-budget scoring from `0.103515625` to `0.1123046875`, and
separated second contacts from `0.0078125` to `0.01171875`.

## Why it was rejected

Every fixed natural validation boundary recorded:

- zero separated second airborne contacts;
- zero goals for while the aerial route was active;
- 103 goals against while the aerial route was active.

The option therefore learned modestly inside both controlled distributions but
did not change the decisive naturally visited trajectory. Packaging it as the
ordinary playable Rival would knowingly expose the same unsafe handoff seen in
V14/V15. No promoted checkpoint was created.

## Next technical action

The next attempt should not repeat a broader approximate reset distribution.
It should capture exact full RivalSim states at natural router activation,
freeze those physical states as a training/validation corpus, and train or
distill against the actual failing handoff trajectory. The deployment artifact
must then bundle the V23 gameplay policy, aerial policy, and frozen router (or
distill them into one policy) so the result is genuinely playable as one Rival
system. Natural self-play remains the promotion authority.
