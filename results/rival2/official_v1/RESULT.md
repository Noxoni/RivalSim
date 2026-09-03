# Rival 2 official capability bundle V1 result

## Result

`READY FOR PLAYTEST (FAIL-CLOSED)`

The official source artifact is
`checkpoints/rival2/official_v1/rival2_official_v1.pt`, SHA-256
`20D03ECFAD8680D9F5464AEBA7C45B3FF86B3FD7FFDA50BE5160F3A4BF1EBC19`.
It contains the exact V23 Blue/Orange base models, Ground-to-Air Goal V3 aerial
model, and Capability Curriculum V2 Blue/Orange models in one checkpoint.

The validated playable path uses the V23 base component for each physical
side. It reproduced the accepted ten-match deterministic Nexto result exactly:
8-2, 159 Rival goals, 111 Nexto goals, and 687 Rival touches. The committed-
source rebuild is component-, router-, contract-, route-, and action-exact to
the physically validated artifact across 4,096 deterministic comparisons.

The RLBot deployment is committed in `Noxoni/Rival` at
`644f2c2cdf6d62c097a4568fe19cbc742d4d288f` under
`bot/rival2_official`. All five TorchScript exports passed exact analog/button
parity and the complete bundle self-test.

## Important limitation

The specialist weights are preserved in the official artifact, but their
automatic takeover is disabled. Enabling all routes produced 2-8 against
Nexto; disabling recovery but retaining aerial/demo takeover produced 3-7.
Those results are not hidden or promoted. This package is ready for a stable
human playtest, but it does not claim that aerial, dash/recovery, or offensive-
demo skills have been safely integrated into natural match control.

No optimizer step, reward change, or policy-parameter mutation occurred.
