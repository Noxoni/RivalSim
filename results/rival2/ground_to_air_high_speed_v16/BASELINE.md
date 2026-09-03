# Ground-to-air high-speed V16 protected-scorer baseline

This is a read-only deterministic probe of the exact protected V3 aerial
specialist (`F7049F8E...CC77154`) on side-balanced, action-free RivalSim states
derived prospectively from the physical distribution observed at natural V23
self-play aerial handoff. A frozen V23 policy controls the live defender.

The 1,024 attempts produced:

- 928 airborne first contacts (`90.625%`);
- 7 separated second airborne contacts (`0.683594%`);
- 103 goals within the six-contact attempt budget (`10.058594%` by the probe's
  goal event, while the environment recorded 114 total goals for);
- 112 ball-ground failures (`10.9375%`);
- 8 goals against;
- no optimizer step, no scripted controller action, no reward change, and no
  checkpoint mutation.

The mean initial state was approximately a `429.7 uu` planar ball gap, `1702
uu/s` car goalward speed, `417.6 uu/s` ball goalward speed, `36.4%` boost, and
`1638.4 uu` opponent-to-ball distance. This is materially closer to the
measured natural V23 handoff distribution than the original low-speed V11
feeds.

## Interpretation

The V3 specialist is a useful scoring parent rather than a dead end: it
retains high first-touch acquisition and nontrivial one-touch/short-chain
finishing at realistic entry speed. Its principal missing capability in this
distribution is continuation after the first contact. A safe integration
campaign should therefore retain the original V11 multi-touch distribution,
add these high-speed states, and keep ordinary V23 self-play in the same batch.
Promotion must still be gated on deterministic natural self-play; this isolated
probe alone does not demonstrate a safe natural takeover.

Machine-readable evidence is in `baseline.json`.
