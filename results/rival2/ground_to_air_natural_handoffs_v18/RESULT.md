# Exact natural aerial-handoff capture V18

## Verdict

**VALID STATE/OBSERVATION CORPUS; NOT A TRANSITION-EQUIVALENT RESET CORPUS.**

The read-only capture preserved all 250 aerial-option activations from the
frozen 512-world V23 self-play run at seed `2026090327`.  The artifact contains
the complete public `StateSnapshot`, lifecycle snapshot, all 78 Rival2 bridge
views, the exact `[250, 2, 182]` observation, both base actions, the protected
V3 option action, physical side, route, source world, and source tick.

Restoring the artifact reproduces every recorded observation bit exactly and
reproduces the protected V3 first action to floating-point inference tolerance
(maximum absolute difference below `4e-7`).  It does **not** reproduce the
subsequent continuous-match dynamics.  The isolated reset replay reports 152
goals for and zero against, while a fresh continuous evaluation using the same
checkpoint, router, 512 worlds, 6,000 ticks, and seed reproduces the established
result: 250 activations, 103 entry contacts, zero separated second contacts,
zero route goals, and 103 goals against while active.

The reset result is therefore diagnostic only.  Exact policy input does not
imply exact simulator transition state: public state plus the current Rival2
bridge/lifecycle views do not preserve every hidden/contact solver state needed
to restart the live trajectory at an arbitrary physics tick.  No promotion,
training target, or capability claim may use the isolated scoring figures.

## Integrity

- corpus SHA-256:
  `829284A872B5B7F9402773C92B2515E65E74A522B46552727672820876F1A0E3`
- semantic array SHA-256:
  `8D8A4AFC71999FE658DAE9D1575E52940E119AD8E2816C0AC63EE8B84FD05A43`
- protected V3 scorer SHA-256:
  `F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154`
- V23 Blue SHA-256:
  `0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C`
- V23 Orange SHA-256:
  `56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A`
- optimizer steps: `0`
- policy mutations: `0`
- reward changes: `0`

## Supported use

The corpus is authoritative for the observation and action distribution at the
handoff boundary.  It is suitable for input-domain analysis, action comparison,
and supervised/distillation studies that do not claim closed-loop physics
equivalence.  Capability promotion must continue to use uninterrupted natural
self-play or an expanded simulator snapshot format proven transition-equivalent
before use.
