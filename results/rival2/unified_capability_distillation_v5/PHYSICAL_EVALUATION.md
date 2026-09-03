# Unified Capability V5 Physical Evaluation

V5 is the first single-network consolidation checkpoint to pass the controlled
aerial gate while retaining natural gameplay, offensive demos, and productive
floor/wall recovery behavior.  Evaluation was deterministic at 120 Hz on fresh
seeds, with one recurrent policy output, no runtime router, and no task or
scenario identifier.

## Result

- Controlled aerial passed every frozen threshold.  V5 recorded 48.2910%
  elevated follow touches, 26.0742% high follow touches, 4.2480% second airborne
  touches, 15.3320% productive continuations, and 2.1484% goals within the
  contact budget.  The same-seed aerial specialist recorded 49.2188%, 28.2227%,
  4.1016%, 14.1602%, and 2.5879% respectively.
- Controlled demos retained specialist behavior: both V5 and the control
  recorded 421 actual demos.  V5 recorded 394 post-demo touches and 305
  post-demo goals versus 397 and 300 for the specialist.
- Controlled dash/recovery behavior exceeded the same-seed specialist counts:
  90 productive floor landings, 136 productive wall landings, and 201
  productive landing chains versus 67, 109, and 154.
- Natural Nexto play remained active in all 256 episodes: 77 Rival goals, 154
  Nexto goals, 25 hard-timeout/tied episodes, 1,289 Rival touches, zero no-touch
  truncations, and 1,176.35 uu/s mean Rival speed.

## Comparison and conclusion

V4 failed the aerial gate at 12.3047% elevated follow touches and lost 51-205
to Nexto.  Replaying the complete recurrent prefix raised V5 to 48.2910% and
improved the same bounded natural methodology to 77-154 with 25 timeouts.  The
controlled capability correction also recovered and surpassed the lost floor
and wall behavior.

This validates the diagnosis: the decisive consolidation defect was recurrent
trajectory alignment, not the absence of a router.  The V5 checkpoint contains
the natural, aerial, demo, and recovery behaviors in one set of network weights
and one recurrent hidden state.  No component policy is consulted at runtime.

This artifact is a deployment candidate, not evidence that it can already beat
Nexto consistently.  Its Nexto result is materially improved but still losing.
The next training stage should begin only from this exact checkpoint, preserve
the unified recurrent architecture and controlled capability tests, and use
closed-loop self-play/Nexto learning without reintroducing a router.
