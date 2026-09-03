# Unified Capability V4 Physical Evaluation

V4 remains a diagnostic single-network checkpoint and is not promoted as the
official Rival.  The deterministic fresh-seed physical evaluation used one
policy output, one recurrent hidden state, no runtime router, and no task or
scenario identifier.

## Result

- Natural Nexto: 51 Rival goals, 205 Nexto goals, 1,277 Rival touches, zero
  no-touch truncations, and 1,210.98 uu/s mean Rival speed across 256 episodes.
- Controlled demos: 399 actual demos, 368 post-demo touches, and 279 post-demo
  goals.  The same-seed specialist control recorded 398, 369, and 281.
- Controlled dashes/recoveries: 62 productive floor landings, 86 productive
  wall landings, and 130 productive landing chains.  The specialist control
  recorded 70, 124, and 167.
- Controlled aerial: 12.3047% elevated follow touches, 1.0254% high follow
  touches, 1.2695% second airborne touches, 3.7598% productive continuations,
  and 0.2441% goals within the contact budget.  The specialist control recorded
  48.7305%, 29.1504%, 4.4434%, 14.5020%, and 2.2949% respectively.

V4 improved the unified aerial elevated-follow rate over V3's 4.1016%, but it
still failed the frozen aerial thresholds.  Demo transfer remained effectively
exact.  Floor and wall recovery behavior remained present but regressed from
both V3 and the specialist control.  Natural play remained active with no
no-touch collapse, but lost 51-205 against Nexto on the V4 seed set.

## Diagnosis

The V4 student-aerial corpus includes only frames after learned control becomes
active.  Training samples those windows with a zero recurrent hidden state.
During physical deployment, however, the same recurrent policy observes the
scripted setup, contact, first jump, jump release, and second jump before learned
control begins.  Its deployed hidden state therefore differs from the hidden
state used for V4 supervision.  Offline student-state action RMSE cannot prove
the correct action for that deployed recurrent state.

The next correction must preserve the full pre-control recurrent prefix and
apply the aerial imitation loss only after learned control begins.  It must
continue natural and capability rehearsal and must be evaluated physically on
fresh seeds without checkpoint reselection.
