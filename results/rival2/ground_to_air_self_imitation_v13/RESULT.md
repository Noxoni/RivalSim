# Ground-to-Air Self-Imitation V13 Result

V13 completed the prospectively frozen 120-block ceiling without a hard
failure.  It trained only the aerial-option actor head from exact sampled
actions in the causal history of an authoritative second airborne contact or
in-budget goal.  The protected V23 policies, aerial trunk, and critic remained
unchanged.

The campaign accumulated 245,841 success-history samples.  Controlled
deterministic validation retained the scorer capability throughout the run.
The validation-selected block-40 snapshot recorded a 0.220703 second-airborne-
contact fraction, 0.189453 goal-within-six-contacts fraction, and 0.091797
ball-ground-failure fraction.

The required natural deterministic gate did not pass.  Across fixed-seed
validation at blocks 20, 40, 60, 80, 100, and 120, the router produced many
activations and first airborne contacts but zero second airborne contacts and
zero route-attributed goals.  Consequently no V13 checkpoint is promoted or
tested against Nexto.  Block 40 remains diagnostic evidence only.

This is a bounded negative result: success-conditioned action rehearsal
preserved the controlled skill but did not solve the controlled-to-natural
state-distribution handoff.  The next prospective attempt should retain the
exact protected multi-touch scorer while mixing its controlled aerial states
directly into ordinary self-play rather than further distilling stochastic
natural rollouts.
