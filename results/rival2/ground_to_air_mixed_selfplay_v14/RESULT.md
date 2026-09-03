# Ground-to-Air Mixed Self-Play V14 Result

V14 completed all 120 prospectively frozen PPO updates without a hard failure.
It started from the exact protected V3 aerial scorer and trained its actor and
critic heads in a 32,768-world mixture containing 75% ordinary self-play and
25% authoritative controlled aerial opportunities. The protected V23 base
policies and aerial trunk remained frozen. No Nexto training, named-mechanic
detector/reward, raw-airtime reward, or KL rejection was introduced.

The campaign retained the scorer's controlled deterministic capability. At
update 120 it recorded a 0.232422 second-airborne-contact fraction, a 0.162109
goal-within-six-contacts fraction, and a 0.082031 ball-ground-failure fraction.
Across training it recorded 50,309 entry airborne contacts, 2,842 separated
second airborne contacts, and 2,359 goals within the six-contact budget. The
update-120 snapshot is diagnostic evidence at SHA-256
`837BE27887EA32E01602087A3F33967D514212E9D7A2C2C4A7AF87255D1917B9`.

The required natural deterministic gate did not pass. Fixed-seed natural
validation at updates 30, 60, 90, and 120 produced zero second airborne
contacts and zero route-attributed goals. Update 120 produced 160 route
activations and 103 entry airborne contacts, but all 103 entry-contact routes
ended on reset before a separated second contact; the other 57 ended when the
ball grounded. Touch acquisition remained healthy at 15.748828 touches per
player-minute with zero no-touch players.

No V14 checkpoint is promoted and the untouched natural test remains unopened.
The validation-best update-90 snapshot also remains diagnostic only. This is a
bounded negative result: the multi-touch/scoring option itself remains real in
controlled states and can learn from mixed self-play, but natural takeover
still places or terminates it in a state sequence that does not reach the
second touch. The next work should diagnose and align the natural route
handoff/lifecycle against the scorer's successful first-touch state
distribution rather than retrain the aerial controller or promote V14.
