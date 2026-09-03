# Ground-to-Air Natural V11 Continuation Probe

This is a read-only diagnostic artifact, not a training authority and not a
reward change. It was added while the prospectively frozen V10 campaign was
still running; it does not modify or influence that process.

V10 proved that the controlled scorer parent already produces many low prompt
airborne recontacts for `low_bounce` and `matched_dribble`, while almost none
become strict elevated contacts, second recontacts, or aerial goals. The first
prompt contact generally occurs around ticks 20–29 after the setup contact.
The existing higher-level continuation shaping is gated behind car height 150
uu and ball height 250 uu, leaving a measurable physical interval between the
entry touch and the old gate.

`PromptContinuationProbe` measures that interval at native 120 Hz. It records:

- vertical and goalward ball-velocity transfer at the prompt touch;
- prompt ball and car vertical velocity and contact distance;
- closest subsequent car-ball separation;
- maximum subsequent car and ball height;
- maximum subsequent upward and goalward ball velocity;
- close, rising and close, goalward continuation ticks;
- maximum consecutive close-control ticks;
- a second separated native recontact;
- crossing into close elevated and close high geometry.

All continuation measurements are bounded to 120 ticks after the prompt touch.
The probe does not classify a named mechanic, synthesize a contact, mutate
state, emit an action, or award reward. A later V11 authority may be frozen
only after V10 terminates and no-learning calibration establishes prospective
thresholds and coefficients from these physical measurements.
