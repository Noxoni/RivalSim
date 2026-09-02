# Rejected short-horizon run

Correction 3 was stopped after block 37 entered the training curve.  The last
saved boundary was block 36, with external checkpoint SHA-256
`F64082BDD028D143959048DF2E925D3F94E27756C350F9D488CFF35A163B1C64`.
It was not promoted and is not a parent for subsequent training.

The momentum correction worked: on the block-36 fixed attacking-half
validation, useful >=600 uu/s goalward contacts reached 25.0%/22.0%, elevated
follow contacts reached 44.6%/49.9%, and the deepest-trajectory median reached
Y=4,107/4,086 uu.  The ball was centered when it got deep: 23/23 and 19/25
deep trajectories were inside the goal mouth.  It still scored only 3/1 goals
out of 1,024 attempts.

The remaining constraint was artificial time.  The pack horizon was 300 ticks
(2.5 seconds), while the user's reference clip takes roughly five seconds from
the low setup/lift through the live goal.  At block 36, 550/550 attempts ended
on the ground and approximately 471/473 additional attempts reached the
horizon without a goal.  The user's hard constraint is no more than six
distinct contacts, not a 2.5-second finish.

Correction 4 retains the six-contact cap and all momentum/goal targeting, adds
explicit horizon-timeout telemetry, and extends the horizon to 600 ticks (five
seconds).  It restarts from the byte-exact passing V2 checkpoint.
