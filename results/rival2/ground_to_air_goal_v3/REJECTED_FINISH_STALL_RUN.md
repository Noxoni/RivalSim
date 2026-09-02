# Rejected finish-stall run

Correction 2 was stopped after block 49 entered the harder phase.  The last
saved checkpoint is block 48 and was not promoted.  It is retained as
`checkpoints/rival2/ground_to_air_goal_v3_diagnostic/rival2_block48.pt` with
SHA-256
`4E9CAF2A9417A2480EE8B0EF7C0AEFBA2D165C36C0C2374E869E53A194A6DE71`.

The policy materially improved launch continuation but did not make scoring
routine.  At the block-48 fixed validation boundary, the two canonical
perspectives produced one goal each out of 1,024 attempts (0.0977%), below the
prospective 1.5% gate.  The run was therefore stopped rather than advancing a
non-scoring option into self-play.

The independent 2,048-world-per-side deterministic finish diagnostic compared
the passing V2 parent with the block-48 descendant on the same attacking-half
states.  The descendant increased elevated follow contacts from 204/220 to
939/899, but scored only 2/2 goals.  Median maximum goalward ball position was
Y=4,057/4,031 uu, still about 1,160/1,190 uu short of the scoring plane.  Only
29/17 trajectories reached Y>=4,900; 19/8 of those were already inside both
goal width and height.  Median deepest X was 557/484 uu and median deepest Z
was 221/225 uu, so height and width were not the primary limitation.

The decisive failure was momentum retention.  Cumulative goalward velocity
transfer on eligible airborne contacts was -118,950/-121,132 uu/s.  Correction
3 must therefore condition recontact credit on a useful post-contact goalward
speed and apply the velocity-transfer term to every eligible recontact, not
only the first elevated contact.  It must restart from the byte-exact passing
V2 checkpoint; the block-48 diagnostic policy is evidence only.
