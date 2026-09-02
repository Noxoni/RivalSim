# Rejected uncapped-contact run

The correction-1 run was interrupted after two training blocks and was never
promoted.  Its authority paid every third-or-later separated airborne contact,
which could optimize touch accumulation instead of an efficient scoring chain.

The user supplied a 20.866-second Rocket League clip (SHA-256
`5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`)
showing the intended alternative: a low setup/lift followed by prolonged
low-separation nose-behind-ball control into the goal.  The same live play is
shown again as a replay in the second half of the clip.

Correction 2 therefore freezes the following semantics before restarting from
the byte-exact passing V2 parent:

- the low pop is distinct chain contact one;
- at most five separated airborne recontacts are eligible, for six total;
- a seventh contact ends the attempt as an over-budget failure;
- a goal is credited only while the chain remains within that budget;
- `lifecycle.self_touch_event` is treated as a unique onset, so a sustained
  glued carry is one contact interval rather than per-tick touch events;
- sustained low-separation control has separate bounded telemetry and one
  single-onset continuation event, with no raw airtime reward.

The two rejected blocks are retained in
`training_curve_rejected_uncapped_contacts.jsonl`; their weights were not
checkpointed or reused.
