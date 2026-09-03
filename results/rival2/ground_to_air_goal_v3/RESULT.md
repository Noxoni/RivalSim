# Rival 2 Ground-to-Air Goal V3 Result

Verdict: **controlled PASS; not yet integrated into the competitive policy**.

The block-68 checkpoint passed the frozen validation gate twice consecutively and then passed the untouched 2,048-attempt-per-side test. The critic, protected V23 policies, production reward, and passing V2 parent remained unchanged.

## Promoted isolated option

- Checkpoint: `checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt`
- SHA-256: `F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154`
- Selected block: 68
- Authority SHA-256: `E0E5786EDAF1107FD6A24A4345417F6BFDB3A0D8E49176D8728B71F2A40A66B8`

## Frozen validation result

At the second consecutive passing boundary, valid within-budget aerial-goal rates were 3.0% and 2.25% across the two mirrored 1,024-attempt validation sets. Elevated follow-touch rates were 47.66% and 51.27%. Neither side exceeded the six-contact budget.

## Untouched test result

The untouched test produced:

- side 0: 41/2,048 valid aerial goals (2.0019%), 48.97% elevated follow-through, 11.28% sustained control, 14.84% productive continuation;
- side 1: 47/2,048 valid aerial goals (2.2949%), 51.56% elevated follow-through, 12.26% sustained control, 16.21% productive continuation;
- combined: 88/4,096 valid aerial goals, zero goals over the six-contact budget, and one rejected seventh-contact overrun (0.0244%).

Separated contact chains remained concise. Second airborne recontacts occurred in 5.08%/5.52% of attempts, third recontacts in 1.90%/2.15%, fourth recontacts in 0.68%/1.07%, and fifth recontacts in 0.05%/0.29%. Low-separation nose-close carries were represented by a bounded sustained-control interval rather than per-tick touch events.

## Interpretation and next gate

This checkpoint has learned an isolated pop/follow/air-carry/finish option under controlled physical scenarios. It has not yet demonstrated safe selective invocation in natural gameplay, and it has not replaced or modified V23. The next stage is gated natural self-play integration: invoke the option only from physically eligible possession states, preserve ordinary ground behavior, and require deployment-relevant aerial contacts/goals plus competitive non-regression before promotion.

Machine-readable authoritative evidence is in `result.json`; the complete optimizer and evaluation history is in `training_curve.jsonl`.
