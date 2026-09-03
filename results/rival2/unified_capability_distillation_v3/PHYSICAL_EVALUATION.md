# Unified Capability Distillation V3 Physical Evaluation

V3 is one recurrent network with no runtime routing. Its student-induced
natural rehearsal materially corrected V2's closed-loop Nexto failure, but the
new context parameters shifted the aerial closed-loop state distribution.

## Natural Nexto result

- V2: 0 Rival goals, 256 Nexto goals, 0 hard-time ties.
- V3: 51 Rival goals, 154 Nexto goals, 51 hard-time ties.
- All 256 V3 episodes contained a Rival touch and none ended by no-touch.
- Mean speed increased from 1,172.98 to 1,282.59 uu/s.

## Controlled capability result

The demo/dash behavior remained intact relative to the specialist control:

- actual demos: 400 unified / 399 specialist;
- demo follow-up touches: 373 / 370;
- demo follow-up goals: 285 / 287;
- productive floor landings: 81 / 76;
- productive wall landings: 142 / 121;
- productive landing chains: 199 / 170.

## Controlled aerial result

The physical aerial behavior did not survive V3 even though validation on
teacher-induced aerial sequences improved. Unified elevated follow touches were
4.10%, high follow touches 0.10%, second airborne touches 0.44%, productive
continuations 1.37%, and goals within contact budget 0.20%. The specialist
control passed on the same seeds.

## Verdict

`PARTIAL_NOT_PROMOTED`

This is direct evidence of aerial student-state distribution shift. A
prospective correction should label V3-induced aerial states with the frozen
aerial teacher while continuing V3-induced natural rehearsal. It must remain a
single deployed network with no router.
