# Rival 2.0 Mechanics Catalog V0.1

Status: **research/design index**

This file is the navigable catalog for the mechanics reward research. Detailed physics are defined in:

- `docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`
- `docs/RIVAL2_MECHANICS_DETECTOR_PHYSICS_V0_1.md`
- `docs/RIVAL2_MECHANICS_FLICK_VARIANTS_V0_1.md`

This catalog does **not** authorize a training reward change.

## Status meanings

- **LOW REWARD CANDIDATE** — when the physical event completes, it is a candidate for the common low mechanics bracket in a future reward contract.
- **SAME-FAMILY SUBTYPE** — useful named/physical variant; completion uses the parent family's payout rather than stacking another payout.
- **COMPOUND** — contains two or more separately observable physical accomplishments. Each component may be rewardable only if the future contract authorizes it and the mechanics budget allows it.
- **TELEMETRY / TACTICAL ONLY** — useful state or tactic, but no direct mechanics payout proposed.
- **OBSERVE FIRST** — physically interesting but not yet justified as a direct reward candidate.

## 1. Dash and movement family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Wavedash | LOW REWARD CANDIDATE | Actual airborne dodge + near-surface landing transition + useful surface-tangent momentum result. |
| Landing wavedash | SAME-FAMILY SUBTYPE | Wavedash whose airborne phase was already a landing rather than necessarily a fresh jump. |
| Wall dash | SAME-FAMILY SUBTYPE | Successful dash on wall-like surface normal. |
| Curve dash | SAME-FAMILY SUBTYPE | Successful dash on curved-transition surface. |
| Ceiling dash | SAME-FAMILY SUBTYPE | Surface-relative successful dash on ceiling; ceiling context retained. |
| Side-entry dash | SAME-FAMILY SUBTYPE | Successful dash with lateral entry geometry. |
| 180 dash | SAME-FAMILY SUBTYPE | Successful dash plus large controlled heading reversal while retaining useful tangent motion. |
| 360 dash | SAME-FAMILY SUBTYPE | Successful dash sequence with full heading/orientation loop; same dash payout if physical success occurs. |
| Rival double dash | LOW REWARD CANDIDATE | Two successful measured dash outcomes with intervening support/contact and net sequence tangent-speed gain. Project definition remains based on observed Rival telemetry. |
| Zapdash | LOW REWARD CANDIDATE | Front-wheel-first landing -> partial grounded jump -> pop -> successful landing dash, with useful tangent result. |
| General chain dash | OBSERVE FIRST | Three or more successful dash transitions. Keep counts/physics; do not allow unbounded payout. |
| Haon/Horse-style chain dash | OBSERVE FIRST | Single/partial wheel support plus repeated pop/dash resource transitions. Map to dash primitives if discovered, then evaluate frequency/exploit risk. |
| Speedflip | LOW REWARD CANDIDATE | Forward-diagonal dodge with early pitch arrest/cancel, retained translation, and rapid forward-axis/travel realignment. |
| Half-flip | LOW REWARD CANDIDATE | Backward dodge cancel/reorientation producing useful approximately 180-degree heading reversal with retained motion. |
| Fast aerial | TELEMETRY / TACTICAL ONLY | Basic movement efficiency should be learned through ball/play reward unless evidence shows a bottleneck. |
| Generic recovery/landing | TELEMETRY / TACTICAL ONLY | No direct recovery reward proposed; gameplay already incentivizes returning to useful play. |

## 2. Dodge-resource / reset family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Ball flip reset | LOW REWARD CANDIDATE | Airborne car gains/re-gains untimed dodge when >=3 wheels support on ball and resource state transitions accordingly. |
| Chain/double reset | LOW REWARD CANDIDATE | Previously acquired resource is consumed/lost, then a distinct later reset reacquires it before world grounding. |
| Pre-flip reset | SAME-FAMILY SUBTYPE | Dodge is consumed before the ball-wheel reset that restores it. |
| Pop reset | COMPOUND | Valid reset acquisition followed by quick use of new dodge to create a controlled upward/outward ball pop while aerial control remains. |
| Rapid reset | SAME-FAMILY SUBTYPE | Fast pre-flip/resource-consumption sequence followed by short-latency reset reacquisition. |
| Pancake/wall reset | SAME-FAMILY SUBTYPE / COMPOUND | Ball-wall compression coincides with wheel-grounding reset. Reset family payout; pinch component is separately observable if it actually completes. |
| Car reset | LOW REWARD CANDIDATE | Same resource transition as ball reset but wheel support body is another car. |
| Ceiling retained dodge | TELEMETRY ORIGIN | Falling/leaving ceiling without jumping naturally preserves untimed dodge; departure alone does not pay. |
| Ceiling shot | COMPOUND | Ceiling-retained resource is later used in meaningful aerial ball manipulation/flick/redirect. |
| Stall reset | COMPOUND | Stall/near-zero translational dodge participates in later valid reset acquisition. Bare stall does not pay. |
| Musty reset / reset-through-Musty motion | COMPOUND / SUBTYPE | Musty-like rotation participates in acquiring/reacquiring reset. Resource acquisition and any later flick are separately auditable. |
| Lix/Hel-style retained-resource launch | OBSERVE FIRST | Surface/jump/dash sequence intended to launch while preserving useful resource. Require real resource/momentum benefit before considering payout. |

## 3. Possession and ground-control family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Controlled two-touch possession | LOW REWARD CANDIDATE | Same player makes second distinct touch within one uninterrupted possession epoch. |
| Controlled three-touch possession | LOW REWARD CANDIDATE | Same possession reaches third distinct touch; lock out further mechanics payout for that epoch. |
| Ground dribble/carry | LOW REWARD CANDIDATE | Ball remains in calibrated upper-car support/control region with stable relative motion. One sequence event, not per tick. |
| Bounce dribble | LOW REWARD CANDIDATE | Controlled possession survives one or more ball-ground bounce -> player-touch cycles. |
| Soft catch/control acquisition | OBSERVE FIRST / ORIGIN | Incoming uncontrolled ball is converted into a stable possession state. Important origin telemetry; direct reward only if later evidence justifies it. |
| Roof/rear catch | SAME CONTROL ORIGIN | Stable upper/rear possession geometry used by delayed flicks/Musties. |
| Nose catch / nose-loaded control | SAME CONTROL ORIGIN | Ball controlled farther forward in car-local frame; important for flat/fast release options. |
| Catch-pop | COMPOUND ORIGIN | Controlled catch followed by upward ball displacement while maintaining continuation; later air-dribble/flick/reset remains separate. |
| 360 dribble / two-wheel dribble | TELEMETRY ONLY | Style/control variants; possession reward already captures useful control without paying rotation/style itself. |
| Low 50/50 | TELEMETRY / TACTICAL ONLY | Tactical challenge choice. Outcome/game reward should teach when it is useful. |
| Fake challenge / fake flick without later mechanic | TELEMETRY / TACTICAL ONLY | No direct deception reward. |

## 4. Aerial-control family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Ground-to-air continuation | LOW REWARD CANDIDATE | Controlled ground touch/pop creates upward ball motion, Rival becomes airborne, then earns next distinct aerial touch before opponent/landing. |
| Two-touch air dribble | LOW REWARD CANDIDATE | Two distinct Rival airborne touches inside one aerial possession epoch. |
| Three-touch air dribble | LOW REWARD CANDIDATE | Same aerial epoch reaches third touch; further touches are telemetry/gameplay reward only. |
| Extended air dribble | TELEMETRY AFTER CAP | More than three touches remain measured but do not print unbounded mechanics reward. |
| Wall-to-air dribble | SAME-FAMILY ORIGIN | Aerial control epoch originates from wall. |
| Air-dribble bump | COMPOUND | Aerial-control capability plus ordinary demo/bump outcome; do not invent an extra bump-mechanic payout. |
| Air-dribble fake/drop | TELEMETRY / TACTICAL ONLY | Variant selection/deception should be learned from opponent outcome. |

## 5. Flick family

All successful flick variants live in the same low mechanics bracket unless a future contract deliberately says otherwise. They remain distinct physical/tactical labels.

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Front flick | LOW REWARD CANDIDATE | Controlled possession + actual forward dodge contact + deliberate ball release. |
| Side/diagonal flick | LOW REWARD CANDIDATE | Controlled possession + corresponding actual dodge/contact geometry + release. |
| 45-degree flick | SAME-FAMILY SUBTYPE | Off-axis release geometry and diagonal/side-biased dodge. |
| 90/180/270 flick | SAME-FAMILY SUBTYPE | Orientation-path variants; setup time/release angle retained. |
| Delayed flick | TACTICAL SUBTYPE | Valid flick after measurable delayed possession/setup. Delay itself does not pay. |
| Mawkzy-style power flick | SAME-FAMILY SUBTYPE | Rapid off-axis setup, car positioned more behind ball, backward-diagonal power transfer, flatter/faster release and useful recovery signature. |
| JZR-style high-gain 45 | SAME-FAMILY SUBTYPE | 45-family setup with cancel/reorientation producing a higher-gain powerful release. |
| Musty | LOW REWARD CANDIDATE / DISTINCT SUBTYPE | Backward-dodge rotating hitbox surface materially scoops through ball; rotational contact-point velocity is part of the transfer. |
| Reverse Musty | SAME-FAMILY SUBTYPE | Same rotational-scoop invariant with opposite/backward release geometry. |
| Roof-catch Musty | SAME-FAMILY ORIGIN | Musty from stable roof/rear catch. |
| Nose-loaded Musty | SAME-FAMILY ORIGIN | Musty from forward/nose-loaded control geometry. |
| Catch-pop Musty | COMPOUND ORIGIN | Catch/pop transition followed by Musty scoop. |
| Reset/ceiling/wall Musty | COMPOUND/ORIGIN LABEL | Resource/origin event plus later Musty release. |
| Hidden/occluded Musty | TACTICAL SUBTYPE | Defender sight geometry/occlusion retained; no occlusion bonus. |
| Breezi | LOW REWARD CANDIDATE / DISTINCT SUBTYPE | Sustained tornado-like roll/yaw setup while preserving control -> Musty-class terminal scoop. |
| Classy/tornado-inverted backflip flick | SAME-FAMILY SUBTYPE | Roll/yaw setup reaches inverted state -> backward-dodge flick. Distinct path from Breezi. |
| Suffo-style fake-away flick | SAME-FAMILY SUBTYPE | Initial chassis motion away from controlled ball -> diagonal-backward dodge returns for late release contact. |
| Wizard/Evoh-style late-dodge underside flick | SAME-FAMILY SUBTYPE | Dodge initially travels under/past ball -> later phase of same dodge re-enters ball path and creates actual release. |
| Bismillah/inverted front-scoop | SAME-FAMILY SUBTYPE | Inverted/near-inverted setup + forward-biased dodge whose front/corner rotating surface scoops ball. |
| Dash flick | COMPOUND | Successful dash primitive plus later successful flick primitive. |
| Luther-style ground-pinch flick | COMPOUND | Flick/control release plus actual ground-pinch compression outcome if both occur. |
| 180 wavedash flick | COMPOUND | 180-dash subtype followed by distinct flick release. |

## 6. Redirect/rebound family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Redirect | LOW REWARD CANDIDATE | Incoming moving ball is materially redirected by Rival while retaining useful outgoing speed. |
| Sidewall read/redirect | SAME-FAMILY SUBTYPE / COMPOUND REBOUND | Initial sidewall rebound is read and followed by a redirect. |
| Double tap/double touch | LOW REWARD CANDIDATE | Rival touch -> opponent backboard rebound -> Rival next touch materially redirects away from board. |
| Triple tap/extended rebound touch | SAME-FAMILY EXTENSION | Additional controlled rebound touches; reward family remains capped. |
| Psycho / Musty psycho | COMPOUND / OBSERVE FIRST | Extreme rebound/redirect plus optional Musty release. Existing redirect/flick primitives can represent it if discovered. |
| Doomsee-style wall/backboard dish | SAME REBOUND FAMILY | Rebound-read subtype if the authoritative geometry matches; goal outcome handled normally. |

## 7. Pinch family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Ground pinch | LOW REWARD CANDIDATE | Car-ball and ball-ground manifolds overlap with opposing compression normals, then release creates material ball transfer. |
| Wall/Kuxir pinch | SAME-FAMILY SUBTYPE | Same compression topology against wall. |
| Ceiling pinch | SAME-FAMILY SUBTYPE | Same topology against ceiling. |
| Post pinch | SAME-FAMILY SUBTYPE | Same topology against goalpost geometry. |
| Corner/Astral-style pinch | SAME-FAMILY SUBTYPE | Compression against corner/transition geometry. |
| BBL/backside pinch | SAME-FAMILY SUBTYPE | Pinch where active car contact geometry is backside/rear rather than conventional front; same compression payout. |
| Team pinch | NOT APPLICABLE TO CURRENT 1V1 / FUTURE | Would require two allied car contacts; no current 1v1 reward need. |

## 8. Pogo family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Pogo | LOW REWARD CANDIDATE | Airborne chassis corner/edge impacts world; normal motion reverses/redirects; car rebounds without settling grounded. |
| Reverse/back-corner pogo | SAME-FAMILY SUBTYPE | Pogo using rear corner geometry. |
| Wall/ceiling pogo | SAME-FAMILY SUBTYPE | Same rebound physics on non-floor surface. |
| Ball-synchronized/disguised pogo | TACTICAL SUBTYPE | Ball/car reach surface in same short window; occlusion and defender reaction retained. |
| Musty/catch-origin pogo | ORIGIN SUBTYPE | Pogo begins from setup that could also produce a flick/catch, preserving ambiguity telemetry. |
| Flingshot-style pogo-ball launch | COMPOUND | Pogo rebound plus later controlled ball contact/redirect. |

## 9. Stall family

| Mechanic | Status | Detector mapping / physical meaning |
|---|---|---|
| Bare stall | TELEMETRY ONLY | Actual dodge consumption with near-zero translational dodge effect. Not directly useful enough to pay by itself. |
| Stall -> reset | COMPOUND | Stall participates in later valid reset acquisition. |
| Stall -> controlled aerial continuation | COMPOUND | Stall followed by separately completed useful control/contact event. |

## 10. Explicitly not direct mechanics rewards

The following can be valuable but should remain normal game reward or tactical telemetry unless later evidence shows a learning bottleneck:

- generic jump;
- generic flip/dodge;
- generic aerial state;
- generic boost use;
- generic recovery/landing;
- simply driving fast;
- simply approaching the ball;
- being on a wall/ceiling;
- powerslide itself;
- being hidden behind the ball;
- waiting/delaying by itself;
- opponent committing early;
- fake challenge;
- low 50/50 choice;
- ordinary shot/clear/pass labels when no separate physical mechanic completes.

## 11. Reward-accounting principle

The mechanics bracket should tell Rival:

> A useful physical capability completed successfully; preserve the ability to do this.

It should not tell Rival:

> Use this mechanic now.

Variant selection is learned from ordinary return under opponent-conditioned states. Same low payout does not collapse variant identity because each event retains its complete physical/tactical label and surrounding state.

Any future implementation must retain family-level lockouts and a bounded mechanics-reward budget so high-frequency movement/contact events cannot dominate goals, saves, ball progress, touches, and other gameplay objectives.