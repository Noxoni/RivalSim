# Natural ground-to-air mechanics and V10 curriculum

## Human reference and published execution guidance

The supplied 20.866-second Rocket League clip is
`C:/Users/patri/Videos/Medal/Edits/MedalTVRocketLeague20260902174434464-trim-1788386200969.mp4`,
SHA-256
`5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`.
The live play begins with a ball that is already bouncing ahead of the car.  A
compact lift/follow sequence gets the car below the ball, and only a few soft
airborne contacts are used before the finish.  The clip does not support a
stationary dead-ball vertical-launch requirement.

The training interpretation was cross-checked against these player and physics
sources:

- Rocket Science's measured jump physics: held first-jump force lasts 200 ms;
  the second jump is an impulse without the held-jump force.  This supports the
  already selected 24-tick hold and four-tick release at 120 Hz.
  <https://rocketscience.fyi/know/videos/boost-and-jump>
- Dignitas with Yukeo: for ground air dribbles, use a bouncing ball, contact and
  jump just after the bounce, stay at comparable pace, contact underneath, and
  feather boost.
  <https://dignitas.gg/articles/setting-up-an-air-dribble-and-performing-them-a-guide-with-yukeo>
- Dignitas with Stizzy: a bouncing ball can be lifted with the front of the car
  and followed by a double jump; a rolling ball should receive a soft touch,
  double jump, and follow.  Overboosting turns control into a chase.
  <https://dignitas.gg/articles/rocket-league-mechanics-air-dribbling-with-stizzy>
- AirCharged's current ground-to-air tutorial separately demonstrates chip,
  rolling-ball chip, and air-roll/tornado-touch setups.  It is used as an
  implementation reference, not as acceptance evidence.
  <https://www.youtube.com/watch?v=3miaW-kwQQg>

The mechanics therefore form multiple valid launch routes rather than one
mandatory animation:

1. **Natural low bounce.** The ball supplies the initial rise.  Rival approaches
   slightly faster, touches the lower/front region lightly, leaves the ground
   immediately, and matches the ball's rising momentum.
2. **Incoming-ball chip.** A low ball rolling toward Rival can be converted into
   a rising path by a forward/underside collision.  The car must then pursue
   that new path instead of treating the chip as a completed play.
3. **Matched dribble double jump.** With car and ball speeds already close, the
   car can jump through the ball and use the second-jump impulse to create
   separation while its nose is becoming goalward/upward.
4. **Partial tornado/front-corner variant.** Simultaneous yaw and roll can expose
   a front corner during the double-jump transition.  The useful action is a
   brief orientation correction stopped at the contact geometry; continuous
   spinning is neither required nor rewarded.  A plain double jump stays valid.

## What V9 actually proved

The old event called an aerial follow only after the car center reached 150 uu
and the ball reached 250 uu.  A deterministic 12-row, 256-world native probe of
the protected scorer showed that threshold discarded nearly all real initial
airborne recontacts:

| Entry family | Prompt airborne recontact range | Typical tick lag | Typical car / ball height |
| --- | ---: | ---: | ---: |
| low bounce | 34.8%--41.4% | 20--22 | 69--75 / 180--183 uu |
| incoming chip | 0.0%--1.6% | 23--26 where present | 83--93 / 215--223 uu |
| matched dribble | 78.9%--83.2% | 24 | 80--82 / 195--197 uu |

Across all rows, the protected scorer produced a prompt airborne recontact in
39.91% of attempts, versus only 0.33% meeting the old strict threshold.  V9's
best checkpoint reached 40.07% prompt recontacts, slightly worse any-airborne
coverage, and no high touch, second elevated contact, productive continuation,
or controlled goal.  V9 was correctly not promoted.

## V10 physical training contract

V10 restarts from the unchanged protected scorer.  It adds one bounded
training-only event worth 1.5 reward units for the first separated native
self-touch onset that occurs while the car is airborne and no more than 60
physics ticks (0.5 seconds) after the low setup touch.  It can be paid at most
once per attempt.  It does not reward raw airtime, height by itself, a named
mechanic, continuous contact duration, or controller inputs.

The existing downstream physical requirements remain:

- the old strict elevated contact at car >=150 uu and ball >=250 uu;
- a high follow and additional distinct goalward contacts;
- productive/connected continuation;
- a goal within no more than six total distinct contact onsets;
- both parked and live V23 defender rows, both team perspectives, and all three
  entry families.

Selection first closes the frozen prompt-recontact minima for every row, then
uses the existing complete-outcome score.  The prompt ratio is capped at one so
excess easy matched-dribble recontacts cannot outrank later physical progress.
The untouched test remains sealed unless the complete validation gate passes.
Production `RIVAL2_REWARD_GAMEPLAY_120_V2` is unchanged.
