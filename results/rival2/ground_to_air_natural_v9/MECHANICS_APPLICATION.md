# Ground-to-air aerial mechanics applied to Natural V9

## Scope

This note translates observed human play and published Rocket League mechanics
guidance into the physical setup distribution for the next bounded Rival
training run. It is not a named-mechanic classifier and it does not add a
production reward. The run still starts from the protected controlled scorer,
uses the protected V23 policies only as inference-only defenders, and opens its
test seed only after the frozen validation gate passes.

The reference clip is
`C:/Users/patri/Videos/Medal/Edits/MedalTVRocketLeague20260902174434464-trim-1788386200969.mp4`,
SHA-256
`5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`.
The clip shows the relevant compact chain: approach a naturally rising ball,
make a light lift/setup touch, immediately follow from underneath, and use only
a few controlled airborne contacts to carry the ball into the goal. It does not
support a stationary dead-ball launcher requirement.

## Physical findings used

- Rocket Science's measured jump physics says the held first-jump force remains
  active for the first 200 ms. Its fast-aerial explanation also separates that
  sustained first-jump force from the second-jump impulse. This directly
  motivated the matched V8 timing sweep rather than treating the old 66.7 ms
  hold as authoritative.
  <https://rocketscience.fyi/know/videos/boost-and-jump>
- Dignitas's ground-air-dribble guidance with Yukeo emphasizes a bouncing ball,
  a hit and jump together immediately after the bounce, a low/underside touch,
  matching the ball's speed, and feathered boost. Those are the causal features
  represented by the `low_bounce` and `matched_dribble` scenario families.
  <https://dignitas.gg/articles/setting-up-an-air-dribble-and-performing-them-a-guide-with-yukeo>
- Dignitas's Stizzy guide explicitly distinguishes a bouncing approach (front
  of the car to pop, then double jump) from a rolling approach (soft touch,
  double jump, then follow). It also warns that overboosting and a heavy first
  touch turn control into a chase. That supports keeping `incoming_chip` as a
  first-class route and rewarding close, productive continuation rather than
  raw height or airtime.
  <https://dignitas.gg/articles/rocket-league-mechanics-air-dribbling-with-stizzy>
- ApparentlyJack's guidance says jumping with the ball gives immediate control
  and that no spin is often better than unnecessary continuous spinning. This
  is why V9 does not force a tornado spin into every launch. Narrow steer/yaw/
  roll residuals remain available for alignment while the simple double-jump
  route stays valid.
  <https://dignitas.gg/articles/a-guide-to-dribbling-in-rocket-league-with-apparentlyjack>
- The setup taxonomy is also cross-checked against current ground-air-dribble
  tutorials that separately demonstrate chip, bounce, and carry setups, plus
  tornado/reverse-tornado takeoffs. These are implementation references, not
  acceptance evidence:
  <https://www.youtube.com/watch?v=3miaW-kwQQg> and
  <https://www.youtube.com/watch?v=pl3Dno7DRjY>.

## V9 translation

V9 therefore keeps all three natural entry families and excludes a dead-ball
vertical launcher:

1. `low_bounce`: the ball already has a low bounce in front of Rival; the
   controlled scorer needs a light forward/underside contact and immediate
   takeoff.
2. `incoming_chip`: a rolling or low incoming ball can be chipped upward by a
   normal forward touch and followed.
3. `matched_dribble`: car and ball enter at compatible forward speeds so a
   double-jump touch can lift and follow the ball. The narrow air-roll residual
   can express a partial corner/tornado alignment when useful, but it is not
   mandatory.

The V8 no-learning calibration selected a 24-tick (200 ms) first-jump hold and
a four-tick release before the second jump. This improved the worst-row median
maximum car height from 92.72 uu to 141.76 uu and median upward speed from
288.69 uu/s to 387.39 uu/s without worsening the frozen broad proximity
measures. V9 uses that schedule prospectively from the original controlled
scorer; it does not load V7's diagnostic descendant.

The reward remains physical and bounded: proximity/progress, actual setup and
elevated contacts, productive goalward transfer, sustained close control, and a
goal within at most six distinct contacts. Raw airtime has zero reward. No
named mechanic is detected or rewarded. This lets a two- or three-touch finish,
a backboard follow-up, a reset-like contact, or a glued carry succeed based on
its physical outcome without requiring any of those labels.

## Current status at authority freeze

The protected controlled scorer reliably obtains many initial setup touches,
and the 200 ms timing produces materially better vertical launch physics.
However, the fixed parent still has at least one zero-elevated-follow scenario
row and no reliable high follow-up capability. V9 is intended to learn the
remaining alignment and continuation problem. Neither V7 nor the V8 timing
sweep is a promotable aerial policy, and V23 remains the competitive policy
until a prospectively gated validation and untouched test both pass.
