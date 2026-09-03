# Ground-to-air mechanics research for Natural V7

Date: 2026-09-03

## Scope

This note freezes the physical interpretation used for the Natural V7
controlled curriculum. It does not define a named-mechanic detector and it does
not change the production gameplay reward. The reviewed user clip remains a
qualitative reference rather than a source of hidden state or scripted action.

## Evidence reviewed

- The user's 20.866-second Rocket League example shows a small first touch that
  converts an existing bounce into a close rising ball, followed immediately by
  a short airborne carry. It does not support training a dead resting ball into
  a large vertical launch. Clip SHA-256:
  `5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`.
- Rocket Science's measured jump explanation states that holding the first jump
  continues applying roof-direction force for up to about 200 ms, while the
  second jump is an impulse without that sustained hold force. This supports
  explicit press/release/second-jump phases rather than an instantaneous jump
  toggle: <https://rocketscience.fyi/know/videos/boost-and-jump>.
- Yukeo's air-dribble guidance emphasizes the setup touch, matching the ball's
  pace, contacting the underside, jumping immediately after a ground-bounce
  touch, and feathering boost in the air:
  <https://dignitas.gg/articles/setting-up-an-air-dribble-and-performing-them-a-guide-with-yukeo>.
- The all-setups ground-to-air tutorial distinguishes chip, carry, bounce,
  rolling-ball chip, and double-jump/tornado variants rather than treating them
  as one launch: <https://www.youtube.com/watch?v=3miaW-kwQQg>.
- The detailed level-by-level tutorial separately identifies chip/carry/bounce
  ground setups, side-angle corner chips, double-jump timing, and a tornado-spin
  catch: <https://www.youtube.com/watch?v=gz-G4gLgPwY>.
- The tornado-specific tutorial treats the tornado/reverse-tornado motion as an
  orientation tool during takeoff rather than a goal in itself:
  <https://www.youtube.com/watch?v=wA29tJRldfo>.

## Frozen mechanical interpretation

Natural V7 covers three distinct launch families:

1. **Low bounce ahead.** The ball already has vertical separation. Rival needs
   a light forward/underside touch and immediate follow, not a large pop.
2. **Incoming rolling chip.** A ball rolling toward Rival is contacted low on
   its front/underside so its existing relative momentum produces the rise.
3. **Matched dribble.** Rival and ball begin with similar ground velocity. A
   held first jump, release, and second jump create separation. Small
   steer/yaw/roll corrections may put a front corner under the ball or form a
   partial tornado-style transition; a plain double jump remains valid.

The first touch is only the launch. Success requires the car to remain behind
and below the ball, converge on its velocity, feather boost, and use a bounded
number of meaningful contacts to progress toward goal. A normal conversion is
expected in roughly one to four follow contacts; six distinct native contact
onsets is the hard episode budget. Continuous contact is one interval and
cannot farm event credit.

## V7 isolation decision

V5 showed that the fixed `pitch=0.5` jump path can convert easy launch
geometry. V6 exposed a full pitch residual and damaged that path. V7 therefore
restarts from the protected controlled scorer, preserves the exact fixed pitch,
and learns only bounded steer/yaw/roll corrections during the existing jump
sequence. It does not concurrently retune jump timing. This isolates whether
corner/tornado-style orientation helps without removing the already observed
plain-double-jump solution. A later prospective experiment may broaden
press/release timing only if V7's physical evidence shows timing is the limiting
factor.
