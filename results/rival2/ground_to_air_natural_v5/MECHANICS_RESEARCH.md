# Natural ground-to-air mechanics and V5 curriculum

## Physical interpretation

The curriculum does not ask Rival to launch a stationary ball vertically.  It
represents three real entry primitives:

1. **Low bounce:** the ball already has a small bounce in front of the car.  A
   light forward contact creates separation, the car closes the gap while the
   ball rises, and the second contact begins the carry.
2. **Incoming rolling chip:** the ball rolls toward Rival.  Contact below the
   center redirects it upward, after which Rival must pitch/boost into the
   rising trajectory rather than wait underneath it.
3. **Matched dribble pop:** car and ball have similar ground velocity.  A held
   first jump plus a released second jump creates the upward transition.  A
   learned air-roll/yaw/pitch corner contact is allowed, including a partial
   tornado-style rotation, but no prescribed animation is rewarded.

The post-entry objective is shared: get behind and below the ball, match its
velocity, use light contacts to move it goalward, and finish within six distinct
touch onsets.  A continuous glued contact interval counts once.  Raw airtime is
worth zero.

## User reference clip

- Path at review time:
  `C:/Users/patri/Videos/Medal/Edits/MedalTVRocketLeague20260902174434464-trim-1788386200969.mp4`
- SHA-256: `5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`
- Duration: `20.866 s`; video cadence: approximately `60 Hz`.

The clip shows a low, close first chip followed immediately into the rising
ball.  The useful behavior is the continuous relationship between car and ball:
close initial separation, car below/behind the ball, forward momentum preserved,
and a short chain of light goalward touches.  It is not a dead-ball vertical
launch and does not require a large number of touches.

## External mechanics evidence

- Psyonix's official Free Play controls expose **Take Possession**, **Start
  Dribble**, **Pass Ball**, and **Launch Ball** as distinct setup primitives.
  That supports training the incoming-ball and dribble-pop families separately:
  https://www.rocketleague.com/news/using-the-new-free-play-controls-and-training-pack-refresh
- Psyonix support notes that holding jump increases jump height, holding jump
  and boost is the quickest way to gain aerial altitude, feathering boost helps
  maintain altitude, wheel contact softens ball impact, and a dodge at contact
  adds power:
  https://www.epicgames.com/help/en-US/c-Category_RocketLeague/c-RocketLeague_Gameplay/where-can-i-find-a-rocket-league-crash-course-a000084273
- Rocket Science's measured jump analysis reports sustained first-jump force for
  roughly 200 ms and no equivalent hold force on the second jump.  Once boost is
  available, pitching toward the flight path before the second jump is more
  useful than treating the second jump as the whole aerial solution:
  https://rocketscience.fyi/know/videos/boost-and-jump
- The ground-to-air tutorial reviewed for setup taxonomy explicitly separates
  chip basics, rolling-ball chip setups, and air-roll/tornado touches:
  https://www.youtube.com/watch?v=3miaW-kwQQg

## V4 evidence and V5 correction

V4 improved defended incoming-chip elevated follow from `35.9%` to `59.8%`
and productive continuation from `10.0%` to `18.6%`, but the mixed-rollout PPO
advantages were normalized globally.  Incoming-chip returns dominated the hard
low-bounce and defended matched-dribble states.  V4 was therefore not promoted.

V5 restarts from the accepted controlled scorer.  It creates one independent
rollout for every setup x defender mode x physical side.  Advantages are
normalized inside each stratum and every stratum gets exactly one optimizer
update per block.  The seventh distinct contact now receives a stronger
scenario-only failure (`-12` instead of `-5`).  The competitive V23 policies,
production gameplay reward, six-contact definition, and untouched test
discipline remain unchanged.
