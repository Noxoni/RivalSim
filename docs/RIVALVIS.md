# RivalVis

RivalVis is a lightweight native 3D spectator/debug tool for RivalSim. It runs one
isolated `Rival2FullMatchEnv` world using the selected checkpoint and renders a
standard five-minute Soccar match at 30 Hz policy cadence and 120 Hz physics.
Training remains headless and GPU-resident; RivalVis never attaches to or copies
state from an active training process.

## Install and launch

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[viewer]"
.\.venv\Scripts\python.exe -m rivalsim.viewer --checkpoint checkpoints\rival2\acquisition_v1\rival2_acquisition_resume.pt
```

The default is stochastic current-policy self-play, matching normal training and
held-out policy sampling. Deterministic debugging is available with:

```powershell
.\.venv\Scripts\python.exe -m rivalsim.viewer --checkpoint checkpoints\rival2\acquisition_v1\rival2_acquisition_resume.pt --deterministic --seed 20260827
```

Use `--speed 0.25`, `0.5`, `1`, `2`, or `4` to choose the initial playback speed.
Use `--collision-dir PATH` if the collision meshes are not in the normal sibling
`RLBot-Rival/bot/collision_meshes` checkout. `RIVALSIM_COLLISION_DIR` is also
accepted.

## Camera controls

- `0`: free camera. Move with `W/A/S/D`; move vertically with `Q/E`.
- `1`: smooth follow camera behind Blue.
- `2`: smooth follow camera behind Orange.
- `3`: ball/director camera. This is the default.
- Hold right mouse: look/orbit.
- Mouse wheel: adjust follow distance, or free-camera movement speed.

## Playback and match controls

- `Space` or `P`: pause/play.
- `.`: complete one 30 Hz policy decision while paused.
- `,`: advance exactly one authoritative 120 Hz physics tick while paused.
- `-` / `+`: slower/faster through 0.25x, 0.5x, 1x, 2x, and 4x.
- `=`: return to 1x.
- `R`: restart with the same seed.
- `N`: restart with the next seed.
- `Esc`: close RivalVis.

The renderer interpolates positions and quaternions between authoritative physics
frames. Interpolation never feeds back into physics, observations, actions, rewards,
or lifecycle state.

## What is shown

The arena surfaces are rendered from the exact `ArenaGeometry` CMF vertices and
triangles already loaded by RivalSim physics, plus RivalSim's existing analytic
floor, ceiling, and side-plane definitions. This exposes the real curved walls and
corners, ramps, backboards, goal recesses and surfaces, and ceiling instead of a
rectangular visual approximation. Presentation-only center markings, kickoff mark,
and boost-pad markers are layered over that geometry. The ball and clean low-poly
Octane-sized Blue and Orange cars use RivalSim's actual position and quaternion; the
viewer never derives orientation from velocity.

The HUD reports score, regulation clock/overtime, kickoff state, physics tick,
policy decision, playback speed, last toucher, boost, speed, wheel/ground state,
jump/double-jump/dodge/flip state, supersonic state, ball distance, touches, last
decision reward, and all eight controller channels. A visible exhaust plume reflects
the emitted boost control.

The match lifecycle is `RIVAL2_EPISODE_FULL_MATCH_V1`: persistent regulation score,
goal kickoff resets, five-minute regulation, and sudden-death overtime when tied.
This spectator lifecycle is intentionally separate from the short acquisition
episodes used to produce the current checkpoint.

## Architecture and checkpoint safety

The dependency direction is deliberately narrow:

```text
Rival2FullMatchEnv -> ViewerStateAdapter -> ViewerFrame -> Panda3D renderer
```

`ViewerFrame` contains only render/HUD data. Panda3D does not reach into or mutate
the physics implementation. The selected checkpoint is loaded once, hashed before
the window opens, and checked again at clean exit. RivalVis does not write a replay,
checkpoint, optimizer, policy, reward, observation, action, or simulator state file.

Supported checkpoint inputs are the normal `RIVAL2_CHECKPOINT_V1` resumable format,
a mapping containing `model` plus an optional `policy_config`, and a plain Rival 2.0
model state dictionary.

## Verification

The focused geometry/contact validation is reproducible with:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_rivalvis_geometry_validation.py
```

The 2026-08-27 validation loaded CMF content SHA-256
`2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`
(4,468 source vertices and 8,020 triangles). The renderer emitted all 8,020
source triangles, and its float32 tight bounds matched the source CMF bounds after
the existing UU-to-meter presentation scale. The four scripted contact views all
passed:

- straight side-wall/ramp contact reversed ball X velocity;
- curved-corner contact reversed both planar velocity components;
- backboard contact reversed ball Y velocity;
- the ball crossed the visible goal mouth into the recess and produced the expected
  Blue score plus standard kickoff reset.

The focused automated gate also covers source-mesh vertex/bounds identity, real
position/quaternion and controller-state capture, scripted steering/boost/jump
motion, score/kickoff lifecycle, 120 Hz tick stepping, 30 Hz decision stepping,
interpolation, and checkpoint hash immutability. It passes together with the
adjacent full-match goal-reset and regulation/overtime lifecycle tests.
