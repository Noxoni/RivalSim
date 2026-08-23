"""Source-backed constants for the bounded v0.1 mechanics.

The RocketSim-derived values are from ZealanL/RocketSim ``src/RLConst.h`` at
commit c2baacb8f4b441dd8505e63c2aeb5a1679b60b02.  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import numpy as np

TICK_RATE = 120
DT = np.float32(1.0 / TICK_RATE)

GRAVITY_Z = np.float32(-650.0)
CAR_MAX_SPEED = np.float32(2300.0)
CAR_MAX_ANG_SPEED = np.float32(5.5)
BALL_MAX_SPEED = np.float32(6000.0)
BALL_MAX_ANG_SPEED = np.float32(6.0)
BALL_DRAG = np.float32(0.03)

BOOST_MAX = np.float32(100.0)
BOOST_USED_PER_SECOND = np.float32(100.0 / 3.0)
BOOST_MIN_TIME = np.float32(0.1)
BOOST_ACCEL_GROUND = np.float32(2975.0 / 3.0)
BOOST_ACCEL_AIR = np.float32(3175.0 / 3.0)

THROTTLE_AIR_ACCEL = np.float32(200.0 / 3.0)

JUMP_ACCEL = np.float32(4375.0 / 3.0)
JUMP_IMMEDIATE_FORCE = np.float32(875.0 / 3.0)
JUMP_MIN_TIME = np.float32(0.025)
JUMP_RESET_TIME_PAD = np.float32(1.0 / 40.0)
JUMP_MAX_TIME = np.float32(0.2)
JUMP_PRE_MIN_ACCEL_SCALE = np.float32(0.62)
JUMP_STICKY_ACCEL = np.float32(325.0)
JUMP_STICKY_TICKS = 3
DOUBLEJUMP_MAX_DELAY = np.float32(1.25)

DODGE_DEADZONE = np.float32(0.5)
FLIP_Z_DAMP_120 = np.float32(0.35)
FLIP_Z_DAMP_START = np.float32(0.15)
FLIP_Z_DAMP_END = np.float32(0.21)
FLIP_TORQUE_TIME = np.float32(0.65)
FLIP_PITCHLOCK_EXTRA_TIME = np.float32(0.3)
FLIP_INITIAL_VEL_SCALE = np.float32(500.0)
FLIP_TORQUE_X = np.float32(260.0)
FLIP_TORQUE_Y = np.float32(224.0)
FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = np.float32(1.0)
FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = np.float32(1.9)
FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = np.float32(2.5)
FLIP_BACKWARD_IMPULSE_SCALE_X = np.float32(16.0 / 15.0)

CAR_TORQUE_SCALE = np.float32(2.0 * np.pi / (1 << 16) * 1000.0)
CAR_AIR_CONTROL_TORQUE = np.asarray((130.0, 95.0, 400.0), dtype=np.float32)
CAR_AIR_CONTROL_DAMPING = np.asarray((30.0, 20.0, 50.0), dtype=np.float32)

SUPERSONIC_START_SPEED = np.float32(2200.0)
SUPERSONIC_MAINTAIN_MIN_SPEED = np.float32(2100.0)
SUPERSONIC_MAINTAIN_MAX_TIME = np.float32(1.0)

# Bullet applies linear damping as pow(1 - damping, dt) before force integration.
BALL_DRAG_FACTOR = np.float32(np.power(np.float32(1.0) - BALL_DRAG, DT))
