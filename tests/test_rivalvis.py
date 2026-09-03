from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
import warp as wp

pytest.importorskip("panda3d")

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_policy import deterministic_hybrid_action
from rivalsim.viewer.app import PHYSICS_SECONDS, PlaybackState, parse_args
from rivalsim.viewer.frame import interpolate_viewer_frame
from rivalsim.viewer.rendering import make_arena_collision_geometry, panda_quaternion
from rivalsim.viewer.spectator import (
    RivalVisReplay,
    RivalVisSession,
    ViewerStateAdapter,
)

CHECKPOINT = Path(
    "checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def test_checkpoint_session_uses_exact_state_and_policy_and_is_read_only(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, _geometry, _meshes = arena_assets
    before_hash = _sha256(CHECKPOINT)
    session = RivalVisSession(
        CHECKPOINT,
        collision_root=root,
        seed=20260827,
        stochastic=False,
    )
    initial_observation = session.env.observation.clone()
    actor, _value = session.model(initial_observation.reshape(-1, OBS_DIM))
    expected_action = deterministic_hybrid_action(
        actor, session.policy_config
    ).reshape(1, 2, 8)
    initial_frame = session.current_frame
    exact_position = (
        session.env.bridge.views["car_pos"].reshape(1, 2, 3).cpu().numpy()[0]
    )
    exact_quaternion = (
        session.env.bridge.views["car_quat"].reshape(1, 2, 4).cpu().numpy()[0]
    )
    for side in range(2):
        assert initial_frame.cars[side].transform.position == pytest.approx(
            exact_position[side], abs=0.0
        )
        assert initial_frame.cars[side].transform.quaternion == pytest.approx(
            exact_quaternion[side], abs=0.0
        )
    assert initial_frame.cars[0].transform.position[1] < 0.0
    assert initial_frame.cars[1].transform.position[1] > 0.0

    first_tick = session.advance_physics_tick()
    assert first_tick.physics_tick == 1
    assert first_tick.policy_decision == 0
    torch.testing.assert_close(session.current_action, expected_action, rtol=0.0, atol=0.0)
    first_decision = session.advance_policy_decision()
    assert first_decision.physics_tick == 4
    assert first_decision.policy_decision == 1
    session.close()
    assert session.checkpoint_unchanged()
    assert _sha256(CHECKPOINT) == before_hash


def test_rendered_collision_mesh_is_the_loaded_physics_geometry(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    _root, geometry, _meshes = arena_assets
    node = make_arena_collision_geometry(geometry)
    vertex_data = node.node().getGeom(0).getVertexData()
    assert vertex_data.getNumRows() == geometry.triangle_count * 3
    minimum, maximum = node.getTightBounds()
    # Panda stores the submitted CMF coordinates as float32 vertex data.
    assert tuple(minimum) == pytest.approx(geometry.bounds_min * 0.01, abs=5.0e-6)
    assert tuple(maximum) == pytest.approx(geometry.bounds_max * 0.01, abs=5.0e-6)


def test_scripted_mechanics_are_visible_in_authoritative_viewer_frames(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, geometry, meshes = arena_assets
    env = Rival2FullMatchEnv(
        1,
        root,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=0,
        car_visitation_order="a_then_b",
    )
    adapter = ViewerStateAdapter(env)
    zero = torch.zeros((1, 2, 8), device=env.device)

    def capture(action: torch.Tensor) -> object:
        return adapter.capture(
            policy_decision=env.decision_count,
            controls=action,
            rewards=torch.zeros((1, 2), device=env.device),
            last_touch=None,
            match_finished=False,
            winner=None,
        )

    initial = capture(zero)
    drive = zero.clone()
    drive[..., 0] = 1.0
    drive[..., 1] = 1.0
    drive[..., 6] = 1.0
    for _ in range(20):
        env.step(drive)
    driven = capture(drive)
    assert driven.cars[0].boost < initial.cars[0].boost
    assert driven.cars[0].speed > initial.cars[0].speed
    assert driven.cars[0].transform.quaternion != initial.cars[0].transform.quaternion

    jump = zero.clone()
    jump[..., 5] = 1.0
    env.step(jump)
    jumped = capture(jump)
    assert jumped.cars[0].transform.position[2] > driven.cars[0].transform.position[2]
    assert jumped.cars[0].has_jumped


def test_goal_updates_score_and_applies_standard_kickoff_reset(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, _geometry, _meshes = arena_assets
    session = RivalVisSession(
        CHECKPOINT,
        collision_root=root,
        seed=20260827,
        stochastic=False,
    )
    session.env.bridge.views["ball_pos"].reshape(1, 3).copy_(
        torch.tensor(((0.0, 5300.0, 93.15),), device=session.device)
    )
    wp.to_torch(session.env.world.ball_world.position_bt).reshape(1, 3).copy_(
        torch.tensor(((0.0, 106.0, 1.863),), device=session.device)
    )
    result = session.advance_policy_decision()
    assert result.blue_score == 1
    assert result.orange_score == 0
    assert not result.match_finished
    assert result.kickoff_active
    assert result.ball.transform.position == pytest.approx((0.0, 0.0, 93.15), abs=1.0e-4)
    session.close()


def test_visual_interpolation_quaternion_and_playback_controls() -> None:
    playback = PlaybackState()
    assert playback.due_ticks(PHYSICS_SECONDS * 4.1) == 4
    playback.paused = True
    assert playback.due_ticks(1.0) == 0
    playback.paused = False
    playback.slower()
    assert playback.speed == 0.5
    playback.faster()
    playback.faster()
    assert playback.speed == 2.0
    playback.normal()
    assert playback.speed == 1.0

    identity = panda_quaternion((0.0, 0.0, 0.0, 1.0))
    assert tuple(identity) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=0.0)

    # An interpolated frame changes only presentation transforms, not current HUD state.
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root:
        pytest.skip("exact local CMFs are required")
    session = RivalVisSession(CHECKPOINT, collision_root=root, stochastic=False)
    previous = session.current_frame
    session.advance_physics_tick()
    current = session.current_frame
    visual = interpolate_viewer_frame(previous, current, 0.5)
    assert visual.physics_tick == current.physics_tick
    assert visual.blue_score == current.blue_score
    session.close()


def test_buffered_replay_preserves_every_authoritative_frame(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, _geometry, _meshes = arena_assets
    live = RivalVisSession(
        CHECKPOINT,
        collision_root=root,
        seed=20260827,
        stochastic=False,
    )
    replay = RivalVisReplay.record(
        live,
        maximum_frames=9,
        progress_interval=0,
    )
    assert len(replay.frames) == 9
    assert tuple(frame.physics_tick for frame in replay.frames) == tuple(range(9))
    assert replay.current_frame.physics_tick == 0
    for expected_tick in range(1, 9):
        assert replay.advance_physics_tick().physics_tick == expected_tick
    assert replay.match_finished
    assert replay.checkpoint_unchanged()


def test_viewer_cli_accepts_frozen_wisp_opponent() -> None:
    args = parse_args(["--checkpoint", str(CHECKPOINT), "--opponent", "wisp"])
    assert args.opponent == "wisp"


def test_wisp_viewer_drives_orange_with_pinned_adapter(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, _geometry, _meshes = arena_assets
    before_hash = _sha256(CHECKPOINT)
    session = RivalVisSession(
        CHECKPOINT,
        collision_root=root,
        seed=20260827,
        stochastic=False,
        opponent="wisp",
    )
    assert session.blue_label == "RIVAL"
    assert session.orange_label == "WISP"
    assert session.wisp is not None
    session.advance_physics_tick()
    assert session.wisp.inference_calls >= 1
    torch.testing.assert_close(
        session.current_action[0, 1],
        session.wisp.previous_action[0],
        rtol=0.0,
        atol=0.0,
    )
    session.close()
    assert session.checkpoint_unchanged()
    assert _sha256(CHECKPOINT) == before_hash
