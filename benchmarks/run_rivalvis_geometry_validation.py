"""Tiny visual/physics correspondence smoke for RivalVis arena geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename, TextNode, loadPrcFileData

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.viewer.frame import ViewerFrame
from rivalsim.viewer.rendering import RivalVisScene
from rivalsim.viewer.spectator import ViewerStateAdapter, resolve_collision_root


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    ball_position: tuple[float, float, float]
    ball_velocity: tuple[float, float, float]
    camera: tuple[float, float, float]
    look_at: tuple[float, float, float]
    decisions: int
    expected: str


SCENARIOS = (
    Scenario(
        "side_wall",
        (3700.0, 0.0, 500.0),
        (1800.0, 250.0, 0.0),
        (12.0, -22.0, 13.0),
        (40.0, 0.0, 5.0),
        16,
        "x_reversal",
    ),
    Scenario(
        "curved_corner",
        (3200.0, 4200.0, 500.0),
        (1400.0, 1400.0, 0.0),
        (8.0, 19.0, 15.0),
        (35.0, 46.0, 5.0),
        22,
        "xy_reversal",
    ),
    Scenario(
        "backboard",
        (1500.0, 4750.0, 1050.0),
        (100.0, 1800.0, 250.0),
        (22.0, 31.0, 16.0),
        (15.0, 51.0, 10.0),
        18,
        "y_reversal",
    ),
    Scenario(
        "goal_recess",
        (0.0, 4700.0, 220.0),
        (0.0, 1800.0, 0.0),
        (19.0, 36.0, 11.0),
        (0.0, 54.0, 3.0),
        16,
        "blue_goal",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".tools/rivalvis/geometry-validation")
    )
    return parser.parse_args()


def _frame(
    env: Rival2FullMatchEnv,
    adapter: ViewerStateAdapter,
    action: torch.Tensor,
) -> ViewerFrame:
    winner_value = int(env.full_match_views["winner"].item())
    return adapter.capture(
        policy_decision=env.decision_count,
        controls=action,
        rewards=torch.zeros((1, 2), device=env.device),
        last_touch=None,
        match_finished=bool(env.full_match_views["match_done"].item()),
        winner=winner_value if winner_value in (0, 1) else None,
    )


def _reversed(values: list[float]) -> bool:
    return any(left > 0.0 and right < 0.0 for left, right in pairwise(values))


def run_scenario(
    scenario: Scenario,
    *,
    collision_root: Path,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> tuple[ViewerFrame, dict[str, Any]]:
    state = make_standard_kickoff_state(1, 0)
    state.ball_pos[0] = scenario.ball_position
    state.ball_vel[0] = scenario.ball_velocity
    env = Rival2FullMatchEnv(
        1,
        str(collision_root),
        device=device,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=0,
        car_visitation_order="a_then_b",
    )
    adapter = ViewerStateAdapter(env)
    action = torch.zeros((1, 2, 8), device=env.device)
    frames = [_frame(env, adapter, action)]
    score_before = (frames[0].blue_score, frames[0].orange_score)
    goal_frame: ViewerFrame | None = None
    for _ in range(scenario.decisions):
        before = frames[-1]
        env.step(action)
        after = _frame(env, adapter, action)
        if (after.blue_score, after.orange_score) != score_before and goal_frame is None:
            goal_frame = before
        frames.append(after)
    velocity_x = [frame.ball.linear_velocity[0] for frame in frames]
    velocity_y = [frame.ball.linear_velocity[1] for frame in frames]
    if scenario.expected == "x_reversal":
        passed = _reversed(velocity_x)
        selected = max(frames, key=lambda frame: frame.ball.transform.position[0])
    elif scenario.expected == "xy_reversal":
        passed = _reversed(velocity_x) or _reversed(velocity_y)
        selected = max(
            frames,
            key=lambda frame: abs(frame.ball.transform.position[0])
            + abs(frame.ball.transform.position[1]),
        )
    elif scenario.expected == "y_reversal":
        passed = _reversed(velocity_y)
        selected = max(frames, key=lambda frame: frame.ball.transform.position[1])
    else:
        passed = frames[-1].blue_score > score_before[0]
        selected = goal_frame or max(
            frames, key=lambda frame: frame.ball.transform.position[1]
        )
    result = {
        "scenario": scenario.name,
        "expected": scenario.expected,
        "passed": passed,
        "selected_ball_position_uu": list(selected.ball.transform.position),
        "initial_ball_velocity_uu_per_s": list(frames[0].ball.linear_velocity),
        "final_ball_velocity_uu_per_s": list(frames[-1].ball.linear_velocity),
        "blue_score": frames[-1].blue_score,
        "orange_score": frames[-1].orange_score,
    }
    return selected, result


def main() -> int:
    args = parse_args()
    collision_root = resolve_collision_root(args.collision_dir)
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry, args.device)
    selected: list[tuple[Scenario, ViewerFrame]] = []
    results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        frame, result = run_scenario(
            scenario,
            collision_root=collision_root,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
        )
        selected.append((scenario, frame))
        results.append(result)

    loadPrcFileData(
        "",
        "window-type offscreen\nwin-size 1200 800\nsync-video false\n"
        "show-frame-rate-meter false",
    )
    app = ShowBase()
    app.disableMouse()
    app.setBackgroundColor(0.018, 0.026, 0.045, 1.0)
    scene = RivalVisScene(app.render, geometry)
    label = TextNode("Validation scenario")
    label.setTextColor(0.95, 0.97, 1.0, 1.0)
    label.setAlign(TextNode.ACenter)
    label_path = app.aspect2d.attachNewNode(label)
    label_path.setScale(0.06)
    label_path.setPos(0.0, 0.0, 0.9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario, frame in selected:
        scene.update(frame)
        app.camera.setPos(*scenario.camera)
        app.camera.lookAt(*scenario.look_at)
        label.setText(scenario.name.replace("_", " ").upper())
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.renderFrame()
        app.win.saveScreenshot(
            Filename.fromOsSpecific(str((args.output_dir / f"{scenario.name}.png").resolve()))
        )
    app.destroy()
    summary = {
        "arena_content_sha256": geometry.content_sha256,
        "arena_vertices": geometry.vertex_count,
        "arena_triangles": geometry.triangle_count,
        "results": results,
        "verdict": "PASS_GREEN" if all(item["passed"] for item in results) else "FAIL_RED",
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
