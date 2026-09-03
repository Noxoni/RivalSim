"""Interactive Panda3D application for RivalVis."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import ClockObject, Filename, TextNode, Vec3, loadPrcFileData

from rivalsim.viewer.frame import ViewerFrame, interpolate_viewer_frame, quaternion_forward
from rivalsim.viewer.rendering import WORLD_SCALE, RivalVisScene
from rivalsim.viewer.spectator import RivalVisSession

PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
PHYSICS_SECONDS = 1.0 / 120.0


@dataclass(slots=True)
class PlaybackState:
    speed: float = 1.0
    paused: bool = False
    accumulator: float = 0.0
    last_overrun_ticks: int = 0
    total_overrun_ticks: int = 0

    def __post_init__(self) -> None:
        if self.speed not in PLAYBACK_SPEEDS:
            raise ValueError(f"speed must be one of {PLAYBACK_SPEEDS}")

    def slower(self) -> None:
        index = PLAYBACK_SPEEDS.index(self.speed)
        self.speed = PLAYBACK_SPEEDS[max(0, index - 1)]

    def faster(self) -> None:
        index = PLAYBACK_SPEEDS.index(self.speed)
        self.speed = PLAYBACK_SPEEDS[min(len(PLAYBACK_SPEEDS) - 1, index + 1)]

    def normal(self) -> None:
        self.speed = 1.0

    def due_ticks(self, wall_seconds: float) -> int:
        if self.paused:
            self.last_overrun_ticks = 0
            return 0
        self.accumulator += max(0.0, min(0.25, wall_seconds)) * self.speed
        due = int(self.accumulator / PHYSICS_SECONDS)

        # A one-world GPU simulation can occasionally take longer than one
        # 120 Hz interval, particularly on a frame that also evaluates an
        # opponent policy.  Carrying that wall-clock debt forward creates a
        # feedback loop: the next render attempts several ticks, takes even
        # longer, and asks for an ever larger catch-up batch.  The renderer
        # then visibly jumps over authoritative states.
        #
        # Smooth playback deliberately advances at most one consecutive
        # physics tick per rendered frame at normal speed.  Faster playback is
        # an explicit request to permit a correspondingly larger batch.  When
        # production falls behind, discard only the impossible wall-clock
        # debt; no simulator tick or policy decision is synthesized or
        # skipped.  Playback therefore slows gracefully instead of entering a
        # catch-up spiral.
        tick_budget = max(1, math.ceil(self.speed))
        ticks = min(due, tick_budget)
        self.last_overrun_ticks = max(0, due - ticks)
        self.total_overrun_ticks += self.last_overrun_ticks
        if self.last_overrun_ticks:
            self.accumulator -= due * PHYSICS_SECONDS
        else:
            self.accumulator -= ticks * PHYSICS_SECONDS
        return ticks

    @property
    def interpolation_alpha(self) -> float:
        if self.paused:
            return 1.0
        return min(1.0, max(0.0, self.accumulator / PHYSICS_SECONDS))


def _clock(frame: ViewerFrame) -> str:
    if frame.overtime:
        return "OT"
    seconds = max(0, math.ceil(frame.regulation_ticks_remaining / 120.0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _boolean(value: bool, true: str, false: str = "no") -> str:
    return true if value else false


def _car_hud(side: str, frame: ViewerFrame, index: int) -> str:
    car = frame.cars[index]
    controls = car.controls
    jump = "jumping" if car.is_jumping else ("used" if car.has_jumped else "ready")
    flip = "active" if car.is_flipping else ("used" if car.has_flipped else "idle")
    return (
        f"{side}\n"
        f"Boost {car.boost:6.1f}   Speed {car.speed:7.1f} uu/s\n"
        f"{_boolean(car.on_ground, 'grounded', 'airborne')}  wheels {car.wheel_contacts}/4  "
        f"supersonic {_boolean(car.is_supersonic, 'yes')}\n"
        f"Jump {jump}  double-jumped {_boolean(car.has_double_jumped, 'yes')}  "
        f"dodge {_boolean(car.dodge_available, 'ready', 'unavailable')}  flip {flip}\n"
        f"Ball distance {car.distance_to_ball:7.1f} uu  touches {car.touches}  "
        f"last reward {car.reward:+.4f}\n"
        f"thr {controls[0]:+5.2f}  steer {controls[1]:+5.2f}  pitch {controls[2]:+5.2f}\n"
        f"yaw {controls[3]:+5.2f}  roll {controls[4]:+5.2f}  "
        f"jump {int(controls[5] >= 0.5)}  boost {int(controls[6] >= 0.5)}  "
        f"handbrake {int(controls[7] >= 0.5)}"
    )


class RivalVisApp(ShowBase):
    """Small checkpoint spectator; it never attaches to the trainer."""

    def __init__(
        self,
        session: RivalVisSession,
        *,
        playback_speed: float = 1.0,
        smoke_frames: int = 0,
        screenshot: Path | None = None,
    ):
        super().__init__()
        self.session = session
        self.playback = PlaybackState(playback_speed)
        self.smoke_frames = max(0, int(smoke_frames))
        self.screenshot = screenshot.resolve() if screenshot else None
        self.rendered_frames = 0
        self.disableMouse()
        self.camLens.setFov(72.0)
        self.camLens.setNearFar(0.05, 1000.0)
        self.setBackgroundColor(0.018, 0.026, 0.045, 1.0)
        self.scene = RivalVisScene(
            self.render, session.env.world.meshes.geometry
        )
        self.camera_mode = 3
        self.follow_distance = 16.0
        self.follow_height = 6.2
        self.free_position = Vec3(0.0, -72.0, 38.0)
        self.free_yaw = 0.0
        self.free_pitch = -0.32
        self.free_speed = 24.0
        self.orbit_yaw = 0.0
        self.orbit_pitch = 0.0
        self.mouse_look = False
        self.last_mouse: tuple[float, float] | None = None
        self.keys: dict[str, bool] = {
            name: False for name in ("w", "a", "s", "d", "q", "e")
        }
        self._build_hud()
        self._bind_controls()
        self.scene.update(session.current_frame)
        self._update_hud(session.current_frame)
        self.taskMgr.add(self._update, "RivalVis authoritative playback")

    def _build_hud(self) -> None:
        self.match_text = OnscreenText(
            parent=self.a2dTopCenter,
            text="",
            pos=(0.0, -0.08),
            scale=0.052,
            fg=(0.94, 0.96, 1.0, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
            shadow=(0.0, 0.0, 0.0, 0.8),
        )
        self.blue_text = OnscreenText(
            parent=self.a2dTopLeft,
            text="",
            pos=(0.04, -0.12),
            scale=0.034,
            fg=(0.42, 0.68, 1.0, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
            shadow=(0.0, 0.0, 0.0, 0.82),
        )
        self.orange_text = OnscreenText(
            parent=self.a2dTopRight,
            text="",
            pos=(-0.04, -0.12),
            scale=0.034,
            fg=(1.0, 0.62, 0.30, 1.0),
            align=TextNode.ARight,
            mayChange=True,
            shadow=(0.0, 0.0, 0.0, 0.82),
        )
        self.help_text = OnscreenText(
            parent=self.a2dBottomCenter,
            text=(
                "0 free  1 follow Blue  2 follow Orange  3 director  |  RMB look  wheel zoom  "
                "|  Space/P pause  . decision  , tick  -/+ speed  = 1x  |  R restart  N new seed"
            ),
            pos=(0.0, 0.055),
            scale=0.029,
            fg=(0.86, 0.89, 0.94, 1.0),
            align=TextNode.ACenter,
            shadow=(0.0, 0.0, 0.0, 0.85),
        )
        self.final_text = OnscreenText(
            parent=self.aspect2d,
            text="",
            pos=(0.0, 0.08),
            scale=0.09,
            fg=(1.0, 0.93, 0.58, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
            shadow=(0.0, 0.0, 0.0, 0.9),
        )

    def _bind_controls(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("space", self._toggle_pause)
        self.accept("p", self._toggle_pause)
        self.accept(".", self._step_decision)
        self.accept(",", self._step_tick)
        self.accept("-", self.playback.slower)
        self.accept("+", self.playback.faster)
        self.accept("=", self.playback.normal)
        self.accept("0", self._set_camera, [0])
        self.accept("1", self._set_camera, [1])
        self.accept("2", self._set_camera, [2])
        self.accept("3", self._set_camera, [3])
        self.accept("r", self._restart, [False])
        self.accept("n", self._restart, [True])
        self.accept("wheel_up", self._zoom, [-1.0])
        self.accept("wheel_down", self._zoom, [1.0])
        self.accept("mouse3", self._set_mouse_look, [True])
        self.accept("mouse3-up", self._set_mouse_look, [False])
        for key in self.keys:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])

    def _set_key(self, key: str, value: bool) -> None:
        self.keys[key] = value

    def _set_camera(self, mode: int) -> None:
        self.camera_mode = mode
        self.last_mouse = None

    def _set_mouse_look(self, enabled: bool) -> None:
        self.mouse_look = enabled
        self.last_mouse = None

    def _zoom(self, direction: float) -> None:
        if self.camera_mode == 0:
            self.free_speed = min(100.0, max(5.0, self.free_speed + direction * 4.0))
        else:
            self.follow_distance = min(
                42.0, max(6.0, self.follow_distance + direction * 1.5)
            )

    def _toggle_pause(self) -> None:
        self.playback.paused = not self.playback.paused

    def _step_tick(self) -> None:
        if self.playback.paused and not self.session.match_finished:
            self.session.advance_physics_tick()
            self.playback.accumulator = 0.0

    def _step_decision(self) -> None:
        if self.playback.paused and not self.session.match_finished:
            self.session.advance_policy_decision()
            self.playback.accumulator = 0.0

    def _restart(self, new_seed: bool) -> None:
        self.session.restart(new_seed=new_seed)
        self.playback.accumulator = 0.0
        self.playback.paused = False

    def _mouse_delta(self) -> tuple[float, float]:
        if not self.mouse_look or not self.mouseWatcherNode.hasMouse():
            self.last_mouse = None
            return 0.0, 0.0
        mouse = self.mouseWatcherNode.getMouse()
        current = (float(mouse.x), float(mouse.y))
        if self.last_mouse is None:
            self.last_mouse = current
            return 0.0, 0.0
        delta = (current[0] - self.last_mouse[0], current[1] - self.last_mouse[1])
        self.last_mouse = current
        return delta

    def _update_camera(self, frame: ViewerFrame, dt: float) -> None:
        mouse_x, mouse_y = self._mouse_delta()
        if self.camera_mode == 0:
            self.free_yaw -= mouse_x * 2.2
            self.free_pitch = min(1.45, max(-1.45, self.free_pitch + mouse_y * 1.8))
            forward = Vec3(
                math.sin(self.free_yaw) * math.cos(self.free_pitch),
                math.cos(self.free_yaw) * math.cos(self.free_pitch),
                math.sin(self.free_pitch),
            )
            right = Vec3(math.cos(self.free_yaw), -math.sin(self.free_yaw), 0.0)
            move = Vec3(0.0, 0.0, 0.0)
            move += forward * (float(self.keys["w"]) - float(self.keys["s"]))
            move += right * (float(self.keys["d"]) - float(self.keys["a"]))
            move.z += float(self.keys["e"]) - float(self.keys["q"])
            if move.lengthSquared() > 0.0:
                move.normalize()
                self.free_position += move * self.free_speed * dt
            self.camera.setPos(self.free_position)
            self.camera.lookAt(self.free_position + forward)
            return
        self.orbit_yaw -= mouse_x * 2.0
        self.orbit_pitch = min(0.9, max(-0.45, self.orbit_pitch + mouse_y * 1.5))
        if self.camera_mode in (1, 2):
            car = frame.cars[self.camera_mode - 1]
            target = Vec3(*(value * WORLD_SCALE for value in car.transform.position))
            forward_np = quaternion_forward(car.transform.quaternion)
            heading = Vec3(float(forward_np[0]), float(forward_np[1]), 0.0)
            if heading.lengthSquared() < 1.0e-8:
                heading = Vec3(0.0, 1.0, 0.0)
            heading.normalize()
            angle = self.orbit_yaw
            rotated = Vec3(
                heading.x * math.cos(angle) - heading.y * math.sin(angle),
                heading.x * math.sin(angle) + heading.y * math.cos(angle),
                0.0,
            )
            desired = (
                target
                - rotated * self.follow_distance
                + Vec3(0.0, 0.0, self.follow_height + self.orbit_pitch * 8.0)
            )
            look_at = target + Vec3(0.0, 0.0, 1.6)
        else:
            ball = Vec3(*(value * WORLD_SCALE for value in frame.ball.transform.position))
            side = -1.0 if ball.y >= 0.0 else 1.0
            desired = ball + Vec3(27.0, side * 22.0, 15.0 + self.orbit_pitch * 8.0)
            horizontal = Vec3(desired.x - ball.x, desired.y - ball.y, 0.0)
            angle = self.orbit_yaw
            desired.x = ball.x + horizontal.x * math.cos(angle) - horizontal.y * math.sin(angle)
            desired.y = ball.y + horizontal.x * math.sin(angle) + horizontal.y * math.cos(angle)
            look_at = ball
        smoothing = 1.0 - math.exp(-max(0.0, dt) * 5.0)
        self.camera.setPos(self.camera.getPos() + (desired - self.camera.getPos()) * smoothing)
        self.camera.lookAt(look_at)

    def _update_hud(self, frame: ViewerFrame) -> None:
        phase = "OT" if frame.overtime else "REG"
        kickoff = "KICKOFF" if frame.kickoff_active else "OPEN PLAY"
        playback = f"{self.playback.speed:g}x"
        if self.playback.paused:
            playback += " PAUSED"
        elif self.playback.last_overrun_ticks:
            playback += " SMOOTH-LIMIT"
        touch = (
            "none"
            if frame.last_touch is None
            else ("Blue" if frame.last_touch == 0 else "Orange")
        )
        self.match_text.setText(
            f"{self.session.blue_label} {frame.blue_score}  |  {_clock(frame)}  |  "
            f"{frame.orange_score} {self.session.orange_label}\n"
            f"{phase}  {kickoff}  tick {frame.physics_tick}  decision {frame.policy_decision}  "
            f"{playback}  last touch {touch}  camera {self.camera_mode}"
        )
        self.blue_text.setText(
            _car_hud(f"{self.session.blue_label} - BLUE", frame, 0)
        )
        self.orange_text.setText(
            _car_hud(f"{self.session.orange_label} - ORANGE", frame, 1)
        )
        if frame.match_finished:
            winner = (
                f"{self.session.blue_label} WINS"
                if frame.winner == 0
                else f"{self.session.orange_label} WINS"
            )
            self.final_text.setText(
                f"FINAL  {frame.blue_score} - {frame.orange_score}\n{winner}\n"
                "R: same seed   N: new seed"
            )
        else:
            self.final_text.setText("")

    def _update(self, task: Task) -> int:
        dt = float(ClockObject.getGlobalClock().getDt())
        if self.smoke_frames:
            if not self.session.match_finished:
                self.session.advance_physics_tick()
        else:
            for _ in range(self.playback.due_ticks(dt)):
                if self.session.match_finished:
                    self.playback.paused = True
                    break
                self.session.advance_physics_tick()
        if self.playback.paused or self.smoke_frames:
            frame = self.session.current_frame
        else:
            frame = interpolate_viewer_frame(
                self.session.previous_frame,
                self.session.current_frame,
                self.playback.interpolation_alpha,
            )
        self.scene.update(frame)
        self._update_hud(self.session.current_frame)
        self._update_camera(frame, dt)
        self.rendered_frames += 1
        if self.smoke_frames and self.rendered_frames >= self.smoke_frames:
            if self.screenshot is not None:
                self.screenshot.parent.mkdir(parents=True, exist_ok=True)
                self.win.saveScreenshot(Filename.fromOsSpecific(str(self.screenshot)))
            self.userExit()
        return Task.cont

    def destroy(self) -> None:
        self.session.close()
        super().destroy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a Rival 2.0 checkpoint play a real RivalSim five-minute match."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collision-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--speed", type=float, choices=PLAYBACK_SPEEDS, default=1.0)
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--stochastic", dest="stochastic", action="store_true")
    behavior.add_argument("--deterministic", dest="stochastic", action="store_false")
    parser.set_defaults(stochastic=True)
    parser.add_argument(
        "--opponent",
        choices=("self", "wisp"),
        default="self",
        help="Orange-side opponent; default current-policy self-play",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loadPrcFileData(
        "",
        "\n".join(
            (
                (
                    "window-title RivalVis - Rival vs Wisp"
                    if args.opponent == "wisp"
                    else "window-title RivalVis - RivalSim checkpoint spectator"
                ),
                "win-size 1600 900",
                "sync-video true",
                "show-frame-rate-meter false",
                "textures-power-2 none",
            )
        ),
    )
    if args.smoke_frames:
        loadPrcFileData("", "window-type offscreen")
    session = RivalVisSession(
        args.checkpoint,
        collision_root=args.collision_dir,
        device=args.device,
        seed=args.seed,
        stochastic=args.stochastic,
        opponent=args.opponent,
    )
    info = session.checkpoint_info
    print(
        f"RivalVis checkpoint={info.path} sha256={info.sha256} "
        f"mode={'stochastic' if args.stochastic else 'deterministic'} "
        f"opponent={args.opponent} seed={args.seed}",
        flush=True,
    )
    app = RivalVisApp(
        session,
        playback_speed=args.speed,
        smoke_frames=args.smoke_frames,
        screenshot=args.screenshot,
    )
    app.run()
    if not session.checkpoint_unchanged():
        raise RuntimeError("RivalVis checkpoint changed while the read-only viewer was open")
    return 0


__all__ = ["PLAYBACK_SPEEDS", "PlaybackState", "RivalVisApp", "main", "parse_args"]
