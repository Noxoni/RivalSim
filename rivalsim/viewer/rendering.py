"""Small procedural Panda3D scene used by RivalVis."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LColor,
    LineSegs,
    LQuaternionf,
    NodePath,
)

from rivalsim.arena import SOCCAR_EXTENT_X, SOCCAR_HEIGHT, ArenaGeometry
from rivalsim.viewer.frame import Quaternion, ViewerFrame

WORLD_SCALE = 0.01
BLUE = (0.08, 0.38, 1.0, 1.0)
ORANGE = (1.0, 0.34, 0.05, 1.0)


def panda_quaternion(value: Quaternion) -> LQuaternionf:
    """Convert RivalSim x/y/z/w without deriving orientation from velocity."""

    x, y, z, w = value
    return LQuaternionf(w, x, y, z)


def _geom_node(
    name: str,
    vertices: Iterable[tuple[tuple[float, float, float], tuple[float, float, float]]],
    indices: Iterable[tuple[int, int, int]],
    color: tuple[float, float, float, float],
) -> NodePath:
    data = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
    vertex = GeomVertexWriter(data, "vertex")
    normal = GeomVertexWriter(data, "normal")
    colors = GeomVertexWriter(data, "color")
    for position, direction in vertices:
        vertex.addData3(*position)
        normal.addData3(*direction)
        colors.addData4(*color)
    triangles = GeomTriangles(Geom.UHStatic)
    for a, b, c in indices:
        triangles.addVertices(a, b, c)
    geometry = Geom(data)
    geometry.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geometry)
    return NodePath(node)


def make_box(
    name: str,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> NodePath:
    hx, hy, hz = half_extents
    faces = (
        (((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)), (1, 0, 0)),
        (((-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)), (-1, 0, 0)),
        (((hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)), (0, 1, 0)),
        (((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)), (0, -1, 0)),
        (((hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (-hx, -hy, hz)), (0, 0, 1)),
        (((hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz)), (0, 0, -1)),
    )
    vertices: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    indices: list[tuple[int, int, int]] = []
    for positions, normal in faces:
        base = len(vertices)
        vertices.extend((position, normal) for position in positions)
        indices.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return _geom_node(name, vertices, indices, color)


def make_sphere(
    name: str,
    radius: float,
    color: tuple[float, float, float, float],
    *,
    rings: int = 8,
    segments: int = 12,
) -> NodePath:
    vertices: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for ring in range(rings + 1):
        latitude = -0.5 * math.pi + math.pi * ring / rings
        z = math.sin(latitude)
        radial = math.cos(latitude)
        for segment in range(segments + 1):
            longitude = 2.0 * math.pi * segment / segments
            normal = (radial * math.cos(longitude), radial * math.sin(longitude), z)
            vertices.append((tuple(radius * value for value in normal), normal))
    indices: list[tuple[int, int, int]] = []
    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            lower = ring * stride + segment
            upper = lower + stride
            indices.extend(((lower, upper, upper + 1), (lower, upper + 1, lower + 1)))
    return _geom_node(name, vertices, indices, color)


def _make_lines(
    name: str,
    segments: Iterable[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    color: tuple[float, float, float, float],
    thickness: float = 2.0,
) -> NodePath:
    lines = LineSegs(name)
    lines.setColor(*color)
    lines.setThickness(thickness)
    for start, end in segments:
        lines.moveTo(*start)
        lines.drawTo(*end)
    return NodePath(lines.create())


def _surface_color(
    center_uu: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, float, float, float]:
    """Presentation-only coloring over the source collision triangles."""

    arena_height = float(SOCCAR_HEIGHT)
    if abs(float(normal[2])) > 0.72 and float(center_uu[2]) < 0.35 * arena_height:
        return (0.055, 0.21, 0.145, 1.0)
    if float(center_uu[2]) > 0.72 * arena_height:
        return (0.12, 0.17, 0.24, 1.0)
    return (0.10, 0.16, 0.23, 1.0)


def make_arena_collision_geometry(geometry: ArenaGeometry) -> NodePath:
    """Convert the exact loaded Soccar CMF triangles into flat-shaded geometry."""

    data = GeomVertexData(
        "Authoritative Soccar collision geometry",
        GeomVertexFormat.getV3n3c4(),
        Geom.UHStatic,
    )
    vertex = GeomVertexWriter(data, "vertex")
    normal_writer = GeomVertexWriter(data, "normal")
    colors = GeomVertexWriter(data, "color")
    triangles = GeomTriangles(Geom.UHStatic)
    emitted = 0
    for mesh in geometry.meshes:
        vertices = mesh.vertices_uu
        for face in mesh.triangles:
            points = vertices[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            length = float(np.linalg.norm(normal))
            if length <= 1.0e-8:
                continue
            normal /= np.float32(length)
            color = _surface_color(points.mean(axis=0), normal)
            base = emitted
            for point in points:
                vertex.addData3(*(float(value) * WORLD_SCALE for value in point))
                normal_writer.addData3(*(float(value) for value in normal))
                colors.addData4(*color)
                emitted += 1
            triangles.addVertices(base, base + 1, base + 2)
    primitive = Geom(data)
    primitive.addPrimitive(triangles)
    node = GeomNode("Authoritative Soccar collision geometry")
    node.addGeom(primitive)
    return NodePath(node)


class RivalVisScene:
    """Procedural field and render nodes; no simulator behavior lives here."""

    def __init__(self, render: NodePath, arena_geometry: ArenaGeometry):
        self.render = render
        self.root = render.attachNewNode("rivalvis")
        self._build_lighting()
        self._build_field(arena_geometry)
        self.car_roots = (
            self._build_car("Blue car", BLUE),
            self._build_car("Orange car", ORANGE),
        )
        self.ball_root = self.root.attachNewNode("Ball")
        make_sphere("Ball mesh", 92.75 * WORLD_SCALE, (0.86, 0.88, 0.91, 1.0)).reparentTo(
            self.ball_root
        )
        marker = make_box(
            "Ball orientation marker", (0.7, 0.055, 0.055), (0.12, 0.14, 0.18, 1.0)
        )
        marker.setX(0.45)
        marker.reparentTo(self.ball_root)
        self.pad_nodes: list[NodePath] = []
        self._last_pad_count = 0

    def _build_lighting(self) -> None:
        ambient = AmbientLight("RivalVis ambient")
        ambient.setColor(LColor(0.33, 0.36, 0.42, 1.0))
        self.root.setLight(self.root.attachNewNode(ambient))
        sun = DirectionalLight("RivalVis sun")
        sun.setColor(LColor(0.85, 0.87, 0.92, 1.0))
        sun_node = self.root.attachNewNode(sun)
        sun_node.setHpr(-35.0, -58.0, 0.0)
        self.root.setLight(sun_node)

    def _build_field(self, geometry: ArenaGeometry) -> None:
        arena = make_arena_collision_geometry(geometry)
        arena.setTwoSided(True)
        arena.reparentTo(self.root)
        # RocketSim adds four analytic plane bodies around the CMFs: floor,
        # ceiling, and the two side planes. Their dimensions come from the
        # same arena constants and loaded collision bounds used by RivalSim.
        extent_x = float(SOCCAR_EXTENT_X) * WORLD_SCALE
        height = float(SOCCAR_HEIGHT) * WORLD_SCALE
        extent_y = float(
            max(abs(geometry.bounds_min[1]), abs(geometry.bounds_max[1]))
        ) * WORLD_SCALE
        floor = make_box(
            "Analytic floor plane",
            (extent_x, extent_y, 0.025),
            (0.045, 0.17, 0.115, 1.0),
        )
        floor.setZ(-0.025)
        floor.reparentTo(self.root)
        ceiling = make_box(
            "Analytic ceiling plane",
            (extent_x, extent_y, 0.025),
            (0.07, 0.10, 0.15, 0.50),
        )
        ceiling.setZ(height + 0.025)
        ceiling.setTransparency(True)
        ceiling.reparentTo(self.root)
        for side in (-1.0, 1.0):
            wall = make_box(
                "Analytic side plane",
                (0.025, extent_y, height * 0.5),
                (0.10, 0.15, 0.22, 0.76),
            )
            wall.setPos(side * (extent_x + 0.025), 0.0, height * 0.5)
            wall.setTransparency(True)
            wall.reparentTo(self.root)
        lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
            ((-extent_x, 0.0, 0.02), (extent_x, 0.0, 0.02))
        ]
        radius = 9.15
        for index in range(48):
            first = 2.0 * math.pi * index / 48
            second = 2.0 * math.pi * (index + 1) / 48
            lines.append(
                (
                    (radius * math.cos(first), radius * math.sin(first), 0.02),
                    (radius * math.cos(second), radius * math.sin(second), 0.02),
                )
            )
        _make_lines("Field markings", lines, (0.78, 0.83, 0.84, 0.72), 1.5).reparentTo(
            self.root
        )
        make_sphere("Kickoff mark", 0.22, (0.9, 0.9, 0.9, 1.0), rings=4, segments=8).reparentTo(
            self.root
        )

    def _build_car(
        self, name: str, color: tuple[float, float, float, float]
    ) -> tuple[NodePath, NodePath]:
        root = self.root.attachNewNode(name)
        body = make_box("Octane-sized body", (0.602, 0.433, 0.193), color)
        body.setPos(0.139, 0.0, 0.208)
        body.reparentTo(root)
        cabin_color = (*tuple(min(1.0, c + 0.16) for c in color[:3]), 1.0)
        cabin = make_box("Cabin", (0.25, 0.34, 0.15), cabin_color)
        cabin.setPos(-0.05, 0.0, 0.49)
        cabin.reparentTo(root)
        nose = make_box("Forward marker", (0.14, 0.31, 0.09), (0.93, 0.95, 1.0, 1.0))
        nose.setPos(0.66, 0.0, 0.25)
        nose.reparentTo(root)
        plume = make_sphere("Boost plume", 0.18, (1.0, 0.68, 0.08, 0.88), rings=4, segments=8)
        plume.setScale(2.0, 0.72, 0.72)
        plume.setPos(-0.68, 0.0, 0.22)
        plume.setTransparency(True)
        plume.reparentTo(root)
        plume.hide()
        return root, plume

    def _ensure_pads(self, frame: ViewerFrame) -> None:
        if self._last_pad_count == len(frame.boost_pads):
            return
        for node in self.pad_nodes:
            node.removeNode()
        self.pad_nodes.clear()
        for index, pad in enumerate(frame.boost_pads):
            radius = 0.34 if pad.is_large else 0.18
            node = make_sphere(
                f"Boost pad {index}", radius, (1.0, 0.68, 0.08, 0.88), rings=4, segments=8
            )
            node.setScale(1.0, 1.0, 0.28)
            node.setTransparency(True)
            node.setPos(*(value * WORLD_SCALE for value in pad.position))
            node.reparentTo(self.root)
            self.pad_nodes.append(node)
        self._last_pad_count = len(frame.boost_pads)

    def update(self, frame: ViewerFrame) -> None:
        self._ensure_pads(frame)
        self.ball_root.setPos(*(value * WORLD_SCALE for value in frame.ball.transform.position))
        self.ball_root.setQuat(panda_quaternion(frame.ball.transform.quaternion))
        for (root, plume), car in zip(self.car_roots, frame.cars, strict=True):
            root.setPos(*(value * WORLD_SCALE for value in car.transform.position))
            root.setQuat(panda_quaternion(car.transform.quaternion))
            root.show() if not car.is_demoed else root.hide()
            if not car.is_demoed and car.controls[6] >= 0.5 and car.boost > 0.0:
                plume.show()
            else:
                plume.hide()
        for node, pad in zip(self.pad_nodes, frame.boost_pads, strict=True):
            node.show() if pad.active else node.hide()


__all__ = [
    "BLUE",
    "ORANGE",
    "WORLD_SCALE",
    "RivalVisScene",
    "make_arena_collision_geometry",
    "make_box",
    "make_sphere",
    "panda_quaternion",
]
