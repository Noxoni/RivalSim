"""Compare RivalSim's reconstructed Bullet BVH ranks with the pinned native tree."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.arena import (
    ArenaGeometry,
    build_bullet_bvh_rank,
    build_face_mesh_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--diagnostic-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def native_layouts(executable: Path, collision_dir: str) -> dict[int, dict[str, Any]]:
    command = [
        str(executable.resolve()),
        str(Path(collision_dir).resolve()),
        "custom",
        "1",
        # position, velocity, angular velocity
        "0",
        "0",
        "500",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        # forward, right, up
        "1",
        "0",
        "0",
        "0",
        "1",
        "0",
        "0",
        "0",
        "1",
        # throttle, steer, handbrake
        "0",
        "0",
        "0",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )
    layouts: dict[int, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith('{"record":"bvh_layout"'):
            continue
        record = json.loads(line)
        layouts[int(record["world_body_index"])] = record
    return layouts


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    body_indices = build_face_mesh_index(geometry)
    ranks = build_bullet_bvh_rank(geometry)
    layouts = native_layouts(args.diagnostic_exe, args.collision_dir)

    meshes: list[dict[str, Any]] = []
    exact = True
    face_offset = 0
    for mesh in geometry.meshes:
        count = mesh.triangle_count
        body_index = int(body_indices[face_offset])
        record = layouts[body_index]
        native = np.asarray(record["cache_order"][:count], dtype=np.int32)
        computed = np.argsort(
            ranks[face_offset : face_offset + count], kind="stable"
        ).astype(np.int32)
        differences = np.flatnonzero(native != computed)
        match = differences.size == 0
        exact = exact and match
        first = int(differences[0]) if differences.size else None
        meshes.append(
            {
                "mesh_file": mesh.path.name,
                "body_index": body_index,
                "triangles": count,
                "exact": match,
                "different_positions": int(differences.size),
                "first_difference": first,
                "native_face": None if first is None else int(native[first]),
                "computed_face": None if first is None else int(computed[first]),
                "native_bvh_min": record["bvh_min"],
                "native_bvh_max": record["bvh_max"],
                "native_quantization": record["quantization"],
            }
        )
        face_offset += count

    result = {
        "schema": "rivalsim-bullet-bvh-rank-parity-v1",
        "exact": exact,
        "geometry_sha256": geometry.content_sha256,
        "mesh_count": len(meshes),
        "meshes": meshes,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(args.output.resolve())
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
