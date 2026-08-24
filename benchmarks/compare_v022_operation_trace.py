"""Compare one cached RivalSim diagnostic with its deep native Bullet trace."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.arena import ArenaGeometry, build_face_mesh_index
from rivalsim.kernels.vehicle import CONTACT_BREAKING_THRESHOLD

_NONSTANDARD_NAN = re.compile(r"(?<![A-Za-z])[-+]?nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--trace-case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    discovery = _load_jsonl_gz(args.trace_case_dir / "discovery.jsonl.gz")
    operations = _load_jsonl_gz(args.trace_case_dir / "operations.jsonl.gz")
    face_to_body = build_face_mesh_index(geometry)
    body_to_offset = _body_to_face_offset(geometry)
    comparisons = [
        _compare_tick(
            tick_record,
            discovery,
            operations,
            face_to_body,
            body_to_offset,
        )
        for tick_record in diagnostic["trace"]
    ]
    first_difference = next(
        (
            {
                "tick": item["tick"],
                "stage": stage,
                "details": item[stage],
            }
            for item in comparisons
            for stage in (
                "contact_added_stream",
                "gjk_iteration_stream",
                "raw_witness_stream",
                "manifold_history",
                "final_manifold",
            )
            if not item[stage]["matches"]
        ),
        None,
    )
    result = {
        "schema_version": 1,
        "case_id": diagnostic["case"]["case_id"],
        "diagnostic": str(args.diagnostic.resolve()),
        "trace_case_dir": str(args.trace_case_dir.resolve()),
        "comparison_order": [
            "contact_added_stream",
            "gjk_iteration_stream",
            "raw_witness_stream",
            "manifold_history",
            "final_manifold",
        ],
        "nonblocking_characterization": ["candidate_stream"],
        "operation_stream_matches": first_difference is None,
        "first_difference": first_difference,
        "ticks": comparisons,
        "claim_boundary": (
            "collision witness/manifold comparison only; later constraint solving and "
            "integration are not included in this classification"
        ),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(args.output.resolve())
    return 0 if first_difference is None else 1


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            # RocketSim's diagnostic ostream spells non-finite floats as
            # lowercase `nan`/`-nan`, which is not valid JSON. These values
            # occur only in unrelated wheel debug fields. Normalize them for
            # parsing and reject any non-finite value that reaches a compared
            # collision field below.
            records.append(json.loads(_NONSTANDARD_NAN.sub("NaN", line)))
    return records


def _body_to_face_offset(geometry: ArenaGeometry) -> dict[int, int]:
    body_order = {
        mesh.path: body_index
        for body_index, mesh in enumerate(
            sorted(geometry.meshes, key=lambda value: value.path.name.casefold())
        )
    }
    result: dict[int, int] = {}
    offset = 0
    for mesh in geometry.meshes:
        result[body_order[mesh.path]] = offset
        offset += mesh.triangle_count
    return result


def _global_face(record: dict[str, Any], key: str, offsets: dict[int, int]) -> int:
    return offsets[int(record["world_body_index"])] + int(record[key])


def _global_faces(record: dict[str, Any], key: str, offsets: dict[int, int]) -> list[int]:
    offset = offsets[int(record["world_body_index"])]
    return [offset + int(face) for face in record[key]]


def _first_list_difference(left: list[Any], right: list[Any]) -> dict[str, Any] | None:
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return {"index": index, "rivalsim": left_value, "rocketsim": right_value}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "rivalsim": left[index] if index < len(left) else None,
            "rocketsim": right[index] if index < len(right) else None,
        }
    return None


def _finite_array(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"non-finite compared collision value: {label}")
    return result


def _f32_bits(value: Any) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def _compare_gjk_iterations(
    gpu_transform: dict[str, Any],
    gpu_records: list[dict[str, Any]],
    native_records: list[dict[str, Any]],
    body_to_offset: dict[int, int],
) -> dict[str, Any]:
    """Compare every shared GJK/Voronoi operation as exact float32 bits.

    RocketSim records all Bullet BVH leaf probes, while the RivalSim diagnostic
    records the SAT-retained face stream.  Compare the complete iteration trace
    for every RivalSim face against the corresponding native face instead of
    treating the extra rejected native leaves as a collision discrepancy.
    """

    native_by_face = {
        _global_face(record, "face", body_to_offset): record
        for record in native_records
        if int(record["world_body_index"]) in body_to_offset
    }
    vector_fields = {
        "axis": "axis",
        "direction_a": "direction_a",
        "point_a": "p",
        "point_b": "q",
        "w": "w",
        "cached_p1": "cached_point_a",
        "cached_p2": "cached_point_b",
        "cached_v": "cached_v",
    }
    scalar_fields = {
        "delta": "delta",
        "squared_distance_before": "squared_distance_before",
    }
    compared_float32_values = 0
    float32_bit_mismatches = 0
    max_abs_error = 0.0
    compared_steps = 0
    first_difference = None

    def compare_value(
        face: int,
        iteration: int,
        field: str,
        component: int | None,
        gpu_value: Any,
        native_value: Any,
    ) -> None:
        nonlocal compared_float32_values
        nonlocal float32_bit_mismatches
        nonlocal max_abs_error
        nonlocal first_difference
        compared_float32_values += 1
        gpu_bits = _f32_bits(gpu_value)
        native_bits = _f32_bits(native_value)
        error = abs(float(np.float32(gpu_value)) - float(np.float32(native_value)))
        max_abs_error = max(max_abs_error, error)
        if gpu_bits != native_bits:
            float32_bit_mismatches += 1
            if first_difference is None:
                first_difference = {
                    "face": face,
                    "iteration": iteration,
                    "field": field,
                    "component": component,
                    "rivalsim": float(np.float32(gpu_value)),
                    "rocketsim": float(np.float32(native_value)),
                    "rivalsim_float32_bits": f"0x{gpu_bits:08X}",
                    "rocketsim_float32_bits": f"0x{native_bits:08X}",
                }

    input_mismatches_before = float32_bit_mismatches
    native_transform = next(
        (
            record
            for record in native_records
            if int(record["world_body_index"]) in body_to_offset
        ),
        None,
    )
    if native_transform is None:
        first_difference = {"reason": "missing_native_transform"}
    else:
        for component, (gpu_value, native_value) in enumerate(
            zip(
                gpu_transform["child_center_bt"],
                native_transform["transform_a_origin"],
                strict=True,
            )
        ):
            compare_value(
                -1,
                -1,
                "input_transform_a_origin_bt",
                component,
                gpu_value,
                native_value,
            )
        for component, (gpu_value, native_value) in enumerate(
            zip(
                gpu_transform["position_offset_bt"],
                native_transform["position_offset"],
                strict=True,
            )
        ):
            compare_value(
                -1,
                -1,
                "input_position_offset_bt",
                component,
                gpu_value,
                native_value,
            )
        for row, (gpu_row, native_row) in enumerate(
            zip(
                gpu_transform["basis_rows"],
                native_transform["transform_a_basis"],
                strict=True,
            )
        ):
            for column, (gpu_value, native_value) in enumerate(
                zip(gpu_row, native_row, strict=True)
            ):
                compare_value(
                    -1,
                    -1,
                    "input_transform_a_basis",
                    row * 3 + column,
                    gpu_value,
                    native_value,
                )
    input_float32_bit_mismatches = float32_bit_mismatches - input_mismatches_before

    for gpu_record in gpu_records:
        face = int(gpu_record["face"])
        native_record = native_by_face.get(face)
        if native_record is None:
            if first_difference is None:
                first_difference = {
                    "face": face,
                    "reason": "missing_native_face_trace",
                }
            continue
        gpu_steps = gpu_record["steps"]
        native_steps = native_record["steps"]
        if len(gpu_steps) != len(native_steps) and first_difference is None:
            first_difference = {
                "face": face,
                "reason": "iteration_count",
                "rivalsim": len(gpu_steps),
                "rocketsim": len(native_steps),
                "rivalsim_truncated": bool(gpu_record["truncated"]),
            }
        for gpu_step, native_step in zip(gpu_steps, native_steps, strict=False):
            compared_steps += 1
            iteration = int(gpu_step["iteration"])
            if iteration != int(native_step["iteration"]) and first_difference is None:
                first_difference = {
                    "face": face,
                    "reason": "iteration_index",
                    "rivalsim": iteration,
                    "rocketsim": int(native_step["iteration"]),
                }
            for gpu_field, native_field in vector_fields.items():
                if gpu_field not in gpu_step or native_field not in native_step:
                    continue
                for component, (gpu_value, native_value) in enumerate(
                    zip(gpu_step[gpu_field], native_step[native_field], strict=True)
                ):
                    compare_value(
                        face,
                        iteration,
                        gpu_field,
                        component,
                        gpu_value,
                        native_value,
                    )
            for gpu_field, native_field in scalar_fields.items():
                if gpu_field not in gpu_step or native_field not in native_step:
                    continue
                compare_value(
                    face,
                    iteration,
                    gpu_field,
                    None,
                    gpu_step[gpu_field],
                    native_step[native_field],
                )
            if (
                "simplex_count_after" in gpu_step
                and "simplex_count_after" in native_step
                and int(gpu_step["simplex_count_after"])
                != int(native_step["simplex_count_after"])
                and first_difference is None
            ):
                first_difference = {
                    "face": face,
                    "iteration": iteration,
                    "field": "simplex_count_after",
                    "rivalsim": int(gpu_step["simplex_count_after"]),
                    "rocketsim": int(native_step["simplex_count_after"]),
                }

    missing_native_faces = sum(
        1 for record in gpu_records if int(record["face"]) not in native_by_face
    )
    return {
        "matches": first_difference is None and float32_bit_mismatches == 0,
        "input_transform_matches": (
            native_transform is not None and input_float32_bit_mismatches == 0
        ),
        "input_transform_float32_bit_mismatches": input_float32_bit_mismatches,
        "rivalsim_face_count": len(gpu_records),
        "native_comparable_face_count": len(native_by_face),
        "missing_native_faces": missing_native_faces,
        "compared_steps": compared_steps,
        "compared_float32_values": compared_float32_values,
        "float32_bit_mismatches": float32_bit_mismatches,
        "max_abs_error": max_abs_error,
        "first_difference": first_difference,
    }


def _f32(value: float | np.floating[Any]) -> np.float32:
    return np.float32(value)


def _area_score_f32(a: list[float], b: list[float], c: list[float], d: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    cv = np.asarray(c, dtype=np.float32)
    dv = np.asarray(d, dtype=np.float32)
    ax = _f32(av[0] - bv[0])
    ay = _f32(av[1] - bv[1])
    az = _f32(av[2] - bv[2])
    bx = _f32(cv[0] - dv[0])
    by = _f32(cv[1] - dv[1])
    bz = _f32(cv[2] - dv[2])
    x = _f32(_f32(ay * bz) - _f32(az * by))
    y = _f32(_f32(az * bx) - _f32(ax * bz))
    z = _f32(_f32(ax * by) - _f32(ay * bx))
    return float(_f32(_f32(_f32(x * x) + _f32(y * y)) + _f32(z * z)))


def _replacement_scores_f32(
    candidate: dict[str, Any], cached: list[dict[str, Any]]
) -> dict[str, Any]:
    deepest = -1
    deepest_distance = _f32(candidate["distance_bt"])
    for index, point in enumerate(cached):
        distance = _f32(point["distance_bt"])
        if distance < deepest_distance:
            deepest = index
            deepest_distance = distance
    candidate_point = candidate["local_a"]
    points = [point["local_a"] for point in cached]
    scores = [0.0, 0.0, 0.0, 0.0]
    if deepest != 0:
        scores[0] = _area_score_f32(candidate_point, points[1], points[3], points[2])
    if deepest != 1:
        scores[1] = _area_score_f32(candidate_point, points[0], points[3], points[2])
    if deepest != 2:
        scores[2] = _area_score_f32(candidate_point, points[0], points[3], points[1])
    if deepest != 3:
        scores[3] = _area_score_f32(candidate_point, points[0], points[2], points[1])
    selected = 0
    maximum = scores[0]
    for index in range(1, 4):
        if scores[index] > maximum:
            selected = index
            maximum = scores[index]
    return {
        "candidate": candidate,
        "cached": cached,
        "deepest_index": deepest,
        "scores": scores,
        "selected_replacement_index": selected,
    }


def _first_manifold_area_difference(
    history_difference: dict[str, Any] | None,
    gpu_history: list[dict[str, Any]],
    gpu_added: list[dict[str, Any]],
    native_added: list[dict[str, Any]],
    body_to_offset: dict[int, int],
) -> dict[str, Any] | None:
    if history_difference is None or history_difference["index"] <= 0:
        return None
    index = int(history_difference["index"])
    previous_faces = gpu_history[index - 1]["faces"]
    if len(previous_faces) != 4:
        return None
    candidate_face = int(gpu_history[index]["candidate_face"])
    gpu_by_face = {
        int(record["face"]): {
            "face": int(record["face"]),
            "distance_bt": float(record["distance_bt"]),
            "local_a": [float(value) for value in record["local_a"]],
        }
        for record in gpu_added
    }
    native_by_face = {
        _global_face(record, "face", body_to_offset): {
            "face": _global_face(record, "face", body_to_offset),
            "distance_bt": float(record["raw_distance_bt"]),
            "local_a": [float(value) for value in record["raw_local_a"]],
        }
        for record in native_added
    }
    if candidate_face not in gpu_by_face or candidate_face not in native_by_face:
        return None
    if any(face not in gpu_by_face or face not in native_by_face for face in previous_faces):
        return None
    return {
        "previous_faces": previous_faces,
        "candidate_face": candidate_face,
        "rivalsim": _replacement_scores_f32(
            gpu_by_face[candidate_face], [gpu_by_face[face] for face in previous_faces]
        ),
        "rocketsim_recorded_witness": _replacement_scores_f32(
            native_by_face[candidate_face],
            [native_by_face[face] for face in previous_faces],
        ),
    }


def _compare_tick(
    tick_record: dict[str, Any],
    discovery: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    face_to_body: np.ndarray,
    body_to_offset: dict[int, int],
) -> dict[str, Any]:
    tick = int(tick_record["tick"])
    rival = tick_record["rivalsim"]
    gpu_candidates = [int(face) for face in rival["mesh_candidate_faces"]]
    native_traversals = sorted(
        (
            record
            for record in discovery
            if record["record"] == "bvh_traversal" and int(record["tick"]) == tick
        ),
        key=lambda record: int(record["world_body_index"]),
    )
    native_candidates = [
        face
        for traversal in native_traversals
        for face in _global_faces(traversal, "faces", body_to_offset)
    ]
    candidate_difference = _first_list_difference(gpu_candidates, native_candidates)

    gpu_sequence = rival["pre_step_manifold_sequence"]
    gpu_added = [
        record
        for record in gpu_sequence
        if bool(record["valid"])
        and float(record["distance_bt"]) * 50.0 < CONTACT_BREAKING_THRESHOLD
    ]
    native_added = sorted(
        (
            record
            for record in operations
            if record["record"] == "contact_added"
            and int(record["tick"]) == tick
            and int(record["world_body_index"]) in body_to_offset
        ),
        key=lambda record: (
            int(record["world_body_index"]),
            native_candidates.index(_global_face(record, "face", body_to_offset)),
        ),
    )
    gpu_added_faces = [int(record["face"]) for record in gpu_added]
    native_added_faces = [_global_face(record, "face", body_to_offset) for record in native_added]
    added_difference = _first_list_difference(gpu_added_faces, native_added_faces)

    native_gjk_iterations = [
        record
        for record in operations
        if record["record"] == "gjk_iterations" and int(record["tick"]) == tick
    ]
    gjk_iteration_comparison = _compare_gjk_iterations(
        rival["pre_step_body_transform"],
        rival["pre_step_pair_iterations"],
        native_gjk_iterations,
        body_to_offset,
    )

    witness_pairs = list(zip(gpu_added, native_added, strict=False))
    witness_records = []
    for index, (gpu, native) in enumerate(witness_pairs):
        gpu_face = int(gpu["face"])
        native_face = _global_face(native, "face", body_to_offset)
        if gpu_face != native_face:
            continue
        distance_error = abs(float(gpu["distance_bt"]) - float(native["raw_distance_bt"]))
        local_error = float(
            np.linalg.norm(
                _finite_array(gpu["local_a"], f"gpu local_a {gpu_face}")
                - _finite_array(native["raw_local_a"], f"native local_a {native_face}")
            )
        )
        witness_records.append(
            {
                "index": index,
                "face": gpu_face,
                "distance_bt_abs_error": distance_error,
                "local_a_l2_error_bt": local_error,
            }
        )
    max_distance_error = max(
        (record["distance_bt_abs_error"] for record in witness_records), default=None
    )
    max_local_error = max(
        (record["local_a_l2_error_bt"] for record in witness_records), default=None
    )
    witness_matches = (
        added_difference is None
        and len(witness_records) == len(gpu_added) == len(native_added)
        and max_distance_error is not None
        and max_distance_error <= 5.0e-5
        and max_local_error is not None
        and max_local_error <= 5.0e-5
    )

    native_manifolds = [
        record
        for record in operations
        if record["record"] == "manifold_after_add"
        and int(record["tick"]) == tick
        and int(record["world_body_index"]) in body_to_offset
    ]
    native_history = [
        {
            "candidate_face": _global_face(record, "candidate_face", body_to_offset),
            "faces": _global_faces(record, "faces", body_to_offset),
        }
        for record in native_manifolds
    ]
    gpu_history = [
        {
            "candidate_face": int(record["face"]),
            "faces": [int(face) for face in record["retained_faces"]],
        }
        for record in gpu_added
    ]
    history_difference = _first_list_difference(gpu_history, native_history)
    history_area_difference = _first_manifold_area_difference(
        history_difference,
        gpu_history,
        gpu_added,
        native_added,
        body_to_offset,
    )

    native_post = next(
        record
        for record in operations
        if record["record"] == "state" and int(record["tick"]) == tick and record["phase"] == "post"
    )
    gpu_contacts = [contact for contact in rival["contacts"] if int(contact["face"]) >= 0]
    native_contacts = [
        contact
        for contact in native_post["manifolds"]
        if int(contact["world_body_index"]) in body_to_offset
    ]
    gpu_final_faces = [int(contact["face"]) for contact in gpu_contacts]
    native_final_faces = [
        _global_face(contact, "index_1", body_to_offset) for contact in native_contacts
    ]
    final_difference = _first_list_difference(gpu_final_faces, native_final_faces)
    final_errors = []
    for gpu, native in zip(gpu_contacts, native_contacts, strict=False):
        gpu_face = int(gpu["face"])
        native_face = _global_face(native, "index_1", body_to_offset)
        if gpu_face != native_face:
            continue
        final_errors.append(
            {
                "face": gpu_face,
                "distance_uu_abs_error": abs(
                    float(gpu["distance"]) - float(native["distance_bt"]) * 50.0
                ),
                "local_a_l2_error_bt": float(
                    np.linalg.norm(
                        _finite_array(gpu["local_a"], f"gpu final local_a {gpu_face}")
                        - _finite_array(
                            native["stored_local_a"], f"native final local_a {native_face}"
                        )
                    )
                ),
                "normal_l2_error": float(
                    np.linalg.norm(
                        _finite_array(gpu["normal"], f"gpu final normal {gpu_face}")
                        - _finite_array(native["normal_b"], f"native final normal {native_face}")
                    )
                ),
            }
        )
    max_final_distance = max(
        (record["distance_uu_abs_error"] for record in final_errors), default=None
    )
    max_final_local = max(
        (record["local_a_l2_error_bt"] for record in final_errors), default=None
    )
    max_final_normal = max(
        (record["normal_l2_error"] for record in final_errors), default=None
    )
    final_matches = (
        final_difference is None
        and len(final_errors) == len(gpu_contacts) == len(native_contacts)
        and max_final_distance is not None
        and max_final_distance <= 5.0e-3
        and max_final_local is not None
        and max_final_local <= 5.0e-5
        and max_final_normal is not None
        and max_final_normal <= 5.0e-5
    )

    gjk_methods = Counter(
        int(record["method"])
        for record in operations
        if record["record"] == "gjk_probe" and int(record["tick"]) == tick
    )
    epa_modes = Counter(
        str(record["selected_mode"])
        for record in operations
        if record["record"] == "epa_solver_trace" and int(record["tick"]) == tick
    )
    return {
        "tick": tick,
        "state_deltas": tick_record["deltas"],
        "candidate_stream": {
            "comparable": False,
            "reason": (
                "RocketSim records every Bullet BVH leaf visit, including rejected pairs; "
                "RivalSim records only box-triangle contacts retained after its mesh query"
            ),
            "matches": candidate_difference is None,
            "rivalsim_count": len(gpu_candidates),
            "rocketsim_count": len(native_candidates),
            "first_difference": candidate_difference,
        },
        "contact_added_stream": {
            "matches": added_difference is None,
            "rivalsim_count": len(gpu_added_faces),
            "rocketsim_count": len(native_added_faces),
            "first_difference": added_difference,
        },
        "gjk_iteration_stream": gjk_iteration_comparison,
        "raw_witness_stream": {
            "matches": witness_matches,
            "compared_count": len(witness_records),
            "max_distance_bt_abs_error": max_distance_error,
            "max_local_a_l2_error_bt": max_local_error,
            "worst_distance": max(
                witness_records,
                key=lambda record: record["distance_bt_abs_error"],
                default=None,
            ),
            "worst_local_a": max(
                witness_records,
                key=lambda record: record["local_a_l2_error_bt"],
                default=None,
            ),
        },
        "manifold_history": {
            "matches": history_difference is None,
            "rivalsim_steps": len(gpu_history),
            "rocketsim_steps": len(native_history),
            "first_difference": history_difference,
            "first_difference_area_scores": history_area_difference,
        },
        "final_manifold": {
            "matches": final_matches,
            "rivalsim_faces": gpu_final_faces,
            "rocketsim_faces": native_final_faces,
            "first_difference": final_difference,
            "max_distance_uu_abs_error": max_final_distance,
            "max_local_a_l2_error_bt": max_final_local,
            "max_normal_l2_error": max_final_normal,
        },
        "native_solver_path_summary": {
            "gjk_method_counts": dict(sorted(gjk_methods.items())),
            "epa_selected_mode_counts": dict(sorted(epa_modes.items())),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
