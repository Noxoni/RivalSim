"""Compare one v0.2.2 case alone with the same case in its pilot chunk."""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp
from run_v022_breadth import _selected_indices

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.dfh_breadth import (
    build_breadth_catalog,
    cases_to_controls,
    cases_to_state,
    generate_breadth_cases,
)
from rivalsim.reference.rocketsim_oracle import RocketSimStaticWorldBatchOracle
from rivalsim.static_world import StaticWorldSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    all_cases = generate_breadth_cases(build_breadth_catalog(geometry))
    corpus_index = next(
        (index for index, case in enumerate(all_cases) if case.case_id == args.case_id),
        None,
    )
    if corpus_index is None:
        raise ValueError(f"unknown case: {args.case_id}")
    selection = _selected_indices(len(all_cases), args.sample_count, all_cases)
    try:
        selected_position = selection.index(corpus_index)
    except ValueError as error:
        raise ValueError("case is not present in the requested pilot selection") from error
    chunk_start = selected_position // args.chunk_size * args.chunk_size
    chunk_indices = selection[chunk_start : chunk_start + args.chunk_size]
    chunk_cases = tuple(all_cases[index] for index in chunk_indices)
    local_index = chunk_indices.index(corpus_index)
    single_cases = (all_cases[corpus_index],)

    meshes = WarpArenaMeshes(geometry, args.device)
    single_oracle, single_sim = _systems(
        single_cases, args.collision_dir, geometry, meshes, args.device
    )
    batch_oracle, batch_sim = _systems(
        chunk_cases, args.collision_dir, geometry, meshes, args.device
    )
    records = []
    for tick in range(1, args.ticks + 1):
        single_sim.step(1)
        batch_sim.step(1)
        single_oracle.step()
        batch_oracle.step()
        single_state = single_sim.snapshot()
        batch_state = batch_sim.snapshot()
        single_vehicle = single_sim.vehicle_snapshot()
        batch_vehicle = batch_sim.vehicle_snapshot()
        single_reference = single_oracle.frame()
        batch_reference = batch_oracle.frame()
        records.append(
            {
                "tick": tick,
                "rocketsim": _differences(
                    single_reference.car_pos[0],
                    batch_reference.car_pos[local_index],
                    single_reference.car_vel[0],
                    batch_reference.car_vel[local_index],
                    single_reference.car_ang_vel[0],
                    batch_reference.car_ang_vel[local_index],
                    single_reference.world_contact_normal[0],
                    batch_reference.world_contact_normal[local_index],
                    single_reference.wheel_contacts[0],
                    batch_reference.wheel_contacts[local_index],
                ),
                "rivalsim": _differences(
                    single_state.car_pos[0, 0],
                    batch_state.car_pos[local_index, 0],
                    single_state.car_vel[0, 0],
                    batch_state.car_vel[local_index, 0],
                    single_state.car_ang_vel[0, 0],
                    batch_state.car_ang_vel[local_index, 0],
                    single_vehicle.world_contact_normal[0],
                    batch_vehicle.world_contact_normal[local_index * 2],
                    single_vehicle.wheel_contact[0],
                    batch_vehicle.wheel_contact[local_index * 2],
                ),
            }
        )
    result = {
        "case_id": args.case_id,
        "corpus_index": corpus_index,
        "pilot_selected_position": selected_position,
        "chunk_range": [chunk_start, chunk_start + len(chunk_cases)],
        "chunk_local_index": local_index,
        "chunk_case_count": len(chunk_cases),
        "records": records,
        "invariant": all(
            all(value == 0.0 for value in side.values())
            for record in records
            for side in (record["rocketsim"], record["rivalsim"])
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["invariant"] else 1


def _systems(cases, collision_dir, geometry, meshes, device):
    intended = cases_to_state(cases)
    controls = cases_to_controls(cases)
    oracle = RocketSimStaticWorldBatchOracle(intended, collision_dir)
    authoritative = oracle.authoritative_snapshot()
    sim = StaticWorldSim(
        len(cases),
        collision_dir,
        variant="B3",
        device=device,
        initial=authoritative.copy(),
        geometry=geometry,
        meshes=meshes,
    )
    sim.set_controls(controls)
    oracle.set_controls(controls)
    return oracle, sim


def _differences(pos_a, pos_b, vel_a, vel_b, ang_a, ang_b, normal_a, normal_b, wheels_a, wheels_b):
    return {
        "position": float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b))),
        "linear_velocity": float(np.linalg.norm(np.asarray(vel_a) - np.asarray(vel_b))),
        "angular_velocity": float(np.linalg.norm(np.asarray(ang_a) - np.asarray(ang_b))),
        "world_contact_normal": float(
            np.linalg.norm(np.asarray(normal_a) - np.asarray(normal_b))
        ),
        "wheel_contact": float(np.count_nonzero(np.asarray(wheels_a) != np.asarray(wheels_b))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
