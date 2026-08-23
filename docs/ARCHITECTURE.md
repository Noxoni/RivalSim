# RivalSim v0.2 Architecture

## Design target and measured boundary

RivalSim is a specialized, batched Soccar 1v1 transition engine. It is not a line-by-line CUDA
translation of RocketSim or Bullet. The v0.2 proof asks only whether two cars can interact with
one immutable static Soccar arena at useful GPU throughput and RocketSim fidelity.

The implementation is complete through that boundary, but its combined verdict is
`PAUSE_RED`: B3 throughput is well above the performance threshold while contact-rich
RocketSim parity fails. This document describes the measured implementation, not a claim that
RivalSim is a complete or drop-in simulator.

## World and state model

Each environment retains the v0.1 flattened, device-resident two-car/one-free-ball state at a
fixed 120 Hz timestep. v0.2 adds one shared arena asset per GPU device and structure-of-arrays
vehicle state for eight wheels per world.

Per-wheel state includes:

- ray origin, direction, nearest hit point, normal, distance, and face;
- suspension length, relative velocity, clipped inverse-contact factor, spring force, and
  extra pushback;
- engine/brake acceleration, steer angle, and lateral/longitudinal friction terms;
- contact and static-world-contact flags.

Per-car additions include handbrake interpolation, wheels-in-contact, broadphase candidate
and accepted-contact counters/maxima, maximum accumulated penetration, and up to four contact
points, normals, and depths. Arrays are allocated once and remain on the selected Warp device.

## Shared Soccar geometry

`ArenaGeometry` reads the exact 16 external RocketSim Soccar `.cmf` files in natural numeric
order. The parser validates little-endian counts, exact file length, finite vertices, triangle
index bounds, file SHA-256, and RocketSim's internal mesh hash. It scales Bullet-space vertices
to Unreal units and concatenates the files deterministically.

The measured asset has 4,468 vertices and 8,020 triangles. No mesh byte is duplicated per
world and no extracted/repacked collision asset is committed to this repository.

`WarpArenaMeshes` creates two acceleration structures over the same immutable point/index
arrays:

- the normal Warp mesh/BVH, used for `mesh_query_aabb` chassis candidates;
- the Warp 1.16 cuBQL mesh, used for suspension rays after independent query parity and a B1
  throughput comparison.

RocketSim also adds analytical Soccar floor, ceiling, and X-side planes around the CMFs. The
CPU reference and GPU wheel/query kernels include those four shapes; a procedural arena does
not replace the triangle geometry.

## Independent geometry-query check

The CPU reference performs two-sided Möller-Trumbore intersection directly over the stored
triangles, plus the four analytical planes. Its deterministic corpus covers floor, ceiling,
side walls, ramps/curves, goals/back wall, corners, exact shared boundaries, near-surface
origins, and misses.

Normal and cuBQL Warp rays are checked against that independent result before backend
selection. Co-nearest face ties are classified explicitly so a physically identical boundary
hit is not mislabeled as a unique-face mismatch.

## Vehicle and contact pipeline

`StaticWorldSim` layers v0.2 around the frozen v0.1 transition:

1. load the current entry from a 64-action device tape, holding each entry for four ticks;
2. transform the four Octane-compatible wheel connection points for each car;
3. cast each wheel along the chassis down axis against cuBQL triangles and analytical planes;
4. compute compression/relaxation, clipped suspension force, pushback, drive/brake/coast,
   steering, lateral friction, handbrake/powerslide, and wheel impulses;
5. run the existing v0.1 airborne/jump/boost/gravity/integration transition;
6. build the true conservative world AABB of the oriented chassis box;
7. enumerate overlapping mesh triangles with the normal Warp BVH;
8. reject noncontacts with a 13-axis triangle-vs-OBB SAT and retain at most four contacts;
9. apply normal/friction/restitution impulses, positional correction, and off-center angular
   response, including analytical plane contacts;
10. advance the device tick counter.

This ordering preserves the distinction between RocketSim-derived wheel preparation, the
rigid-body integration path inherited from v0.1, and post-integration static chassis response.
The small solver intentionally approximates Bullet rather than porting its full manifold and
constraint system. The measured parity failure shows that this approximation, especially its
wheel friction/steering and surface-transition response, is not yet faithful enough.

## Benchmark decomposition

The benchmark uses the same deterministic contact-rich state distribution and action-tape
discipline while selecting progressively larger layers:

| Variant | Included work |
|---|---|
| B0 | Frozen v0.1 contact-free transition |
| B1 | Eight static-world wheel rays only; origins stay fixed |
| B2 | Rays plus wheel transforms, suspension, and ground forces, then v0.1 integration |
| B3 | Complete B2 path plus chassis broadphase, SAT narrow phase, and contact response |

CUDA graphs execute eight-tick blocks. Initialization, checkpoint restoration, verification
readback, telemetry, and synchronization around timing are charged or reported separately;
full state does not round-trip through the CPU inside the timed loop.

## Determinism and numerical mode

Published parity and performance evidence uses Warp's normal FP32 mode with no separately
advertised fast-math approximation. Repeats restore identical device checkpoints so timing CV
is not confounded by different evolving contact states. The stress gate repeats the same
64-world, 2,400-tick workload and compares full-state digests.

## Frozen v0.1 and explicit exclusions

The v0.1 airborne/contact-free mechanics and evidence remain frozen. v0.2 does not add
ball-world, car-ball, car-car, boost-pad, scoring/reset, training, or policy-inference paths.
The free ball remains only the v0.1 integration scaffold. No v0.3 work is authorized by this
architecture result.
