# RocketSim native diagnostic

This source-only helper is an internal v0.2.1 fidelity oracle. It builds the
pinned `RocketSimPython` C++ sources without the Python extension and emits
JSONL vehicle, wheel, and Bullet manifold state. No collision meshes or build
products are committed.

Configure with `ROCKETSIM_BINDING_SOURCE_DIR` pointing at the exact checkout
recorded by the v0.2.1 authority package, then pass the external collision mesh
root, a supported v0.2 scenario name, and a tick count to the executable.
