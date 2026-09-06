"""Freeze inference identity and qualified packet mapping; no installation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/entity_native_packet_v1"
BRIDGE = ROOT / "results/rival2/entity_native_bridge_v1"
DEPLOY = ROOT / "deployment/entity_native_v1"
sys.path.insert(0, str(DEPLOY))
from runtime import VERSION, OBS_HASH


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((BRIDGE/"input_availability.json").read_text())
    external = Path("G:/dev/RLBot-Rival/bot/rival2_v23/models/rival2_v23_blue.json")
    assert sha(external) == audit["source_sha256"][str(external)]
    spec = json.loads(external.read_text())["observation"]
    for label in ("reference600", "finishing650"):
        source = json.loads((BRIDGE/f"{label}.json").read_text())
        assert source["contracts"]["observation_schema_sha256"] == OBS_HASH
        assert sha(ROOT/source["artifact"]["path"]) == source["artifact"]["sha256"]
        payload = dict(format=VERSION, candidate=label, artifact=source["artifact"], source=source["source"],
            contracts=dict(physics_hz=120, policy_hz=30, hold_ticks=4, observation_dim=182,
                           observation_schema_sha256=OBS_HASH, action_version="RIVAL2_ACTION_JOINT90_30HZ_V1"),
            observation=spec, fields=audit["fields"], quality_codes=dict(direct=0, derived=1, approximate=2, unavailable=3),
            qualified_proxy_policy="Existing aggregate AirState broadcast for wheel fields and sticky transition heuristic retained as unavailable proxies, not exact measurements. No masking, adapter learning or policy changes.",
            source_audit_sha256=sha(BRIDGE/"input_availability.json"),
            source_file_sha256={str(p.relative_to(ROOT).as_posix()):sha(p) for p in
                (DEPLOY/"packet_observation.py", DEPLOY/"runtime.py", ROOT/"rivalsim/entity_native_bridge.py")},
            installed=False, native_evaluation_result=None)
        path = OUT / f"{label}_manifest.json"
        assert not path.exists()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("Two separately versioned, qualified packet manifests prepared; no installation.")


if __name__ == "__main__":
    run()
