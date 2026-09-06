"""Field-complete source audit; no measured packet parity or feature replacement."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rivalsim import rival2_contracts as c

LIVE = Path("G:/dev/RLBot-Rival")
OUT = ROOT / "results/rival2/entity_native_bridge_v1"


def classify(name):
    if name.startswith("ball."):
        return "direct", "ball.physics location/velocity/angular_velocity; canonical rotation and frozen scale"
    if name.startswith("relative."):
        return "derived", "subtract recorded world positions/velocities, canonicalize and scale as Rival2TensorBridge"
    if name.startswith("boost_pad."):
        return "derived", "canonical xy pad identity; cooldown=max(0,standard_duration-packet.timer); active=(cooldown==0). Standard durations required; active flag discrepancy must be logged."
    if name.startswith("previous_action."):
        return "derived", "previous emitted joint-table control held four physics ticks; stored at decision, zero only on reset; delivery gaps qualified separately"
    if name.startswith("lifecycle."):
        return "approximate", "packet lifecycle/frame/touch/demo transitions aggregated over decision interval. Not bit-exact simulator tick/reset/event-onset accounting; missing packets degrade further."
    _, field = name.split(".", 1)
    if field.startswith("wheel_contact."):
        return "unavailable", "individual wheel not exposed in installed RLBot PlayerInfo. Old builder broadcasts AirState.OnGround; this is an explicit proxy, never an individual measurement."
    if field == "sticky_ticks":
        return "unavailable", "simulator internal sticky-force counter absent. Old builder uses jump-transition heuristic of2ticks; not measured."
    if field.startswith(("position.", "linear_velocity.", "angular_velocity.")):
        return "direct", "player.physics vector; canonical rotation and frozen scale"
    if field.startswith(("forward.", "up.")):
        return "derived", "trigonometric basis from native Euler rotation; same axis/sign convention, floating arithmetic not bit identity"
    if field in ("boost", "has_jumped", "has_double_jumped", "has_flipped", "is_supersonic", "demo_timer_remaining"):
        return "direct", {"has_flipped":"PlayerInfo.has_dodged", "demo_timer_remaining":"max(0,PlayerInfo.demolished_timeout), scaled/clamped"}.get(field, "PlayerInfo."+field)
    if field == "is_demoed":
        return "derived", "PlayerInfo.demolished_timeout>0"
    if field == "jump_available":
        return "derived", "has_jumped==0, exactly the frozen observation formula, not a new claim about every legal native jump"
    if field in ("on_ground", "is_jumping", "is_flipping"):
        return "approximate", "mapped from native AirState; corresponding simulator scalar equivalence has not been measured across wheel/contact/force transitions"
    if field == "dodge_available":
        return "approximate", "same frozen formula using has_jumped/double/dodged, aggregate grounded and reconstructed air_time_since_jump; native dodge_timeout is available but is not silently substituted"
    if field in ("jump_time", "air_time", "air_time_since_jump", "boosting_time", "time_since_boosted", "supersonic_time"):
        return "approximate", "packet frame-delta accumulation and enum/input predicates, not exposed native timer; old builder extrapolates a whole gap from endpoint predicates"
    if field == "flip_time":
        return "approximate", "PlayerInfo.dodge_elapsed when has_dodged and not grounded; direct native timer, but simulator reset/elapsed semantics across all transitions unverified"
    raise ValueError("Unclassified field: "+name)


def audit():
    reference = LIVE / "bot/rival2_v23/models/rival2_v23_blue.json"
    spec = json.loads(reference.read_text())["observation"]
    expected = dict(dimension=182, position_scale=list(c.POSITION_SCALE),
        car_linear_speed_scale=c.CAR_LINEAR_SPEED_SCALE, ball_linear_speed_scale=c.BALL_LINEAR_SPEED_SCALE,
        angular_speed_scale=c.ANGULAR_SPEED_SCALE, boost_scale=c.BOOST_SCALE,
        demo_timer_scale=c.DEMO_TIMER_SCALE, jump_time_scale=c.JUMP_TIME_SCALE,
        air_time_scale=c.AIR_TIME_SCALE, flip_time_scale=c.FLIP_TIME_SCALE,
        boosting_time_scale=c.BOOSTING_TIME_SCALE, time_since_boosted_scale=c.TIME_SINCE_BOOSTED_SCALE,
        supersonic_time_scale=c.SUPERSONIC_TIME_SCALE, sticky_tick_scale=c.STICKY_TICK_SCALE,
        episode_age_scale_ticks=c.EPISODE_AGE_SCALE_TICKS, no_touch_age_scale_ticks=c.NO_TOUCH_AGE_SCALE_TICKS,
        orange_pad_remap=list(c.ORANGE_PAD_REMAP))
    assert all(spec[k] == v for k,v in expected.items())
    from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS, BIG_PAD_COUNT, PAD_COUNT
    import numpy as np
    assert np.array_equal(spec["canonical_boost_pad_positions"], SOCCAR_PAD_POSITIONS)
    assert np.array_equal(spec["canonical_boost_pad_durations"], [10.0]*BIG_PAD_COUNT + [4.0]*(PAD_COUNT-BIG_PAD_COUNT))
    fields = [dict(index=i, field=name, classification=classify(name)[0], semantics=classify(name)[1])
              for i,name in enumerate(c.OBS_FIELD_NAMES)]
    sources = (LIVE / "bot/rival2_live/runtime.py", reference,
               LIVE / ".venv/Lib/site-packages/rlbot_flatbuffers/__init__.pyi",
               ROOT / "rivalsim/rival2_env.py", ROOT / "rivalsim/rival2_contracts.py")
    source_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in sources}
    result = dict(format="RIVAL2_ENTITY_NATIVE_INPUT_SOURCE_AUDIT_V1", fields=fields,
                  counts=dict(Counter(f["classification"] for f in fields)),
                  source_sha256=source_hashes, normalization_and_pad_metadata_match=True,
                  same182_structure_does_not_mean_same_temporal_contract=True,
                  current_model_observation="RIVAL2_OBS_V1, 30Hz decision history/events",
                  old_live_manifest="RIVAL2_OBS_V2_120HZ; never reuse its schema hash for this model",
                  official_sources=["https://wiki.rlbot.org/v5/botmaking/game-data/",
                      "https://raw.githubusercontent.com/RLBot/flatbuffers-schema/main/schema/gamedata.fbs"],
                  evidence_scope="Source-level classification of the complete installed builder and installed schema. Direct/derived denote accessible quantities, not measured full RocketLeague/RivalSim state parity. Unavailable fields' existing proxy behavior documented, not endorsed or relabeled.",
                  unverified=["actual native packet-to-trunk values and action continuity", "per-car force/timer semantics", "individual wheel state", "goal/reset phase boundaries and missed packet degradation"],
                  feature_changes=0, optimizer_steps=0, external_files_changed=False)
    target = OUT / "input_availability.json"
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(dict(fields=len(fields), counts=result["counts"], normalization_match=True)))


if __name__ == "__main__":
    audit()
