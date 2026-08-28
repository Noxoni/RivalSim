from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import uuid
from pathlib import Path

import pytest

from rivalsim.human_demo.analysis import (
    action_variation_collection_report,
    action_variation_report,
    rival_observation_mapping_report,
)
from rivalsim.human_demo.format import (
    ACTION_NAMES,
    CHUNK_HEADER,
    FRAME_FLAG_DUPLICATE_PHYSICS_FRAME,
    CorruptChunkError,
    SessionWriter,
    decode_frame,
    encode_frame,
)
from rivalsim.human_demo.reader import SessionReader


def _native_input(seed: float = 0.0) -> dict[str, float | bool]:
    return {
        "throttle": 0.75 + seed,
        "steer": -0.25,
        "pitch": 0.5,
        "yaw": -0.125,
        "roll": 0.375,
        "dodge_forward": -0.625,
        "dodge_strafe": 0.875,
        "handbrake": True,
        "jump": True,
        "activate_boost": False,
        "holding_boost": True,
        "jumped": True,
    }


def _action() -> dict[str, float | bool]:
    return {
        "throttle": 0.75,
        "steer": -0.25,
        "pitch": 0.5,
        "yaw": -0.125,
        "roll": 0.375,
        "jump": True,
        "boost": True,
        "handbrake": True,
    }


def _car(stable_id: str, *, local: bool) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "player_name": "Human" if local else "Nexto",
        "player_id": 7 if local else 8,
        "team": 1 if local else 0,
        "flags": {
            "is_local_human": local,
            "is_bot": not local,
            "car_present": True,
            "demolished": False,
            "on_ground": False,
            "supersonic": True,
            "jumped": True,
            "double_jumped": False,
            "can_jump": False,
            "has_flip": True,
        },
        "availability": 0x3FF,
        "position": (1.25, -2.5, 3.75),
        "rotation": (123, -456, 789),
        "linear_velocity": (-100.5, 200.25, 300.125),
        "angular_velocity": (1.5, -2.25, 3.125),
        "boost": 42.5,
        "time_off_ground": 0.75,
        "time_on_ground": 0.0,
        "last_ball_touch_frame": 100,
        "last_ball_impact_frame": 101,
        "respawn_time_remaining": 0,
        "num_wheel_world_contacts": 0,
        "num_wheel_contacts": 1,
        "boost_component": {"active": True, "activity_time": 0.1},
        "jump_component": {"active": False, "activity_time": 0.2},
        "double_jump_component": {"active": False, "activity_time": 0.0},
        "dodge_component": {
            "active": True,
            "activity_time": 0.15,
            "direction": (0.25, -0.75, 0.0),
        },
        "flip_component": {
            "active": False,
            "activity_time": 0.0,
            "flip_time": 0.0,
            "flip_right": False,
        },
        "native_input_available": True,
        "native_input": _native_input(),
        "wheels": [
            {
                "index": 0,
                "has_contact": True,
                "has_world_contact": False,
                "contact_change_time": 12.5,
                "contact_location": (10.0, 11.0, 12.0),
                "contact_normal": (0.0, 0.0, 1.0),
                "lateral_direction": (0.0, 1.0, 0.0),
                "longitudinal_direction": (1.0, 0.0, 0.0),
                "reference_location": (9.0, 8.0, 7.0),
                "suspension_distance": 6.5,
                "spin_speed": -5.25,
            }
        ],
    }


def _frame(sequence: int, *, physics_frame: int | None = None) -> dict[str, object]:
    return {
        "sequence": sequence,
        "physics_frame": physics_frame if physics_frame is not None else 1000 + sequence,
        "replicated_physics_frame": 900 + sequence,
        "engine_physics_time": 10.0 + sequence / 120.0,
        "game_time_seconds": 20.0 + sequence / 120.0,
        "monotonic_ns": 1_000_000_000 + sequence * 8_333_333,
        "utc_ns": 2_000_000_000 + sequence * 8_333_333,
        "delta_monotonic_ns": 8_333_333 if sequence else 0,
        "delta_engine_seconds": 1.0 / 120.0 if sequence else 0.0,
        "status_flags": 0,
        "missing_physics_frames": 0,
        "native_input": _native_input(),
        "rival_action": _action(),
        "ball": {
            "availability": 0xFF,
            "position": (-1.0, 2.0, 93.15),
            "rotation": (11, 22, 33),
            "linear_velocity": (1000.0, -500.0, 250.0),
            "angular_velocity": (2.0, 3.0, 4.0),
            "gravity_z": -650.0,
            "gravity_scale": 1.0,
            "last_touch_time": 19.5,
            "last_hit_world_time": 19.25,
            "hit_team": 1,
            "current_affector_id": "Epic|human|0",
        },
        "cars": [_car("Epic|human|0", local=True), _car("Bot|Nexto|0", local=False)],
        "match": {
            "game_mode": "Soccar",
            "map": "Stadium_P",
            "match_guid": "test-match-guid",
            "seconds_elapsed": 20.0,
            "seconds_remaining": 280.0,
            "total_game_time_played": 20.0,
            "overtime_time_played": 0.0,
            "score_team_0": 1,
            "score_team_1": 2,
            "round_number": 3,
            "countdown_number": 0,
            "seconds_remaining_countdown": 0,
            "flags": {
                "paused": False,
                "overtime": False,
                "round_active": True,
                "match_ended": False,
                "ball_has_been_hit": True,
                "kickoff_or_countdown": False,
            },
            "availability": 0x3F,
        },
        "boost_pads": [
            {
                "stable_id": "pickup:1234",
                "position": (0.0, -4240.0, 70.0),
                "is_full_boost": True,
                "flags": {"active": True, "picked_up": False},
                "cooldown_quality": 2,
                "boost_type": 1,
                "boost_amount": 1.0,
                "respawn_delay": 10.0,
                "cooldown_remaining": 0.0,
            }
        ],
    }


def _manifest() -> dict[str, object]:
    return {
        "session_uuid": str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
        "session_type": "match",
        "label": "nexto_1v1",
        "opponent_label": "nexto",
        "capture_start_utc": "2026-08-28T12:00:00+00:00",
    }


def _attach_journal(
    session: Path, name: str, records: list[dict[str, object]]
) -> None:
    path = session / name
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [row for row in manifest.get("files", []) if row.get("path") != name]
    files.append(
        {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        }
    )
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_native_frame_and_action_round_trip_are_byte_exact() -> None:
    frame = _frame(0)
    encoded = encode_frame(frame)
    decoded = decode_frame(encoded)
    assert encode_frame(decoded) == encoded
    assert decoded["native_input"] == frame["native_input"]
    assert decoded["rival_action"] == frame["rival_action"]
    assert tuple(decoded["ball"]["rotation"]) == (11, 22, 33)


def test_chunk_boundaries_and_deterministic_reader(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest(), chunk_frames=2, flush_frames=1)
    for sequence in range(5):
        writer.write_frame(_frame(sequence))
    writer.stop()
    reader = SessionReader(session)
    first = [encode_frame(frame) for frame in reader.iter_frames()]
    second = [encode_frame(frame) for frame in reader.iter_frames()]
    assert first == second
    assert len(first) == 5
    assert len(reader.manifest["chunks"]) == 3
    assert [row["frame_count"] for row in reader.manifest["chunks"]] == [2, 2, 1]
    assert reader.validate().valid


def test_interrupted_session_recovers_only_complete_records(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest(), chunk_frames=10, flush_frames=1)
    for sequence in range(3):
        writer.write_frame(_frame(sequence))
    writer.interrupt_for_test()
    partial = next((session / "chunks").glob("*.partial"))
    with partial.open("r+b") as stream:
        stream.truncate(partial.stat().st_size - 10)
    reader = SessionReader(session)
    recovered = list(reader.iter_frames(recover_partial=True))
    assert [frame["sequence"] for frame in recovered] == [0, 1]
    report = reader.validate(recover_partial=True)
    assert report.container_valid
    assert not report.capture_complete
    assert not report.valid
    assert not report.clean_termination
    assert report.partial_chunks == 1


def test_corrupted_complete_chunk_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest(), chunk_frames=10)
    writer.write_frame(_frame(0))
    writer.stop()
    chunk = next((session / "chunks").glob("*.rvr"))
    with chunk.open("r+b") as stream:
        stream.seek(CHUNK_HEADER.size + 20)
        original = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(CorruptChunkError):
        list(SessionReader(session).iter_frames())
    report = SessionReader(session).validate()
    assert not report.valid
    assert not report.manifest_hashes_valid


def test_sequence_gap_duplicate_and_physics_duplicate_detection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest(), chunk_frames=10)
    writer.write_frame(_frame(0, physics_frame=100))
    duplicate = _frame(0, physics_frame=100)
    duplicate["status_flags"] = FRAME_FLAG_DUPLICATE_PHYSICS_FRAME
    writer.write_frame(duplicate)
    writer.write_frame(_frame(3, physics_frame=103))
    writer.stop()
    report = SessionReader(session).validate()
    assert not report.valid
    assert report.duplicate_sequence_count == 1
    assert report.sequence_gap_count == 1
    assert report.missing_sequence_count == 2
    assert report.duplicate_physics_frame_count == 1


@pytest.mark.parametrize("local_count", [0, 2])
def test_human_car_selection_must_be_unique(tmp_path: Path, local_count: int) -> None:
    session = tmp_path / f"session-{local_count}"
    frame = _frame(0)
    frame["cars"] = [_car(f"id-{index}", local=index < local_count) for index in range(2)]
    writer = SessionWriter(session, _manifest())
    writer.write_frame(frame)
    writer.stop()
    report = SessionReader(session).validate()
    assert not report.valid
    assert report.invalid_human_car_frames == 1


def test_manifest_hash_and_size_verification(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0))
    writer.stop()
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = SessionReader(session).validate()
    assert not report.valid
    assert not report.manifest_hashes_valid


def test_candidate_four_tick_diagnostic_reports_channel_variation() -> None:
    frames = [_frame(sequence) for sequence in range(8)]
    for sequence in range(4, 8):
        frames[sequence]["rival_action"] = dict(frames[sequence]["rival_action"])
        frames[sequence]["rival_action"]["steer"] = (sequence - 4) / 3
    frames[6]["rival_action"]["jump"] = False
    report = action_variation_report(frames, session_class="freeplay:musty")
    assert report["valid_window_count"] == 2
    assert report["all_channels_constant_window_count"] == 1
    assert report["per_channel"]["steer"]["changed_window_count"] == 1
    assert report["per_channel"]["jump"]["changed_window_count"] == 1
    assert report["decision"] is None


def test_mapping_report_accounts_for_all_182_fields() -> None:
    report = rival_observation_mapping_report()
    assert report["field_count"] == 182
    assert sum(report["counts"].values()) == 182
    assert report["counts"] == {
        "approximately_derivable": 102,
        "exact_derivable": 58,
        "exact_direct": 16,
        "unavailable": 6,
    }
    assert report["counts"]["unavailable"] > 0
    assert {row["field"] for row in report["fields"]} == {
        row["field"] for row in report["fields"]
    }
    unavailable = {row["field"] for row in report["fields"] if row["status"] == "unavailable"}
    assert "self.time_since_boosted" in unavailable
    assert "opponent.sticky_ticks" in unavailable


def test_collection_diagnostic_identifies_most_variable_session_class() -> None:
    constant = [_frame(sequence) for sequence in range(8)]
    variable = [_frame(sequence) for sequence in range(8)]
    for sequence, frame in enumerate(variable):
        frame["rival_action"] = dict(frame["rival_action"])
        frame["rival_action"]["steer"] = float(sequence % 4) / 3.0
    report = action_variation_collection_report(
        [("match:nexto", constant), ("freeplay:musty", variable)]
    )
    classes = report["classes_by_intra_window_variation"]
    assert classes[0]["session_class"] == "freeplay:musty"
    assert classes[0]["any_channel_changed_window_fraction"] == 1.0
    assert classes[1]["any_channel_changed_window_fraction"] == 0.0
    assert report["decision"] is None


def test_chunk_header_is_stable_little_endian() -> None:
    assert CHUNK_HEADER.size == 40
    assert struct.calcsize("<8sII16sQ") == 40
    assert tuple(ACTION_NAMES) == (
        "throttle",
        "steer",
        "pitch",
        "yaw",
        "roll",
        "jump",
        "boost",
        "handbrake",
    )


def test_complete_chunk_without_footer_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest(), chunk_frames=10, flush_frames=1)
    writer.write_frame(_frame(0))
    writer.interrupt_for_test()
    partial = next((session / "chunks").glob("*.partial"))
    complete = partial.with_suffix("")
    partial.rename(complete)
    with pytest.raises(CorruptChunkError, match="no footer"):
        list(SessionReader(session).iter_frames())


def test_manifest_journal_hashes_are_verified(tmp_path: Path) -> None:
    session = tmp_path / "session"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0))
    writer.stop()
    events = session / "events.jsonl"
    events.write_text('{"kind":"goal"}\n', encoding="utf-8")
    digest = hashlib.sha256(events.read_bytes()).hexdigest().upper()
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {"path": "events.jsonl", "bytes": events.stat().st_size, "sha256": digest}
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert SessionReader(session).validate().valid
    events.write_text('{"kind":"tampered"}\n', encoding="utf-8")
    report = SessionReader(session).validate()
    assert not report.valid
    assert not report.manifest_hashes_valid


def test_event_activity_after_final_frame_fails_capture_completeness(
    tmp_path: Path,
) -> None:
    session = tmp_path / "event-tail"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0, physics_frame=100))
    writer.write_frame(_frame(1, physics_frame=101))
    writer.stop()
    _attach_journal(
        session,
        "events.jsonl",
        [
            {
                "kind": "jump_onset",
                "physics_frame": 121,
                "engine_physics_time": 10.175,
                "sequence_boundary": 2,
            }
        ],
    )
    report = SessionReader(session).validate()
    assert report.container_valid
    assert not report.capture_complete
    assert not report.overall_demonstration_valid
    assert report.final_event_physics_frame == 121
    assert report.trailing_uncovered_physics_ticks == 20
    assert report.trailing_uncovered_engine_seconds == pytest.approx(1.0 / 6.0)


def test_marker_activity_after_final_frame_fails_capture_completeness(
    tmp_path: Path,
) -> None:
    session = tmp_path / "marker-tail"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0, physics_frame=100))
    writer.stop()
    _attach_journal(
        session,
        "markers.jsonl",
        [
            {
                "kind": "marker",
                "physics_frame": 160,
                "engine_physics_time": 10.5,
                "sequence_boundary": 1,
                "text": "post-capture",
            }
        ],
    )
    report = SessionReader(session).validate()
    assert report.container_valid
    assert not report.capture_completeness_valid
    assert report.final_marker_physics_frame == 160
    assert report.trailing_uncovered_physics_ticks == 60


def test_session_stop_timing_after_final_frame_fails_completeness(
    tmp_path: Path,
) -> None:
    session = tmp_path / "stop-tail"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0, physics_frame=100))
    writer.stop()
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture_stop_physics_frame"] = 120
    manifest["capture_stop_engine_time"] = 10.2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = SessionReader(session).validate()
    assert report.container_valid
    assert not report.capture_complete
    assert report.session_stop_physics_frame == 120
    assert report.trailing_uncovered_physics_ticks == 20


def test_active_capture_rate_is_separate_from_session_wall_rate(tmp_path: Path) -> None:
    session = tmp_path / "rates"
    manifest = _manifest()
    manifest["duration_seconds"] = 10.0
    manifest["observed_capture_rate_hz"] = 12.1
    writer = SessionWriter(session, manifest)
    for sequence in range(121):
        writer.write_frame(_frame(sequence, physics_frame=1000 + sequence))
    writer.stop()
    report = SessionReader(session).validate()
    assert report.container_valid
    assert report.capture_complete
    assert report.active_capture_duration == pytest.approx(1.0)
    assert report.active_capture_rate_hz == pytest.approx(121.0)
    assert report.session_wall_duration == pytest.approx(10.0)
    assert report.session_wide_capture_rate_hz == pytest.approx(12.1)


def test_unique_sequences_with_duplicate_physics_tick_are_not_ingestion_safe(
    tmp_path: Path,
) -> None:
    session = tmp_path / "duplicate-physics"
    writer = SessionWriter(session, _manifest())
    writer.write_frame(_frame(0, physics_frame=100))
    duplicate = _frame(1, physics_frame=100)
    duplicate["status_flags"] = FRAME_FLAG_DUPLICATE_PHYSICS_FRAME
    writer.write_frame(duplicate)
    writer.stop()
    report = SessionReader(session).validate()
    assert report.container_valid
    assert report.duplicate_physics_frame_count == 1
    assert not report.capture_complete
    assert not report.overall_demonstration_valid


def test_cpp_encoder_fixture_is_python_byte_stable(tmp_path: Path) -> None:
    executable = os.environ.get("RIVALREC_CPP_FIXTURE")
    if not executable:
        pytest.skip("set RIVALREC_CPP_FIXTURE after building the native format fixture")
    output = tmp_path / "cpp-frame.bin"
    subprocess.run([executable, str(output)], check=True, capture_output=True, text=True)
    encoded = output.read_bytes()
    decoded = decode_frame(encoded)
    assert decoded["sequence"] == 7
    assert decoded["physics_frame"] == 1007
    assert decoded["native_input"]["dodge_forward"] == pytest.approx(-0.625)
    assert decoded["native_input"]["holding_boost"] is True
    assert decoded["rival_action"]["boost"] is True
    assert sum(car["flags"]["is_local_human"] for car in decoded["cars"]) == 1
    assert encode_frame(decoded) == encoded


def test_plugin_source_has_no_gameplay_mutation_calls() -> None:
    source = (
        Path(__file__).parents[1]
        / "tools"
        / "rival_demo_recorder"
        / "src"
        / "rival_demo_recorder.cpp"
    ).read_text(encoding="utf-8")
    forbidden_calls = (
        "SetInput",
        "SetVehicleInput",
        "OverrideParams",
        "ExecuteUnrealCommand",
        "SetLocation",
        "SetRotation",
        "SetVelocity",
        "SetAngularVelocity",
        "SetPhysicsState",
        "Teleport",
        "Demolish",
        "SpawnBall",
        "SpawnCar",
        "ForceBoost",
        "GiveBoost",
    )
    for call in forbidden_calls:
        assert not re.search(rf"\.\s*{call}\s*\(", source), call
    assert "*static_cast<const ControllerInput*>(params)" in source
    assert "HookEventWithCaller<CarWrapper>" in source
    assert "evaluate_local_car_binding" in source
    assert 'record_event("identity_failure"' in source
    assert 'stop_session(false, "local_human_identity_lost:' in source
    assert 'record_event("local_car_rebind"' in source
    assert "state_debug_telemetry_not_action_label" in source
