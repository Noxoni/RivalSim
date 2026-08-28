"""RIVALRL_NATIVE_DEMO_V1 binary serialization and crash-recoverable writer.

The format deliberately stores native Rocket League state and effective
ControllerInput values.  It does not store a frozen Rival observation vector.
All integers are little-endian and all game numeric values retain their native
32-bit representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Final

SCHEMA_NAME: Final = "RIVALRL_NATIVE_DEMO_V1"
SCHEMA_VERSION: Final = 1
CHUNK_MAGIC: Final = b"RIVRDMO1"
RECORD_MAGIC: Final = b"RVRC"
FRAME_RECORD: Final = 1
FOOTER_RECORD: Final = 2
CHUNK_HEADER = struct.Struct("<8sII16sQ")
RECORD_HEADER = struct.Struct("<4sB3xII")
FRAME_FIXED = struct.Struct("<QiiffQQQfII")
NATIVE_INPUT = struct.Struct("<7f5B3x")
RIVAL_ACTION = struct.Struct("<5f3B1x")
VEC3 = struct.Struct("<3f")
ROTATOR = struct.Struct("<3i")
CAR_FIXED = struct.Struct("<ibII")
CAR_DYNAMICS = struct.Struct("<3f3i2h")
COMPONENT = struct.Struct("<Bf")
WHEEL_FIXED = struct.Struct("<bBf")
MATCH_FIXED = struct.Struct("<4f5iII")
PAD_FIXED = struct.Struct("<3fBBBBfff")
FOOTER = struct.Struct("<QQ")

NATIVE_INPUT_NAMES: Final = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "dodge_forward",
    "dodge_strafe",
    "handbrake",
    "jump",
    "activate_boost",
    "holding_boost",
    "jumped",
)
ACTION_NAMES: Final = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)

FRAME_FLAG_DUPLICATE_PHYSICS_FRAME: Final = 1 << 0
FRAME_FLAG_OUT_OF_ORDER_PHYSICS_FRAME: Final = 1 << 1
FRAME_FLAG_PHYSICS_FRAME_GAP: Final = 1 << 2

CAR_FLAG_NAMES: Final = (
    "is_local_human",
    "is_bot",
    "car_present",
    "demolished",
    "on_ground",
    "supersonic",
    "jumped",
    "double_jumped",
    "can_jump",
    "has_flip",
)
MATCH_FLAG_NAMES: Final = (
    "paused",
    "overtime",
    "round_active",
    "match_ended",
    "ball_has_been_hit",
    "kickoff_or_countdown",
)
PAD_FLAG_NAMES: Final = ("active", "picked_up")


class FormatError(ValueError):
    """Raised when native recording bytes violate the versioned schema."""


class CorruptChunkError(FormatError):
    """Raised when a complete chunk has invalid framing or CRC."""


class _Encoder:
    def __init__(self) -> None:
        self.data = bytearray()

    def pack(self, spec: struct.Struct, *values: Any) -> None:
        self.data.extend(spec.pack(*values))

    def string(self, value: str) -> None:
        raw = value.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise FormatError("UTF-8 string exceeds the uint16 format limit")
        self.data.extend(struct.pack("<H", len(raw)))
        self.data.extend(raw)


class _Decoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def unpack(self, spec: struct.Struct) -> tuple[Any, ...]:
        end = self.offset + spec.size
        if end > len(self.data):
            raise FormatError("truncated frame payload")
        values = spec.unpack_from(self.data, self.offset)
        self.offset = end
        return values

    def string(self) -> str:
        (size,) = self.unpack(struct.Struct("<H"))
        end = self.offset + size
        if end > len(self.data):
            raise FormatError("truncated UTF-8 string")
        try:
            value = self.data[self.offset : end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise FormatError("invalid UTF-8 string") from error
        self.offset = end
        return value

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise FormatError(f"{len(self.data) - self.offset} trailing frame bytes")


def _flags(names: tuple[str, ...], values: dict[str, Any]) -> int:
    return sum((1 << index) for index, name in enumerate(names) if bool(values.get(name, False)))


def _flag_dict(names: tuple[str, ...], value: int) -> dict[str, bool]:
    return {name: bool(value & (1 << index)) for index, name in enumerate(names)}


def _vec3(value: Any) -> tuple[float, float, float]:
    if len(value) != 3:
        raise FormatError("expected a three-component vector")
    return float(value[0]), float(value[1]), float(value[2])


def _rotator(value: Any) -> tuple[int, int, int]:
    if len(value) != 3:
        raise FormatError("expected a three-component rotator")
    return int(value[0]), int(value[1]), int(value[2])


def _encode_input(encoder: _Encoder, value: dict[str, Any]) -> None:
    encoder.pack(
        NATIVE_INPUT,
        *(float(value[name]) for name in NATIVE_INPUT_NAMES[:7]),
        *(int(bool(value[name])) for name in NATIVE_INPUT_NAMES[7:]),
    )


def _decode_input(decoder: _Decoder) -> dict[str, float | bool]:
    values = decoder.unpack(NATIVE_INPUT)
    return {
        **{name: values[index] for index, name in enumerate(NATIVE_INPUT_NAMES[:7])},
        **{
            name: bool(values[index + 7])
            for index, name in enumerate(NATIVE_INPUT_NAMES[7:])
        },
    }


def _encode_action(encoder: _Encoder, value: dict[str, Any]) -> None:
    encoder.pack(
        RIVAL_ACTION,
        *(float(value[name]) for name in ACTION_NAMES[:5]),
        *(int(bool(value[name])) for name in ACTION_NAMES[5:]),
    )


def _decode_action(decoder: _Decoder) -> dict[str, float | bool]:
    values = decoder.unpack(RIVAL_ACTION)
    return {
        **{name: values[index] for index, name in enumerate(ACTION_NAMES[:5])},
        **{name: bool(values[index + 5]) for index, name in enumerate(ACTION_NAMES[5:])},
    }


def encode_frame(frame: dict[str, Any]) -> bytes:
    """Encode one frame without changing any supplied native numeric value."""

    out = _Encoder()
    out.pack(
        FRAME_FIXED,
        int(frame["sequence"]),
        int(frame["physics_frame"]),
        int(frame["replicated_physics_frame"]),
        float(frame["engine_physics_time"]),
        float(frame["game_time_seconds"]),
        int(frame["monotonic_ns"]),
        int(frame["utc_ns"]),
        int(frame["delta_monotonic_ns"]),
        float(frame["delta_engine_seconds"]),
        int(frame.get("status_flags", 0)),
        int(frame.get("missing_physics_frames", 0)),
    )
    _encode_input(out, frame["native_input"])
    _encode_action(out, frame["rival_action"])

    ball = frame.get("ball")
    out.data.append(int(ball is not None))
    if ball is not None:
        out.data.extend(struct.pack("<I", int(ball.get("availability", 0xFFFFFFFF))))
        out.pack(VEC3, *_vec3(ball["position"]))
        out.pack(ROTATOR, *_rotator(ball["rotation"]))
        out.pack(VEC3, *_vec3(ball["linear_velocity"]))
        out.pack(VEC3, *_vec3(ball["angular_velocity"]))
        out.data.extend(
            struct.pack(
                "<4fb",
                float(ball["gravity_z"]),
                float(ball["gravity_scale"]),
                float(ball["last_touch_time"]),
                float(ball["last_hit_world_time"]),
                int(ball["hit_team"]),
            )
        )
        out.string(str(ball.get("current_affector_id", "")))

    cars = frame.get("cars", [])
    if len(cars) > 0xFFFF:
        raise FormatError("car count exceeds uint16")
    out.data.extend(struct.pack("<H", len(cars)))
    for car in cars:
        out.string(str(car["stable_id"]))
        out.string(str(car.get("player_name", "")))
        out.pack(
            CAR_FIXED,
            int(car.get("player_id", -1)),
            int(car.get("team", -1)),
            _flags(CAR_FLAG_NAMES, car.get("flags", {})),
            int(car.get("availability", 0)),
        )
        out.pack(VEC3, *_vec3(car.get("position", (0.0, 0.0, 0.0))))
        out.pack(ROTATOR, *_rotator(car.get("rotation", (0, 0, 0))))
        out.pack(VEC3, *_vec3(car.get("linear_velocity", (0.0, 0.0, 0.0))))
        out.pack(VEC3, *_vec3(car.get("angular_velocity", (0.0, 0.0, 0.0))))
        out.pack(
            CAR_DYNAMICS,
            float(car.get("boost", 0.0)),
            float(car.get("time_off_ground", 0.0)),
            float(car.get("time_on_ground", 0.0)),
            int(car.get("last_ball_touch_frame", -1)),
            int(car.get("last_ball_impact_frame", -1)),
            int(car.get("respawn_time_remaining", 0)),
            int(car.get("num_wheel_world_contacts", 0)),
            int(car.get("num_wheel_contacts", 0)),
        )
        for name in ("boost_component", "jump_component", "double_jump_component"):
            component = car.get(name, {})
            out.pack(
                COMPONENT,
                int(bool(component.get("active", False))),
                float(component.get("activity_time", 0.0)),
            )
        dodge = car.get("dodge_component", {})
        out.pack(
            COMPONENT,
            int(bool(dodge.get("active", False))),
            float(dodge.get("activity_time", 0.0)),
        )
        out.pack(VEC3, *_vec3(dodge.get("direction", (0.0, 0.0, 0.0))))
        flip = car.get("flip_component", {})
        out.data.extend(
            struct.pack(
                "<BffB",
                int(bool(flip.get("active", False))),
                float(flip.get("activity_time", 0.0)),
                float(flip.get("flip_time", 0.0)),
                int(bool(flip.get("flip_right", False))),
            )
        )
        input_available = bool(car.get("native_input_available", False))
        out.data.append(int(input_available))
        _encode_input(out, car.get("native_input", neutral_input()))
        wheels = car.get("wheels", [])
        if len(wheels) > 255:
            raise FormatError("wheel count exceeds uint8")
        out.data.append(len(wheels))
        for wheel in wheels:
            wheel_flags = int(bool(wheel.get("has_contact", False))) | (
                int(bool(wheel.get("has_world_contact", False))) << 1
            )
            out.pack(
                WHEEL_FIXED,
                int(wheel.get("index", -1)),
                wheel_flags,
                float(wheel.get("contact_change_time", 0.0)),
            )
            for name in (
                "contact_location",
                "contact_normal",
                "lateral_direction",
                "longitudinal_direction",
                "reference_location",
            ):
                out.pack(VEC3, *_vec3(wheel.get(name, (0.0, 0.0, 0.0))))
            out.data.extend(
                struct.pack(
                    "<2f",
                    float(wheel.get("suspension_distance", 0.0)),
                    float(wheel.get("spin_speed", 0.0)),
                )
            )

    match = frame["match"]
    out.string(str(match.get("game_mode", "")))
    out.string(str(match.get("map", "")))
    out.string(str(match.get("match_guid", "")))
    out.pack(
        MATCH_FIXED,
        float(match.get("seconds_elapsed", 0.0)),
        float(match.get("seconds_remaining", 0.0)),
        float(match.get("total_game_time_played", 0.0)),
        float(match.get("overtime_time_played", 0.0)),
        int(match.get("score_team_0", 0)),
        int(match.get("score_team_1", 0)),
        int(match.get("round_number", 0)),
        int(match.get("countdown_number", 0)),
        int(match.get("seconds_remaining_countdown", 0)),
        _flags(MATCH_FLAG_NAMES, match.get("flags", {})),
        int(match.get("availability", 0)),
    )
    pads = frame.get("boost_pads", [])
    if len(pads) > 0xFFFF:
        raise FormatError("boost pad count exceeds uint16")
    out.data.extend(struct.pack("<H", len(pads)))
    for pad in pads:
        out.string(str(pad["stable_id"]))
        out.pack(
            PAD_FIXED,
            *_vec3(pad["position"]),
            int(bool(pad.get("is_full_boost", False))),
            _flags(PAD_FLAG_NAMES, pad.get("flags", {})),
            int(pad.get("cooldown_quality", 0)),
            int(pad.get("boost_type", 0)),
            float(pad.get("boost_amount", 0.0)),
            float(pad.get("respawn_delay", 0.0)),
            float(pad.get("cooldown_remaining", 0.0)),
        )
    return bytes(out.data)


def decode_frame(payload: bytes) -> dict[str, Any]:
    """Decode a frame and reject trailing or truncated bytes deterministically."""

    source = _Decoder(payload)
    fixed = source.unpack(FRAME_FIXED)
    frame: dict[str, Any] = {
        "sequence": fixed[0],
        "physics_frame": fixed[1],
        "replicated_physics_frame": fixed[2],
        "engine_physics_time": fixed[3],
        "game_time_seconds": fixed[4],
        "monotonic_ns": fixed[5],
        "utc_ns": fixed[6],
        "delta_monotonic_ns": fixed[7],
        "delta_engine_seconds": fixed[8],
        "status_flags": fixed[9],
        "missing_physics_frames": fixed[10],
        "native_input": _decode_input(source),
        "rival_action": _decode_action(source),
    }
    (ball_present,) = source.unpack(struct.Struct("<B"))
    if ball_present:
        (availability,) = source.unpack(struct.Struct("<I"))
        frame["ball"] = {
            "availability": availability,
            "position": source.unpack(VEC3),
            "rotation": source.unpack(ROTATOR),
            "linear_velocity": source.unpack(VEC3),
            "angular_velocity": source.unpack(VEC3),
        }
        ball_tail = source.unpack(struct.Struct("<4fb"))
        frame["ball"].update(
            {
                "gravity_z": ball_tail[0],
                "gravity_scale": ball_tail[1],
                "last_touch_time": ball_tail[2],
                "last_hit_world_time": ball_tail[3],
                "hit_team": ball_tail[4],
                "current_affector_id": source.string(),
            }
        )
    else:
        frame["ball"] = None

    (car_count,) = source.unpack(struct.Struct("<H"))
    frame["cars"] = []
    for _ in range(car_count):
        stable_id = source.string()
        player_name = source.string()
        car_fixed = source.unpack(CAR_FIXED)
        car: dict[str, Any] = {
            "stable_id": stable_id,
            "player_name": player_name,
            "player_id": car_fixed[0],
            "team": car_fixed[1],
            "flags": _flag_dict(CAR_FLAG_NAMES, car_fixed[2]),
            "availability": car_fixed[3],
            "position": source.unpack(VEC3),
            "rotation": source.unpack(ROTATOR),
            "linear_velocity": source.unpack(VEC3),
            "angular_velocity": source.unpack(VEC3),
        }
        dynamics = source.unpack(CAR_DYNAMICS)
        car.update(
            {
                "boost": dynamics[0],
                "time_off_ground": dynamics[1],
                "time_on_ground": dynamics[2],
                "last_ball_touch_frame": dynamics[3],
                "last_ball_impact_frame": dynamics[4],
                "respawn_time_remaining": dynamics[5],
                "num_wheel_world_contacts": dynamics[6],
                "num_wheel_contacts": dynamics[7],
            }
        )
        for name in ("boost_component", "jump_component", "double_jump_component"):
            component = source.unpack(COMPONENT)
            car[name] = {"active": bool(component[0]), "activity_time": component[1]}
        dodge = source.unpack(COMPONENT)
        car["dodge_component"] = {
            "active": bool(dodge[0]),
            "activity_time": dodge[1],
            "direction": source.unpack(VEC3),
        }
        flip = source.unpack(struct.Struct("<BffB"))
        car["flip_component"] = {
            "active": bool(flip[0]),
            "activity_time": flip[1],
            "flip_time": flip[2],
            "flip_right": bool(flip[3]),
        }
        (input_available,) = source.unpack(struct.Struct("<B"))
        car["native_input_available"] = bool(input_available)
        car["native_input"] = _decode_input(source)
        (wheel_count,) = source.unpack(struct.Struct("<B"))
        car["wheels"] = []
        for _ in range(wheel_count):
            wheel_fixed = source.unpack(WHEEL_FIXED)
            wheel = {
                "index": wheel_fixed[0],
                "has_contact": bool(wheel_fixed[1] & 1),
                "has_world_contact": bool(wheel_fixed[1] & 2),
                "contact_change_time": wheel_fixed[2],
            }
            for name in (
                "contact_location",
                "contact_normal",
                "lateral_direction",
                "longitudinal_direction",
                "reference_location",
            ):
                wheel[name] = source.unpack(VEC3)
            wheel_tail = source.unpack(struct.Struct("<2f"))
            wheel["suspension_distance"] = wheel_tail[0]
            wheel["spin_speed"] = wheel_tail[1]
            car["wheels"].append(wheel)
        frame["cars"].append(car)

    match = {
        "game_mode": source.string(),
        "map": source.string(),
        "match_guid": source.string(),
    }
    match_fixed = source.unpack(MATCH_FIXED)
    match.update(
        {
            "seconds_elapsed": match_fixed[0],
            "seconds_remaining": match_fixed[1],
            "total_game_time_played": match_fixed[2],
            "overtime_time_played": match_fixed[3],
            "score_team_0": match_fixed[4],
            "score_team_1": match_fixed[5],
            "round_number": match_fixed[6],
            "countdown_number": match_fixed[7],
            "seconds_remaining_countdown": match_fixed[8],
            "flags": _flag_dict(MATCH_FLAG_NAMES, match_fixed[9]),
            "availability": match_fixed[10],
        }
    )
    frame["match"] = match
    (pad_count,) = source.unpack(struct.Struct("<H"))
    frame["boost_pads"] = []
    for _ in range(pad_count):
        stable_id = source.string()
        pad = source.unpack(PAD_FIXED)
        frame["boost_pads"].append(
            {
                "stable_id": stable_id,
                "position": pad[:3],
                "is_full_boost": bool(pad[3]),
                "flags": _flag_dict(PAD_FLAG_NAMES, pad[4]),
                "cooldown_quality": pad[5],
                "boost_type": pad[6],
                "boost_amount": pad[7],
                "respawn_delay": pad[8],
                "cooldown_remaining": pad[9],
            }
        )
    source.finish()
    return frame


def neutral_input() -> dict[str, float | bool]:
    return {
        **{name: 0.0 for name in NATIVE_INPUT_NAMES[:7]},
        **{name: False for name in NATIVE_INPUT_NAMES[7:]},
    }


def neutral_action() -> dict[str, float | bool]:
    return {
        **{name: 0.0 for name in ACTION_NAMES[:5]},
        **{name: False for name in ACTION_NAMES[5:]},
    }


def _record(record_type: int, payload: bytes) -> bytes:
    header = RECORD_HEADER.pack(RECORD_MAGIC, record_type, len(payload), zlib.crc32(payload))
    return header + payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class ChunkInfo:
    index: int
    path: str
    frame_count: int
    first_sequence: int
    last_sequence: int
    bytes: int
    sha256: str
    complete: bool


class SessionWriter:
    """Reference writer used for format tests and offline fixture generation.

    The BakkesMod plugin implements the same framing natively.  This writer is
    intentionally not connected to gameplay or training.
    """

    def __init__(
        self,
        session_dir: Path,
        manifest: dict[str, Any],
        *,
        chunk_frames: int = 1024,
        flush_frames: int = 120,
    ) -> None:
        if chunk_frames <= 0 or flush_frames <= 0:
            raise ValueError("chunk and flush frame counts must be positive")
        self.session_dir = Path(session_dir)
        self.chunks_dir = self.session_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=False)
        self.session_id = uuid.UUID(str(manifest["session_uuid"]))
        self.manifest = {
            **manifest,
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "clean_termination": False,
            "chunks": [],
        }
        self.chunk_frames = chunk_frames
        self.flush_frames = flush_frames
        self._chunk: BinaryIO | None = None
        self._chunk_path: Path | None = None
        self._chunk_index = 0
        self._chunk_count = 0
        self._chunk_first_sequence = 0
        self._chunk_last_sequence = 0
        self._total_frames = 0
        _atomic_json(self.session_dir / "manifest.json", self.manifest)

    def _open_chunk(self, first_sequence: int) -> None:
        path = self.chunks_dir / f"{self._chunk_index:06d}.rvr.partial"
        self._chunk = path.open("xb")
        self._chunk.write(
            CHUNK_HEADER.pack(
                CHUNK_MAGIC,
                SCHEMA_VERSION,
                self._chunk_index,
                self.session_id.bytes,
                first_sequence,
            )
        )
        self._chunk_path = path
        self._chunk_count = 0
        self._chunk_first_sequence = first_sequence
        self._chunk_last_sequence = first_sequence

    def write_frame(self, frame: dict[str, Any]) -> None:
        sequence = int(frame["sequence"])
        if self._chunk is None:
            self._open_chunk(sequence)
        assert self._chunk is not None
        self._chunk.write(_record(FRAME_RECORD, encode_frame(frame)))
        self._chunk_count += 1
        self._total_frames += 1
        self._chunk_last_sequence = sequence
        if self._chunk_count % self.flush_frames == 0:
            self._chunk.flush()
            os.fsync(self._chunk.fileno())
        if self._chunk_count >= self.chunk_frames:
            self._close_chunk()

    def _close_chunk(self) -> None:
        if self._chunk is None or self._chunk_path is None:
            return
        self._chunk.write(
            _record(
                FOOTER_RECORD,
                FOOTER.pack(self._chunk_count, self._chunk_last_sequence),
            )
        )
        self._chunk.flush()
        os.fsync(self._chunk.fileno())
        self._chunk.close()
        final = self._chunk_path.with_suffix("")
        os.replace(self._chunk_path, final)
        info = ChunkInfo(
            index=self._chunk_index,
            path=str(final.relative_to(self.session_dir)).replace("\\", "/"),
            frame_count=self._chunk_count,
            first_sequence=self._chunk_first_sequence,
            last_sequence=self._chunk_last_sequence,
            bytes=final.stat().st_size,
            sha256=_sha256(final),
            complete=True,
        )
        self.manifest["chunks"].append(info.__dict__)
        _atomic_json(self.session_dir / "manifest.json", self.manifest)
        self._chunk = None
        self._chunk_path = None
        self._chunk_index += 1

    def stop(self, *, summary: dict[str, Any] | None = None) -> None:
        self._close_chunk()
        self.manifest["clean_termination"] = True
        self.manifest["final_frame_count"] = self._total_frames
        self.manifest["capture_end_utc"] = datetime.now(UTC).isoformat()
        self.manifest["summary"] = summary or {}
        _atomic_json(self.session_dir / "manifest.json", self.manifest)

    def interrupt_for_test(self) -> None:
        """Flush but deliberately leave the current chunk partial and manifest incomplete."""

        if self._chunk is not None:
            self._chunk.flush()
            os.fsync(self._chunk.fileno())
            self._chunk.close()
            self._chunk = None


def iter_chunk_records(
    path: Path, *, recover_partial: bool
) -> Iterator[tuple[int, bytes, int]]:
    """Yield record type, payload, and byte offset from one chunk."""

    partial = path.name.endswith(".partial")
    with path.open("rb") as stream:
        header = stream.read(CHUNK_HEADER.size)
        if len(header) != CHUNK_HEADER.size:
            if recover_partial and partial:
                return
            raise CorruptChunkError(f"truncated chunk header: {path}")
        magic, version, _index, _session, _first_sequence = CHUNK_HEADER.unpack(header)
        if magic != CHUNK_MAGIC or version != SCHEMA_VERSION:
            raise CorruptChunkError(f"unsupported chunk header: {path}")
        saw_footer = False
        while True:
            offset = stream.tell()
            header = stream.read(RECORD_HEADER.size)
            if not header:
                if not partial and not saw_footer:
                    raise CorruptChunkError(f"complete chunk has no footer: {path}")
                return
            if len(header) != RECORD_HEADER.size:
                if recover_partial and partial:
                    return
                raise CorruptChunkError(f"truncated record header at {offset}: {path}")
            magic, record_type, size, expected_crc = RECORD_HEADER.unpack(header)
            if magic != RECORD_MAGIC:
                raise CorruptChunkError(f"bad record magic at {offset}: {path}")
            payload = stream.read(size)
            if len(payload) != size:
                if recover_partial and partial:
                    return
                raise CorruptChunkError(f"truncated record at {offset}: {path}")
            actual_crc = zlib.crc32(payload)
            if actual_crc != expected_crc:
                if recover_partial and partial:
                    return
                raise CorruptChunkError(
                    f"CRC mismatch at {offset}: expected {expected_crc:08X}, "
                    f"got {actual_crc:08X}: {path}"
                )
            if saw_footer:
                raise CorruptChunkError(f"record found after footer at {offset}: {path}")
            if record_type == FOOTER_RECORD:
                if len(payload) != FOOTER.size:
                    raise CorruptChunkError(f"invalid footer size at {offset}: {path}")
                saw_footer = True
            yield record_type, payload, offset
