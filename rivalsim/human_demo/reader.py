"""Deterministic reader and validator for native demonstration sessions."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rivalsim.human_demo.format import (
    ACTION_NAMES,
    CHUNK_HEADER,
    CHUNK_MAGIC,
    FOOTER,
    FOOTER_RECORD,
    FRAME_FLAG_DUPLICATE_PHYSICS_FRAME,
    FRAME_FLAG_OUT_OF_ORDER_PHYSICS_FRAME,
    FRAME_RECORD,
    CorruptChunkError,
    decode_frame,
    iter_chunk_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@dataclass
class ValidationReport:
    session_dir: str
    frame_count: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    final_frame_physics_frame: int | None = None
    final_frame_engine_time: float | None = None
    final_event_physics_frame: int | None = None
    final_event_engine_time: float | None = None
    final_marker_physics_frame: int | None = None
    final_marker_engine_time: float | None = None
    session_stop_physics_frame: int | None = None
    session_stop_engine_time: float | None = None
    trailing_uncovered_physics_ticks: int = 0
    trailing_uncovered_engine_seconds: float = 0.0
    active_capture_start: float | None = None
    active_capture_end: float | None = None
    active_capture_start_physics_frame: int | None = None
    active_capture_end_physics_frame: int | None = None
    active_capture_duration: float = 0.0
    active_capture_rate_hz: float = 0.0
    session_wall_duration: float = 0.0
    session_wide_capture_rate_hz: float = 0.0
    sequence_gap_count: int = 0
    missing_sequence_count: int = 0
    duplicate_sequence_count: int = 0
    out_of_order_sequence_count: int = 0
    duplicate_physics_frame_count: int = 0
    out_of_order_physics_frame_count: int = 0
    missing_physics_frame_count: int = 0
    invalid_human_car_frames: int = 0
    invalid_action_frames: int = 0
    complete_chunks: int = 0
    partial_chunks: int = 0
    clean_termination: bool = False
    manifest_hashes_valid: bool = True
    capture_completeness_valid: bool = False
    errors: list[str] = field(default_factory=list)
    completeness_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def container_valid(self) -> bool:
        return not self.errors

    @property
    def capture_complete(self) -> bool:
        return self.capture_completeness_valid

    @property
    def overall_demonstration_valid(self) -> bool:
        return (
            self.container_valid
            and self.clean_termination
            and self.capture_completeness_valid
        )

    @property
    def valid(self) -> bool:
        """Backward-compatible alias for the ingestion-safe overall verdict."""

        return self.overall_demonstration_valid

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "container_valid": self.container_valid,
            "capture_complete": self.capture_complete,
            "overall_demonstration_valid": self.overall_demonstration_valid,
            "valid": self.valid,
        }


class SessionReader:
    """Read frames in deterministic chunk-index and record-offset order."""

    def __init__(self, session_dir: Path | str) -> None:
        self.session_dir = Path(session_dir).resolve()
        manifest_path = self.session_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"session manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def _chunk_paths(self) -> list[Path]:
        chunks_dir = self.session_dir / "chunks"
        if not chunks_dir.is_dir():
            return []
        return sorted(
            (
                path
                for path in chunks_dir.iterdir()
                if path.name.endswith(".rvr") or path.name.endswith(".rvr.partial")
            ),
            key=lambda path: path.name,
        )

    def iter_frames(self, *, recover_partial: bool = True) -> Iterator[dict[str, Any]]:
        for path in self._chunk_paths():
            for record_type, payload, _offset in iter_chunk_records(
                path, recover_partial=recover_partial
            ):
                if record_type == FRAME_RECORD:
                    yield decode_frame(payload)
                elif record_type == FOOTER_RECORD:
                    continue
                else:
                    raise CorruptChunkError(f"unknown record type {record_type}: {path}")

    def validate(self, *, recover_partial: bool = True) -> ValidationReport:
        report = ValidationReport(session_dir=str(self.session_dir))
        report.clean_termination = bool(self.manifest.get("clean_termination", False))
        report.session_wall_duration = float(
            self.manifest.get(
                "session_wall_duration_seconds",
                self.manifest.get("duration_seconds", 0.0),
            )
        )
        report.session_wide_capture_rate_hz = float(
            self.manifest.get(
                "session_wide_capture_rate_hz",
                self.manifest.get("observed_capture_rate_hz", 0.0),
            )
        )
        manifest_chunks = {
            str(row["path"]): row for row in self.manifest.get("chunks", [])
        }
        for path in self._chunk_paths():
            relative = str(path.relative_to(self.session_dir)).replace("\\", "/")
            if path.name.endswith(".partial"):
                report.partial_chunks += 1
                report.warnings.append(f"recovering incomplete chunk {relative}")
                continue
            report.complete_chunks += 1
            expected = manifest_chunks.get(relative)
            if expected is None:
                report.manifest_hashes_valid = False
                report.errors.append(f"complete chunk absent from manifest: {relative}")
                continue
            actual_sha = _sha256(path)
            if actual_sha != expected.get("sha256"):
                report.manifest_hashes_valid = False
                report.errors.append(
                    f"chunk SHA-256 mismatch for {relative}: {actual_sha}"
                )
            if path.stat().st_size != int(expected.get("bytes", -1)):
                report.manifest_hashes_valid = False
                report.errors.append(f"chunk byte-size mismatch for {relative}")
            try:
                with path.open("rb") as stream:
                    header = stream.read(CHUNK_HEADER.size)
                magic, version, chunk_index, session_bytes, first_sequence = (
                    CHUNK_HEADER.unpack(header)
                )
                if magic != CHUNK_MAGIC:
                    report.errors.append(f"bad chunk magic for {relative}")
                if version != int(self.manifest.get("schema_version", version)):
                    report.errors.append(f"chunk schema version mismatch for {relative}")
                if chunk_index != int(expected.get("index", -1)):
                    report.errors.append(f"chunk index mismatch for {relative}")
                manifest_session = self.manifest.get("session_uuid")
                if manifest_session and session_bytes != uuid.UUID(str(manifest_session)).bytes:
                    report.errors.append(f"chunk session UUID mismatch for {relative}")
                if first_sequence != int(expected.get("first_sequence", -1)):
                    report.errors.append(f"chunk first sequence mismatch for {relative}")
                frame_sequences: list[int] = []
                footer: tuple[int, int] | None = None
                for record_type, payload, _ in iter_chunk_records(
                    path, recover_partial=False
                ):
                    if record_type == FRAME_RECORD:
                        (sequence,) = struct.unpack_from("<Q", payload)
                        frame_sequences.append(sequence)
                    elif record_type == FOOTER_RECORD:
                        footer = FOOTER.unpack(payload)
                if footer is None:
                    report.errors.append(f"complete chunk has no footer: {relative}")
                else:
                    footer_count, footer_last_sequence = footer
                    expected_count = int(expected.get("frame_count", -1))
                    expected_last = int(expected.get("last_sequence", -1))
                    if len(frame_sequences) != expected_count or footer_count != expected_count:
                        report.errors.append(f"chunk frame count mismatch for {relative}")
                    if not frame_sequences or frame_sequences[-1] != expected_last:
                        report.errors.append(f"chunk last sequence mismatch for {relative}")
                    if footer_last_sequence != expected_last:
                        report.errors.append(f"chunk footer last sequence mismatch for {relative}")
            except (CorruptChunkError, ValueError, struct.error) as error:
                report.errors.append(str(error))

        for row in self.manifest.get("files", []):
            relative = str(row.get("path", ""))
            path = self.session_dir / relative
            if not path.is_file():
                report.manifest_hashes_valid = False
                report.errors.append(f"manifest file is missing: {relative}")
                continue
            if path.stat().st_size != int(row.get("bytes", -1)):
                report.manifest_hashes_valid = False
                report.errors.append(f"file byte-size mismatch for {relative}")
            if _sha256(path) != row.get("sha256"):
                report.manifest_hashes_valid = False
                report.errors.append(f"file SHA-256 mismatch for {relative}")

        previous_sequence: int | None = None
        previous_physics: int | None = None
        try:
            frames = self.iter_frames(recover_partial=recover_partial)
            for frame in frames:
                sequence = int(frame["sequence"])
                physics_frame = int(frame["physics_frame"])
                if report.first_sequence is None:
                    report.first_sequence = sequence
                    report.active_capture_start = float(frame["engine_physics_time"])
                    report.active_capture_start_physics_frame = physics_frame
                report.last_sequence = sequence
                report.final_frame_physics_frame = physics_frame
                report.final_frame_engine_time = float(frame["engine_physics_time"])
                report.active_capture_end = report.final_frame_engine_time
                report.active_capture_end_physics_frame = physics_frame
                report.frame_count += 1
                if previous_sequence is not None:
                    if sequence == previous_sequence:
                        report.duplicate_sequence_count += 1
                        report.errors.append(f"duplicate recorder sequence {sequence}")
                    elif sequence < previous_sequence:
                        report.out_of_order_sequence_count += 1
                        report.errors.append(
                            f"out-of-order recorder sequence {sequence} after {previous_sequence}"
                        )
                    elif sequence > previous_sequence + 1:
                        report.sequence_gap_count += 1
                        report.missing_sequence_count += sequence - previous_sequence - 1
                if previous_physics is not None:
                    if physics_frame == previous_physics:
                        report.duplicate_physics_frame_count += 1
                    elif physics_frame < previous_physics:
                        report.out_of_order_physics_frame_count += 1
                report.missing_physics_frame_count += int(
                    frame.get("missing_physics_frames", 0)
                )
                status = int(frame.get("status_flags", 0))
                if status & FRAME_FLAG_DUPLICATE_PHYSICS_FRAME and (
                    previous_physics is None or physics_frame != previous_physics
                ):
                    report.errors.append(
                        f"incorrect duplicate-physics flag at sequence {sequence}"
                    )
                if status & FRAME_FLAG_OUT_OF_ORDER_PHYSICS_FRAME and (
                    previous_physics is None or physics_frame >= previous_physics
                ):
                    report.errors.append(
                        f"incorrect out-of-order-physics flag at sequence {sequence}"
                    )
                local_cars = [
                    car for car in frame["cars"] if car["flags"]["is_local_human"]
                ]
                if len(local_cars) != 1:
                    report.invalid_human_car_frames += 1
                    report.errors.append(
                        f"sequence {sequence} has {len(local_cars)} local human cars"
                    )
                action = frame["rival_action"]
                analog_valid = all(-1.0 <= float(action[name]) <= 1.0 for name in ACTION_NAMES[:5])
                buttons_valid = all(isinstance(action[name], bool) for name in ACTION_NAMES[5:])
                if not analog_valid or not buttons_valid:
                    report.invalid_action_frames += 1
                    report.errors.append(f"invalid Rival action at sequence {sequence}")
                previous_sequence = sequence
                previous_physics = physics_frame
        except (CorruptChunkError, ValueError) as error:
            report.errors.append(str(error))

        expected_count = self.manifest.get("final_frame_count")
        if (
            report.clean_termination
            and expected_count is not None
            and report.frame_count != int(expected_count)
        ):
            report.errors.append(
                f"manifest frame count {expected_count} != decoded {report.frame_count}"
            )
        if not report.clean_termination:
            report.warnings.append("session did not record a clean stop")
        self._evaluate_capture_completeness(report)
        return report

    def _evaluate_capture_completeness(self, report: ValidationReport) -> None:
        if report.active_capture_start is not None and report.active_capture_end is not None:
            report.active_capture_duration = max(
                0.0, report.active_capture_end - report.active_capture_start
            )
        if report.active_capture_duration > 0.0:
            report.active_capture_rate_hz = (
                report.frame_count / report.active_capture_duration
            )

        try:
            events = list(self.iter_events())
            markers = list(self.iter_markers())
        except ValueError as error:
            report.errors.append(str(error))
            events = []
            markers = []

        def latest_timing(
            records: list[dict[str, Any]],
        ) -> tuple[int | None, float | None]:
            candidates: list[tuple[int, float]] = []
            for record in records:
                physics = int(record.get("physics_frame", -1))
                engine_time = float(record.get("engine_physics_time", -1.0))
                if physics >= 0 or engine_time >= 0.0:
                    candidates.append((physics, engine_time))
            if not candidates:
                return None, None
            physics, engine_time = max(candidates, key=lambda value: (value[0], value[1]))
            return (physics if physics >= 0 else None, engine_time if engine_time >= 0 else None)

        (
            report.final_event_physics_frame,
            report.final_event_engine_time,
        ) = latest_timing(events)
        (
            report.final_marker_physics_frame,
            report.final_marker_engine_time,
        ) = latest_timing(markers)

        stop_physics = int(self.manifest.get("capture_stop_physics_frame", -1))
        stop_engine = float(self.manifest.get("capture_stop_engine_time", -1.0))
        report.session_stop_physics_frame = stop_physics if stop_physics >= 0 else None
        report.session_stop_engine_time = stop_engine if stop_engine >= 0.0 else None

        physics_evidence = [
            value
            for value in (
                report.final_frame_physics_frame,
                report.final_event_physics_frame,
                report.final_marker_physics_frame,
                report.session_stop_physics_frame,
            )
            if value is not None
        ]
        engine_evidence = [
            value
            for value in (
                report.final_frame_engine_time,
                report.final_event_engine_time,
                report.final_marker_engine_time,
                report.session_stop_engine_time,
            )
            if value is not None and math.isfinite(value)
        ]
        if report.final_frame_physics_frame is not None and physics_evidence:
            report.trailing_uncovered_physics_ticks = max(
                0, max(physics_evidence) - report.final_frame_physics_frame
            )
        if report.final_frame_engine_time is not None and engine_evidence:
            report.trailing_uncovered_engine_seconds = max(
                0.0, max(engine_evidence) - report.final_frame_engine_time
            )

        tolerance_ticks = int(
            self.manifest.get("capture_completeness_tolerance_ticks", 4)
        )
        engine_rate = float(self.manifest.get("engine_physics_framerate_hz", 120.0))
        tolerance_seconds = max(
            0.05,
            tolerance_ticks / engine_rate if engine_rate > 0.0 else 0.05,
        )
        report.capture_completeness_valid = report.clean_termination and report.frame_count > 0
        if not report.clean_termination:
            report.completeness_errors.append("session termination is incomplete")
        if report.frame_count == 0:
            report.capture_completeness_valid = False
            report.completeness_errors.append("session contains no demonstration frames")
        if report.trailing_uncovered_physics_ticks > tolerance_ticks or (
            report.trailing_uncovered_engine_seconds > tolerance_seconds
        ):
            report.capture_completeness_valid = False
            report.completeness_errors.append(
                "active simulation continues beyond the final demonstration frame: "
                f"{report.trailing_uncovered_physics_ticks} physics ticks, "
                f"{report.trailing_uncovered_engine_seconds:.9g} engine seconds"
            )

        manifest_missing = int(self.manifest.get("missing_physics_frame_count", 0))
        report.missing_physics_frame_count = max(
            report.missing_physics_frame_count, manifest_missing
        )
        discontinuities = {
            "duplicate physics frames": report.duplicate_physics_frame_count,
            "out-of-order physics frames": report.out_of_order_physics_frame_count,
            "missing physics frames": report.missing_physics_frame_count,
            "queue-dropped frames": int(
                self.manifest.get("queue_dropped_frame_count", 0)
            ),
            "retained duplicate frames": int(
                self.manifest.get("duplicate_frames_retained", 0)
            ),
            "identity failures": int(self.manifest.get("identity_failure_count", 0)),
        }
        for label, count in discontinuities.items():
            if count > 0:
                report.capture_completeness_valid = False
                report.completeness_errors.append(f"{label}: {count}")

    def iter_events(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_jsonl("events.jsonl")

    def iter_markers(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_jsonl("markers.jsonl")

    def _iter_jsonl(self, name: str) -> Iterator[dict[str, Any]]:
        path = self.session_dir / name
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    if not self.manifest.get("clean_termination") and line_number > 1:
                        return
                    raise ValueError(f"invalid {name} line {line_number}") from error
