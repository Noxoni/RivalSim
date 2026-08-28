"""Deterministic reader and validator for native demonstration sessions."""

from __future__ import annotations

import hashlib
import json
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
    sequence_gap_count: int = 0
    missing_sequence_count: int = 0
    duplicate_sequence_count: int = 0
    out_of_order_sequence_count: int = 0
    duplicate_physics_frame_count: int = 0
    out_of_order_physics_frame_count: int = 0
    invalid_human_car_frames: int = 0
    invalid_action_frames: int = 0
    complete_chunks: int = 0
    partial_chunks: int = 0
    clean_termination: bool = False
    manifest_hashes_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


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
                report.last_sequence = sequence
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
        return report

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
