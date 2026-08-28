"""Command-line inspection for native Rival 2.0 human demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rivalsim.human_demo.analysis import (
    action_variation_collection_report,
    action_variation_report,
    rival_observation_mapping_report,
)
from rivalsim.human_demo.reader import SessionReader


def _emit(value: Any, output: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rivalsim.human_demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("session", type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--strict-complete", action="store_true")
    mapping = subparsers.add_parser("mapping-report")
    mapping.add_argument("--output", type=Path)
    variation = subparsers.add_parser("action-variation")
    variation.add_argument("session", type=Path, nargs="+")
    variation.add_argument("--output", type=Path)
    variation.add_argument("--analog-epsilon", type=float, default=1e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mapping-report":
        _emit(rival_observation_mapping_report(), args.output)
        return 0
    reader = SessionReader(args.session if isinstance(args.session, Path) else args.session[0])
    if args.command in {"inspect", "validate"}:
        report = reader.validate(recover_partial=not args.strict_complete)
        payload: dict[str, Any] = {"validation": report.as_dict()}
        if args.command == "inspect":
            session_class = ":".join(
                str(reader.manifest.get(key, ""))
                for key in ("session_type", "label", "mechanic_label", "opponent_label")
                if reader.manifest.get(key)
            )
            payload["manifest"] = reader.manifest
            payload["event_count"] = sum(1 for _ in reader.iter_events())
            payload["marker_count"] = sum(1 for _ in reader.iter_markers())
            payload["action_variation"] = action_variation_report(
                reader.iter_frames(), session_class=session_class or "unlabeled"
            )
            payload["observation_mapping"] = rival_observation_mapping_report()["counts"]
        _emit(payload, args.output)
        return 0 if report.valid else 1
    if args.command == "action-variation":
        readers = [SessionReader(path) for path in args.session]
        labeled_sessions = []
        for session_reader in readers:
            session_class = ":".join(
                str(session_reader.manifest.get(key, ""))
                for key in ("session_type", "label", "mechanic_label", "opponent_label")
                if session_reader.manifest.get(key)
            )
            labeled_sessions.append(
                (session_class or "unlabeled", session_reader.iter_frames())
            )
        payload = (
            action_variation_report(
                labeled_sessions[0][1],
                analog_epsilon=args.analog_epsilon,
                session_class=labeled_sessions[0][0],
            )
            if len(labeled_sessions) == 1
            else action_variation_collection_report(
                labeled_sessions, analog_epsilon=args.analog_epsilon
            )
        )
        _emit(payload, args.output)
        return 0
    return 2
