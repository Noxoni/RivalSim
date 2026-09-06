"""Read-only first/post-goal kickoff input tap for the unchanged match evaluator.

Requires the learner's exclusive GPU lease. Exactly replays a saved scheduled
match evaluation; never alters controls, observations, resets or opponent state.
The trace is not interpretable unless every saved raw evaluation value matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.trace_rival2_direct_skills import (  # noqa: E402
    BoundedFrames,
    differences,
    temporary_attribute,
)

SCHEMA = "RIVAL2_DIRECT_SKILLS_KICKOFF_RESET_TRACE_V1"
WINDOW_TICKS = 32


def cpu(value):
    return value.detach().cpu().numpy().copy()


def capture_decision(runner, original, frames, lifecycle_views):
    """Call the exact original action method once; only copy its inputs/outputs."""
    age = cpu(runner.bridge.views["rival2.episode_ticks"])
    eligible = (age < WINDOW_TICKS) & (cpu(runner.match_views["done"]) == 0)
    if not eligible.any():
        return original()
    n = runner.num_worlds
    row = dict(
        host_tick=np.asarray(runner.host_tick, dtype=np.int64),
        eligible=eligible,
        episode_ticks=age,
        observation=cpu(runner.rival_observation),
        hidden_before=cpu(runner.hidden),
        rival_side=cpu(runner.rival_side),
        hidden_reset_count=cpu(runner.hidden_reset_count),
        nexto_previous_action=cpu(runner.nexto.previous_action),
        nexto_neural_counter=cpu(runner.nexto.neural_counter),
        nexto_kickoff_index=cpu(runner.nexto.kickoff_index),
        match_kickoff_active=cpu(runner.match_views["kickoff_active"]),
        match_goal_count=cpu(runner.match_views["goal_count"]),
        match_starting_layout=cpu(runner.match_views["starting_layout"]),
        lifecycle_layout=cpu(lifecycle_views["kickoff_layout"]),
        lifecycle_selector=cpu(lifecycle_views["kickoff_selector"]),
    )
    for name, shape in (
        ("car_pos", (n, 2, 3)),
        ("car_vel", (n, 2, 3)),
        ("car_quat", (n, 2, 4)),
        ("car_ang_vel", (n, 2, 3)),
        ("wheel_contact", (n, 2, 4)),
        ("boost", (n, 2)),
        ("ball_pos", (n, 3)),
        ("ball_vel", (n, 3)),
    ):
        row[name] = cpu(runner.bridge.views[name]).reshape(shape)
    result = original()
    row["rival_action"] = cpu(runner.rival_action)
    row["hidden_after"] = cpu(runner.hidden)
    frames.append(row)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(offset, output):
    # These imports construct no policy or optimizer. Native allocation starts
    # only inside the same exclusive lease used by the production learner.
    import warp as wp

    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import (
        OVERTIME_CAP_TICKS,
        REGULATION_TICKS,
        CandidateMatchRunner,
        summarize,
    )
    from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease

    results = ROOT / "results/rival2/direct_skills_v1"
    source = results / f"full_match_{offset:06d}.json"
    checkpoint = ROOT / f"checkpoints/rival2/direct_skills_v1/plus_{offset:06d}.pt"
    saved = json.loads(source.read_text())
    expected_hash = saved["checkpoint"]["sha256"]
    assert sha(checkpoint) == expected_hash
    assert saved["accepted_updates"] == offset and saved["optimizer_steps"] == 0
    assert saved["model_unchanged"] and saved["checkpoint_unchanged"]
    package = json.loads((results / "package.json").read_text())
    authority = json.loads((results / "authority.json").read_text())
    authority_hash = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest().upper()
    assert saved["authority_sha256"] == package["authority_sha256"] == authority_hash
    for name, expected in package["sources"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        assert actual.hexdigest().upper() == expected, name
    if output.exists():
        raise RuntimeError("Trace output exists; never overwrite a completed or failed attempt")
    with gpu_lease(), owned_match_stream():
        output.mkdir(parents=True, exist_ok=False)
        try:
            runner = CandidateMatchRunner(checkpoint, expected_hash, entity=True)
            frames = BoundedFrames((REGULATION_TICKS + OVERTIME_CAP_TICKS) // 4)
            lifecycle_views = {
                name: wp.to_torch(getattr(runner.world.lifecycle, name))
                for name in ("kickoff_layout", "kickoff_selector")
            }
            original = runner._update_rival_action
            with temporary_attribute(
                runner,
                "_update_rival_action",
                lambda: capture_decision(runner, original, frames, lifecycle_views),
            ):
                runner.run_ticks(REGULATION_TICKS)
                for _ in range(OVERTIME_CAP_TICKS // 600):
                    if bool(runner.phase_status()["done"].all()):
                        break
                    runner.run_ticks(600)
            raw = runner.export()["raw"]
            observed = dict(
                raw={k: v.tolist() for k, v in raw.items()},
                summary=summarize(raw),
                hidden_resets=cpu(runner.hidden_reset_count).tolist(),
            )
            expected = {key: saved[key] for key in observed}
            diff = differences(expected, observed)
            assert not diff, diff
            assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
            assert sha(checkpoint) == expected_hash
            arrays = frames.arrays("")
            assert all(np.isfinite(a).all() for a in arrays.values())
            archive = output / "kickoff_reset.npz"
            np.savez_compressed(archive, **arrays)
            with np.load(archive, allow_pickle=False) as recovered:
                assert set(recovered.files) == set(arrays)
                assert all(np.array_equal(recovered[k], v) for k, v in arrays.items())
            manifest = dict(
                schema=SCHEMA,
                accepted_updates=offset,
                checkpoint_sha256=expected_hash,
                source=source.relative_to(ROOT).as_posix(),
                source_sha256=sha(source),
                authority_sha256=saved["authority_sha256"],
                trace_window_physics_ticks=WINDOW_TICKS,
                method="Eight 30Hz decisions per initial/post-goal kickoff; other rows masked",
                exact_saved_raw_summary_hidden_reset_parity=True,
                model_checkpoint_unchanged=True,
                new_optimizer_steps=0,
                frozen_training_sources=package["sources"],
                archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,
                decision_records=len(frames.rows),
                eligible_world_decisions=int(arrays["eligible"].sum()),
                capture_script_sha256=sha(Path(__file__)),
                array_schema={
                    k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()
                },
            )
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in manifest.items()
                        if k not in ("array_schema", "frozen_training_sources")
                    }
                )
            )
        except BaseException as exc:
            (output / "failure.json").write_text(
                json.dumps(
                    dict(
                        schema=SCHEMA,
                        accepted_updates=offset,
                        checkpoint_sha256=expected_hash,
                        exception_type=type(exc).__name__,
                        reason=str(exc),
                        optimizer_steps=0,
                        trace_interpretation_permitted=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
            raise


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)  # Match the production evaluator's process configuration.
    run(args.update, args.output)
