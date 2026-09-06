"""Read-only six-second trace of Rival, using the unchanged corrected evaluator.

No learner, optimizer, controller replacement, observation transform, or reward.
The existing runner owns every simulation step, action, hidden update and reset.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_direct_skills_native_nexto_v1 import (
    NativeNextoMatchRunner, SOURCES as EVAL_SOURCES, specification,
    text_sha, verify as verify_evaluator,
)
from benchmarks.report_direct_skills_native_nexto_v1 import sha
from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash, utc
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease

OUT = ROOT / "results/rival2/direct_skills_native_kickoff_v1"
BASE = "results/rival2/direct_skills_native_nexto_v1/"
VERSION = "RIVAL2_NATIVE_NEXTO_RIVAL_KICKOFF_TRACE_V1"
TICKS = 720
CHANNELS = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
CASES = {
    "parent650": dict(path="checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt",
        sha256="939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D",
        match=BASE + "finishing650.json"),
    "child10": dict(path="checkpoints/rival2/direct_skills_native_nexto_v1/child_000010.pt",
        sha256="8764A22E0A678A0559DA53F7DD6A9EEC58415AF3565A58C582AEC4630EC1BE1E",
        match=BASE + "child_000010.json"),
    "child25": dict(path="checkpoints/rival2/direct_skills_native_nexto_v1/child_000025.pt",
        sha256="D53603E3B3DBD990EBC06E44179932C33F03358F4B4300098E2CAD1BD523B29D",
        match=BASE + "child_000025.json"),
}
CAR_SCALARS = ("boost", "on_ground", "has_jumped", "is_jumping", "has_double_jumped",
    "has_flipped", "is_flipping", "jump_time", "air_time", "air_time_since_jump",
    "flip_time", "is_supersonic", "car_is_demoed")
SHAPES = {"car_pos": (2, 3), "car_vel": (2, 3), "car_quat": (2, 4),
    "car_ang_vel": (2, 3), "wheel_contact": (2, 4), "ball_pos": (3,),
    "ball_vel": (3,), "ball_ang_vel": (3,), **{k: (2,) for k in CAR_SCALARS}}
SOURCES = tuple(dict.fromkeys((*EVAL_SOURCES,
    "benchmarks/trace_direct_skills_native_kickoff_v1.py",
    "tests/test_direct_skills_native_kickoff_v1.py")))


def save_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def protocol():
    return dict(version=VERSION, cases=CASES, ticks=TICKS, worlds=10,
        evaluation=specification(), channels=CHANNELS, raw_state_shapes=SHAPES,
        method="Subclass calls original action update and original tick exactly once. Copy pre/post native state at every physics tick; capture native control buffers immediately after original set_actions. Record actual 30Hz model observations and recurrent hidden input/output. No scripted Rival input, action sampling, observation edit, or optimizer.",
        interpretation="Initial six seconds only, exclude all post-goal restart ticks. Pre-first-contact race metrics use only ticks before either car's first contact. No mechanic labels, no inferred intent, no causal learning guarantee.",
        integrity="Exact deterministic actor/action/hidden replay at original batch size; all four-tick holds; actual native controls equal scheduled actions; complete finite buffers; unchanged checkpoints and both models; exact goal event prefix versus existing full evaluation. Prefix equality alone is not full trajectory equality.",
        storage="Lossless NumPy NPZ: ticks are zero-based pre-action; post state may include existing four-tick goal reset. goal counts/contact counters identify boundaries. Model arrays indexed every four ticks. Float32 native values preserved. JSON contains array dtypes/shapes/SHA and physical reduction.",
        scope="One case per checkpoint, 7200 world ticks each, shared GPU lease; no PPO or reward/controller changes",
        sources={p: text_sha(ROOT / p) for p in SOURCES},
        full_evaluation_hashes={v["match"]: sha(ROOT / v["match"]) for v in CASES.values()})


def prepare():
    verify_evaluator()
    for case in CASES.values():
        assert sha(ROOT / case["path"]) == case["sha256"]
    save_json(OUT / "protocol.json", protocol())


def verify():
    verify_evaluator()
    frozen = json.loads((OUT / "protocol.json").read_text())
    current = json.loads(json.dumps(protocol()))
    recovery_path = OUT / "serialization_recovery.json"
    extra = []
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        assert recovery["original_protocol_sha256"] == sha(OUT / "protocol.json")
        assert recovery["capture_semantics_changed"] is False
        assert recovery["failed_archive_sha256"] == sha(OUT / "parent650.npz")
        for p, h in recovery["replacement_sources"].items():
            frozen["sources"][p] = h
        extra.append(recovery_path.relative_to(ROOT).as_posix())
    assert frozen == current
    for p in (*SOURCES, (OUT / "protocol.json").relative_to(ROOT).as_posix(), *extra):
        remote = subprocess.check_output(["git", "show", "origin/main:" + p], cwd=ROOT)
        assert remote.replace(b"\r\n", b"\n") == (ROOT / p).read_bytes().replace(b"\r\n", b"\n"), p
    for case in CASES.values():
        assert sha(ROOT / case["path"]) == case["sha256"]
    return frozen


class TraceBuffers:
    """Preallocated device copies; source state is never written through."""
    def __init__(self, runner, ticks=TICKS):
        self.data = {}
        self.ticks = ticks
        self.n = n = runner.num_worlds
        self.sources = {k: runner.bridge.views[k].reshape(n, *shape) for k, shape in SHAPES.items()}
        self.sources["touch_count"] = wp.to_torch(runner.telemetry.touch_count).reshape(n, 2)
        for k in ("goal_count", "kickoff_active", "pending_reset"):
            self.sources[k] = runner.match_views[k]
        for prefix in ("pre", "post"):
            for k, value in self.sources.items():
                self.data[prefix + "." + k] = value.new_empty((ticks, *value.shape))
        self.data["applied_controls"] = runner.actions.new_empty((ticks, n, 2, 8))
        self.data["scheduled_controls"] = torch.empty_like(self.data["applied_controls"])
        self.data["observation"] = runner.rival_observation.new_empty((ticks // 4, n, 182))
        for k in ("hidden_before", "hidden_after"):
            self.data[k] = runner.hidden.new_empty((ticks // 4, *runner.hidden.shape))
        self.data["policy_action"] = runner.rival_action.new_empty((ticks // 4, n, 8))
        self.action_calls = 0

    def state(self, prefix, tick):
        for k, value in self.sources.items():
            self.data[prefix + "." + k][tick].copy_(value)

    def action_input(self, runner):
        i = runner.host_tick // 4
        self.data["observation"][i].copy_(runner.rival_observation[runner.batch_index, runner.rival_side])
        self.data["hidden_before"][i].copy_(runner.hidden)

    def action_output(self, runner):
        i = runner.host_tick // 4
        self.data["hidden_after"][i].copy_(runner.hidden)
        self.data["policy_action"][i].copy_(runner.rival_action)
        self.action_calls += 1


class TracedRunner(NativeNextoMatchRunner):
    def __init__(self, path, expected):
        super().__init__(path, expected)
        self.trace = TraceBuffers(self)
        original_set = self.bridge.set_actions

        def observe_controls(action):
            emitted = original_set(action)
            i = self.host_tick
            self.trace.data["scheduled_controls"][i].copy_(action)
            for ch, name in enumerate(CHANNELS):
                self.trace.data["applied_controls"][i, :, :, ch].copy_(
                    self.bridge.views["control." + name].reshape(self.num_worlds, 2))
            return emitted

        self.bridge.set_actions = observe_controls

    @torch.inference_mode()
    def _update_rival_action(self):
        self.trace.action_input(self)
        super()._update_rival_action()
        self.trace.action_output(self)

    def tick(self):
        self._activate_stream()
        i = self.host_tick
        assert i < TICKS
        self.trace.state("pre", i)
        super().tick()
        self.trace.state("post", i)


def first_index(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def forward(quat):
    x, y, z, w = np.moveaxis(quat, -1, 0)
    return np.stack((1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w)), -1)


def physical_summary(data, sides, layouts):
    """Plain physical measurements, never named-mechanic/reward classifications."""
    result = []
    controls = data["applied_controls"]
    touches = data["post.touch_count"] - data["pre.touch_count"]
    for world, side in enumerate(sides):
        valid = data["pre.goal_count"][:, world] == 0
        touch_ticks = [first_index((touches[:, world, car] > 0) & valid) for car in (0, 1)]
        first = min((v for v in touch_ticks if v is not None), default=len(valid))
        race = valid & (np.arange(len(valid)) < first)
        row = dict(world=world, rival_side=int(side), layout=int(layouts[world]),
            initial_ticks=int(valid.sum()), first_contact_physics_tick=[None if t is None else t+1 for t in touch_ticks],
            pre_contact_ticks=int(race.sum()), cars=[])
        for car in (0, 1):
            pos = data["pre.car_pos"][:, world, car]
            vel = data["pre.car_vel"][:, world, car]
            diff = data["pre.ball_pos"][:, world] - pos
            distance = np.linalg.norm(diff, axis=-1)
            fwd = forward(data["pre.car_quat"][:, world, car])
            facing = np.sum(fwd * diff, axis=-1) / np.maximum(distance, 1e-12)
            along = np.sum(vel * diff, axis=-1) / np.maximum(distance, 1e-12)
            c = controls[:, world, car]
            mask = race
            measurements = dict(car=car, is_rival=bool(car == side),
                first_action=c[0].tolist(),
                first_jump_tick=first_index((c[:, 5] != 0) & valid),
                first_boost_tick=first_index((c[:, 6] != 0) & valid),
                initial_contacts=int(touches[valid, world, car].sum()),
                command_change_ticks=np.flatnonzero(np.any(c[1:] != c[:-1], axis=-1) & valid[1:]).__add__(1).tolist(),
                race_mean_action=c[mask].mean(axis=0).tolist() if mask.any() else None,
                race_speed_mean=float(np.linalg.norm(vel[mask], axis=-1).mean()) if mask.any() else None,
                race_closing_velocity_mean=float(along[mask].mean()) if mask.any() else None,
                race_heading_dot_mean=float(facing[mask].mean()) if mask.any() else None,
                race_heading_dot_min=float(facing[mask].min()) if mask.any() else None,
                race_airborne_fraction=float((data["pre.on_ground"][mask, world, car] == 0).mean()) if mask.any() else None,
                race_reverse_throttle_fraction=float((c[mask, 0] < 0).mean()) if mask.any() else None,
                distance_at_first_contact=float(distance[min(first, len(valid)-1)]),
                race_endpoint_speed=float(np.linalg.norm(vel[min(first, len(valid)-1)])),
                samples=[])
            for t in (0, 60, 120, 180, 240, 360, 480, 600):
                if t >= len(valid) or not valid[t]:
                    continue
                measurements["samples"].append(dict(tick=t, controls=c[t].tolist(),
                    distance=float(distance[t]), speed=float(np.linalg.norm(vel[t])),
                    closing_velocity=float(along[t]), heading_dot=float(facing[t]),
                    position=pos[t].tolist(), boost=float(data["pre.boost"][t, world, car]),
                    on_ground=int(data["pre.on_ground"][t, world, car]),
                    has_flipped=int(data["pre.has_flipped"][t, world, car]),
                    angular_velocity=data["pre.car_ang_vel"][t, world, car].tolist()))
            row["cars"].append(measurements)
        result.append(row)
    return result


def goal_prefix(raw, full, ticks=TICKS):
    checks = []
    for world, total in enumerate(full["match.goal_count"]):
        count = sum(0 <= t <= ticks for t in full["goal_tick"][world][:total])
        match = int(raw["match.goal_count"][world]) == count
        for k in ("goal_tick", "goal_scorer", "goal_overtime", "goal_kickoff", "goal_entry_valid", "goal_entry_x", "goal_entry_z"):
            match = match and np.array_equal(np.asarray(raw[k][world][:count]), np.asarray(full[k][world][:count]))
        checks.append(dict(world=world, expected_goals=count, actual_goals=int(raw["match.goal_count"][world]), exact=bool(match)))
    return checks


@torch.inference_mode()
def run_case(name, attempt=1):
    verify()
    case = CASES[name]
    if attempt != 1:
        assert name == "parent650" and attempt == 2
        assert (OUT / "serialization_recovery.json").exists()
    output_name = name if attempt == 1 else name + "_attempt2"
    assert not (OUT / (output_name + ".started.json")).exists(), "Do not silently rerun a case"
    with gpu_lease(), owned_match_stream():
        save_json(OUT / (output_name + ".started.json"), dict(utc=utc(), case=case, attempt=attempt,
            protocol_sha256=sha(OUT / "protocol.json"),
            recovery_sha256=sha(OUT / "serialization_recovery.json") if (OUT / "serialization_recovery.json").exists() else None))
        runner = TracedRunner(ROOT / case["path"], case["sha256"])
        nexto_before = tensor_hash(runner.nexto.actor.state_dict())
        elapsed = runner.run_ticks(TICKS).seconds
        trace = runner.trace.data
        action_mismatch, hidden_mismatch = 0, 0
        for i in range(TICKS // 4):
            actor, hidden = runner.rival_policy.forward_actor(trace["observation"][i], trace["hidden_before"][i])
            action_mismatch += int(torch.count_nonzero(runner.rival_policy.deterministic(actor) != trace["policy_action"][i]))
            hidden_mismatch += int(torch.count_nonzero(hidden != trace["hidden_after"][i]))
        arrays = {k: v.cpu().numpy().copy() for k, v in trace.items()}
        raw = {k: v.tolist() for k, v in runner.export()["raw"].items()}
        sides = np.asarray(raw["match.rival_side"])
        holds = arrays["policy_action"].repeat(4, axis=0)
        applied = arrays["applied_controls"][:, np.arange(10), sides]
        prefix = goal_prefix(raw, json.loads((ROOT / case["match"]).read_text())["raw"])
        checks = dict(finite=all(np.isfinite(v).all() for v in arrays.values()),
            tick_count=runner.host_tick == TICKS, decision_count=runner.trace.action_calls == TICKS // 4,
            exact_action_replay=action_mismatch == 0, exact_hidden_replay=hidden_mismatch == 0,
            exact_native_controls=np.array_equal(arrays["applied_controls"], arrays["scheduled_controls"]),
            exact_four_tick_holds=np.array_equal(applied, holds),
            rival_model_unchanged=tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before,
            nexto_model_unchanged=tensor_hash(runner.nexto.actor.state_dict()) == nexto_before,
            checkpoint_unchanged=sha(ROOT / case["path"]) == case["sha256"],
            goal_prefix_exact=all(v["exact"] for v in prefix), no_goal_overflow=not any(raw["goal_overflow"]))
        arrays["rival_side"] = sides
        arrays["starting_layout"] = np.asarray(raw["match.starting_layout"])
        path = OUT / (output_name + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        result = dict(version=VERSION, utc=utc(), case=case, wall_seconds=elapsed, checks=checks,
            protocol_sha256=sha(OUT / "protocol.json"), optimizer_steps=0,
            action_replay_differing_values=action_mismatch, hidden_replay_differing_values=hidden_mismatch,
            archive=dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                arrays={k: dict(shape=v.shape, dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest().upper()) for k,v in arrays.items()}),
            goal_prefix=prefix, raw_match=raw,
            physical=physical_summary(arrays, sides, arrays["starting_layout"]))
        if attempt == 2:
            with np.load(OUT / "parent650.npz", allow_pickle=False) as old:
                result["operational_retry_all_arrays_byte_equal"] = all(
                    old[k].dtype == v.dtype and old[k].shape == v.shape and old[k].tobytes() == v.tobytes()
                    for k, v in arrays.items()) and set(old.files) == set(arrays)
            checks["operational_retry_all_arrays_byte_equal"] = result["operational_retry_all_arrays_byte_equal"]
        save_json(OUT / (output_name + ".json"), result)
        print(json.dumps(dict(case=name, checks=checks, seconds=elapsed)), flush=True)
        del runner; gc.collect(); torch.cuda.empty_cache()
        assert all(checks.values()), "Trace integrity failure preserved; audit without silent rerun"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--attempt", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    else:
        if args.case is None:
            parser.error("--case required")
        run_case(args.case, args.attempt)
