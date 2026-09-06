"""Read-only replay of the frozen skill evaluation, with a finishing state tap.

Never launch beside the learner: the existing exclusive GPU lease is mandatory.
No optimizer is constructed. No reward, scenario, policy or evaluation code is
replaced. Saved evaluation parity is required before a trace can be interpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "RIVAL2_DIRECT_SKILLS_FINISHING_TRACE_V1"
CAR_FIELDS = (
    "car_pos",
    "car_vel",
    "car_quat",
    "car_ang_vel",
    "boost",
    "on_ground",
    "has_jumped",
    "is_jumping",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "jump_time",
    "air_time",
    "air_time_since_jump",
    "flip_time",
    "is_boosting",
    "wheel_contact",
    "car_is_demoed",
    "demo_respawn_timer",
)
BALL_FIELDS = ("ball_pos", "ball_vel", "ball_ang_vel")


@contextmanager
def temporary_attribute(obj, name, replacement):
    """Restore the original instance/class binding even when the tap fails."""
    owned = name in vars(obj)
    original = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


class BoundedFrames:
    def __init__(self, maximum):
        self.maximum = maximum
        self.rows = []
        self.schema = None

    def append(self, row):
        if len(self.rows) >= self.maximum:
            raise RuntimeError("trace exceeds prospective frame bound")
        # Copy immediately: native views are reused at the next physics tick.
        row = {key: np.array(value, copy=True) for key, value in row.items()}
        schema = {key: (value.shape, value.dtype.str) for key, value in row.items()}
        if self.schema is not None and schema != self.schema:
            raise RuntimeError("trace shape/dtype/field schema changed")
        if any(value.dtype.hasobject for value in row.values()):
            raise ValueError("object arrays are not a native trace representation")
        self.schema = schema
        self.rows.append(row)

    def arrays(self, prefix):
        if not self.rows:
            raise RuntimeError("no trace frames")
        return {prefix + key: np.stack([row[key] for row in self.rows]) for key in self.schema}


def differences(expected, actual, path="$", limit=32):
    """Exact comparison, including tiny float differences; never relax replay parity."""
    if type(expected) is not type(actual):
        return [path + ": type differs"]
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [path + ": keys differ"]
        found = []
        for key in expected:
            found.extend(differences(expected[key], actual[key], path + "." + key, limit))
            if len(found) >= limit:
                break
        return found[:limit]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [path + ": length differs"]
        found = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            found.extend(differences(left, right, f"{path}[{index}]", limit))
            if len(found) >= limit:
                break
        return found[:limit]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def verify_trace_arrays(arrays, expected, focal_side):
    """Independent accounting from original-case masks, not reward success labels."""
    active = arrays["decision.alive"]
    count, worlds = active.shape
    if count > 360 or worlds != len(focal_side):
        raise ValueError("unexpected bounded trace dimensions")
    if arrays["physics.decision"].shape != (4 * count,):
        raise ValueError("missing or duplicate physics records")
    np.testing.assert_array_equal(arrays["physics.decision"], np.repeat(np.arange(count), 4))
    np.testing.assert_array_equal(arrays["physics.tick"], np.tile(np.arange(4), count))
    if not active[0].all():
        raise ValueError("initial cases must all be active")
    reset = arrays["decision.reset_mask"]
    np.testing.assert_array_equal(active[1:], active[:-1] & ~reset[:-1])
    if (active[-1] & ~reset[-1]).any():
        raise ValueError("original cases not completed")
    rows = np.arange(worlds)
    touch = arrays["decision.native.touch_count"][:, rows, focal_side] * active
    scored = arrays["decision.native.scoring_team"] == focal_side[None]
    goal = arrays["decision.terminated"] & active
    raw = dict(
        touches=touch.sum(0).astype(float).tolist(),
        goals=(goal & scored).sum(0).astype(float).tolist(),
        concedes=(goal & ~scored).sum(0).astype(float).tolist(),
        endings=(
            (
                arrays["decision.terminated"].astype(int)
                + 2 * arrays["decision.truncated"].astype(int)
            )
            * active
        )
        .sum(0)
        .tolist(),
    )
    for key, value in raw.items():
        if value != expected["raw"][key]:
            raise ValueError("trace disagrees with saved raw " + key)
    for key, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError("nonfinite trace: " + key)
    # Native per-decision counters reset at begin_decision, not per physics tick.
    cumulative = arrays["physics.post.interval_touch_count"].reshape(count, 4, worlds, 2)
    prior = np.concatenate((np.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), axis=1)
    valid = arrays["physics.valid"].reshape(count, 4, worlds)
    reconstructed = (np.maximum(cumulative - prior, 0) * valid[..., None]).sum(1)
    np.testing.assert_array_equal(
        reconstructed * active[..., None], arrays["decision.native.touch_count"] * active[..., None]
    )
    return dict(
        original_cases=worlds,
        decisions=count,
        physics_records=4 * count,
        native_contact_accounting=True,
        original_case_outcomes=True,
        ordered_complete_ticks=True,
        finite=True,
    )


def install_finishing_tap(campaign, physics, decisions, identity):
    """Return a subclass that delegates every transition to the frozen environment."""
    import torch
    import warp as wp

    base = campaign.DirectSkillsEnv

    class TracedEnvironment(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace_enabled = bool((self.family == 2).all())
            if not self.trace_enabled:
                return
            if identity:
                raise RuntimeError("more than one finishing batch")
            identity.update(focal_side=self.focal.cpu().tolist(), worlds=self.num_envs)
            self.trace_alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.trace_hits = [
                wp.to_torch(self.world.car_ball.hit_this_tick),
                wp.to_torch(self.world.car_ball_b.hit_this_tick),
            ]

        def native(self):
            n, views = self.num_envs, self.bridge.views
            row = {
                name: views[name].reshape(n, 2, *views[name].shape[1:]).cpu().numpy()
                for name in CAR_FIELDS
            }
            row.update({name: views[name].cpu().numpy() for name in BALL_FIELDS})
            row.update(
                applied_action=views["rival2.previous_action"].reshape(n, 2, 8).cpu().numpy(),
                interval_touch_count=views["rival2.touch_count"].reshape(n, 2).cpu().numpy(),
                goal_latched=self.goal_latched.cpu().numpy(),
                scoring_team=views["rival2.scoring_team_latched"].cpu().numpy(),
                hit_this_tick=torch.stack(self.trace_hits, -1).cpu().numpy(),
            )
            return row

        def _step_impl(self, action, markers=None, tick_action_provider=None):
            if not self.trace_enabled:
                return super()._step_impl(action, markers, tick_action_provider)
            index, tick = len(decisions.rows), 0
            # Copy before _step_impl can recycle the observation buffer.
            row = dict(
                observation=self.observation.cpu().numpy().copy(),
                proposed_rival_action=action.cpu().numpy().copy(),
                alive=self.trace_alive.cpu().numpy().copy(),
            )
            original = self.world.step

            def observed_step(*args, **kwargs):
                nonlocal tick
                if args != (1,) or kwargs or tick >= 4:
                    raise RuntimeError("frozen four-tick stepping semantics changed")
                valid = (self.trace_alive & (self.goal_latched == 0)).cpu().numpy().copy()
                # .copy is needed here even on CPU fake fixtures/native aliases.
                before = {key: value.copy() for key, value in self.native().items()}
                result = original(*args, **kwargs)
                record = dict(decision=index, tick=tick, valid=valid)
                record.update({"pre." + key: value for key, value in before.items()})
                record.update({"post." + key: value for key, value in self.native().items()})
                physics.append(record)
                tick += 1
                return result

            with temporary_attribute(self.world, "step", observed_step):
                transition = super()._step_impl(action, markers, tick_action_provider)
            if tick != 4:
                raise RuntimeError("missing physics tick")
            for name in (
                "transition_observation",
                "reward",
                "terminated",
                "truncated",
                "reset_mask",
            ):
                row[name] = getattr(transition, name).cpu().numpy()
            row.update(
                {"native." + key: value.cpu().numpy() for key, value in self.last_native.items()}
            )
            row.update(
                {"skill." + key: value.cpu().numpy() for key, value in self.last_skill.items()}
            )
            decisions.append(row)
            self.trace_alive &= ~transition.reset_mask
            return transition

    return TracedEnvironment


def run(update, output):
    # Importing the production module does not instantiate its optimizer/run path.
    import torch

    from benchmarks import run_rival2_direct_skills_v1 as campaign

    package = campaign.verify(published=True)
    checkpoint = campaign.CHECKPOINTS / f"plus_{update:06d}.pt"
    evaluation = campaign.RESULTS / f"evaluation_{update:06d}.json"
    expected = json.loads(evaluation.read_text())
    digest = campaign.sha(checkpoint)
    if (
        expected["checkpoint"]["sha256"] != digest
        or expected["accepted_updates"] != update
        or expected["authority_sha256"] != package["authority_sha256"]
        or expected["optimizer_steps"] != 0
    ):
        raise ValueError("checkpoint/evaluation/authority identity mismatch")
    if output.exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    # Acquire before creating CUDA objects or an output directory. Refusal leaves
    # the learner and its STOP/checkpoints/config entirely untouched.
    with campaign.gpu_lease():
        output.mkdir(parents=True, exist_ok=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            payload["accepted_updates"] != update
            or payload["authority_sha256"] != package["authority_sha256"]
        ):
            raise ValueError("checkpoint metadata mismatch")
        model = campaign.EntityJointControlActorCritic().cuda().eval()
        model.load_state_dict(payload["model"], strict=True)
        if model.config.content_hash != payload["policy_config_sha256"]:
            raise ValueError("model config mismatch")
        model_hash = campaign.tensor_hash(model.state_dict())
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu())
        del payload  # Adam is provenance only; never instantiate/restore/use it.
        physics, decisions, identity = BoundedFrames(1440), BoundedFrames(360), {}
        traced = install_finishing_tap(campaign, physics, decisions, identity)
        with temporary_attribute(campaign, "DirectSkillsEnv", traced), torch.no_grad():
            actual = campaign.skill_evaluation(model)
        arrays = {**physics.arrays("physics."), **decisions.arrays("decision.")}
        arrays["focal_side"] = np.asarray(identity["focal_side"], dtype=np.int64)
        arrays["case_index"] = np.arange(identity["worlds"], dtype=np.int64)
        path = output / "finishing.npz"
        np.savez_compressed(path, **arrays)
        # Verify the physical file, including dtype/shape, not just the in-memory copy.
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise ValueError("archive key mismatch")
            for key in arrays:
                if loaded[key].dtype != arrays[key].dtype:
                    raise ValueError("archive dtype mismatch")
                np.testing.assert_array_equal(loaded[key], arrays[key])
        diff = differences(expected["skills"], actual)
        unchanged = (
            campaign.sha(checkpoint) == digest
            and campaign.tensor_hash(model.state_dict()) == model_hash
        )
        checks = verify_trace_arrays(arrays, actual["finishing"], arrays["focal_side"])
        manifest = dict(
            schema=SCHEMA,
            utc=campaign.utc(),
            accepted_updates=update,
            verdict="PASS" if not diff and unchanged else "BLOCKED_REPLAY_MISMATCH",
            checkpoint_sha256=digest,
            evaluation_sha256=campaign.sha(evaluation),
            authority_sha256=package["authority_sha256"],
            runtime_sources=package["sources"],
            trace_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
            trace_sha256=campaign.sha(path),
            trace_bytes=path.stat().st_size,
            exact_saved_evaluation_parity=not diff,
            differences=diff,
            model_and_checkpoint_unchanged=unchanged,
            optimizer_steps=0,
            checks=checks,
            replay=actual,
            semantics={
                "method": "Unchanged fixed 5x64 skill evaluation; finishing-only state tap",
                "axis": "time, original case, car(Blue0/Orange1); focal_side identifies Rival",
                "units": "native uu, uu/s, rad/s, seconds; quaternion xyzw; normalized182 obs",
                "policy": "30Hz deterministic joint90; 120Hz applied actions incl Nexto held15Hz",
                "proposed_rival_action": "Both Rival perspectives; Nexto replaces opponent later",
                "applied_action": "bridge previous_action: exact emitted controls before step",
                "valid": "original episode alive, no earlier goal in this four-tick decision",
                "boundaries": "post physics/pre-reset state; transition_observation before reset",
                "contacts": "hit_this_tick=raw flag; interval_touch_count=debounced native counter",
                "events": "Unmodified task proxies, not mechanic/possession adjudication",
                "limit": "64 finishing cases, <=360 decisions/1440 physics ticks; all retained",
                "interpretation": "Parity failure blocks diagnosis, not permission to retune PPO",
            },
            arrays={
                key: dict(shape=list(value.shape), dtype=value.dtype.str)
                for key, value in arrays.items()
            },
        )
        campaign.write_json(output / "manifest.json", manifest)
        if diff or not unchanged:
            raise RuntimeError(
                "read-only replay not equivalent; do not interpret trace as saved evaluation"
            )
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(8)
    print(json.dumps(run(args.update, args.output), indent=2))
