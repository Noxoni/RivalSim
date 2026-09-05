"""CPU-only native training/full-match interface check, with no learning.

Does not replace CUDA full-match validation. Forced goals are test fixtures,
never counted as policy success or used as training data.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import warp as wp

from benchmarks.evaluate_rival2_ssl_entity_full_match import SEED, assignments
from benchmarks.inspect_fresh_ground_30hz_trajectories import CPUInspectionEnv
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_joint_control import CHECKPOINTS, COLLISION, RESULTS, verify
from rivalsim.full_match import FullMatchState, FullMatchTelemetry
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_FIELD_NAMES
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic

CHECKPOINT_SHA = "EDC6B3C01DF1BD17CD17A9F466FC6676414DB42A9EB5A6358A1C4DC123726983"


@torch.inference_mode()
def run(*, initial_only=False):
    torch.set_num_threads(2)
    verify()
    checkpoint = CHECKPOINTS / "plus_020.pt"
    assert sha(checkpoint) == CHECKPOINT_SHA
    policy = EntityJointControlActorCritic().eval()
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    before_hash = tensor_hash(policy.state_dict())
    layouts, sides = assignments()
    n = len(sides)
    kwargs = dict(device="cpu", seed=SEED, kickoff_selector=layouts, car_lifecycle_seed=SEED)
    train = CPUInspectionEnv(n, COLLISION, **kwargs)
    world = Rival2WorldSim(n, COLLISION, **kwargs)
    bridge = Rival2TensorBridge(world)
    match = FullMatchState(n, world.device, layouts, sides)
    views = match.torch_views()
    telemetry = FullMatchTelemetry(match)
    telemetry.attach(world)
    observation = bridge.observation()
    hidden_train = policy.initial_hidden(n * 2)
    hidden_match = hidden_train.clone()
    initial = []
    checks = []
    for decision in range(12):
        if decision == 6:
            # Same deliberate near-goal physical fixture in both worlds.
            # Native physics/lifecycle must generate and consume the event.
            for target in (train.bridge, bridge):
                target.views["ball_pos"][0] = torch.tensor([0.0, 5100.0, 120.0])
                target.views["ball_vel"][0] = torch.tensor([0.0, 6000.0, 0.0])
            train.observation = train.bridge.observation()
            observation = bridge.observation()
        difference = (train.observation - observation).abs()
        actor_train, hidden_train = policy.forward_actor(
            train.observation.reshape(-1, 182), hidden_train
        )
        actor_match, hidden_match = policy.forward_actor(observation.reshape(-1, 182), hidden_match)
        action = policy.deterministic(actor_train).reshape(n, 2, 8)
        match_action = policy.deterministic(actor_match).reshape(n, 2, 8)
        if decision == 0:
            for row, side in enumerate(sides):
                initial.append(
                    dict(
                        layout=int(layouts[row]),
                        side=int(side),
                        action=dict(zip(ACTION_NAMES, action[row, side].tolist(), strict=True)),
                    )
                )
            initial_report = dict(
                utc=utc(),
                verdict="PASS_INITIAL_ONLY"
                if float(difference.max()) == 0 and torch.equal(actor_train, actor_match)
                else "FAIL",
                checkpoint_sha256=CHECKPOINT_SHA,
                all_182_observation_fields_exact=torch.equal(train.observation, observation),
                actor_logits_exact=torch.equal(actor_train, actor_match),
                actions_exact=torch.equal(action, match_action),
                native_kickoff_actions=initial,
                device="cpu",
                physics_ticks_advanced=0,
                optimizer_steps=0,
                dynamic_reset_check="NOT_RUN",
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                model_unchanged=tensor_hash(policy.state_dict()) == before_hash,
                checkpoint_unchanged=sha(checkpoint) == CHECKPOINT_SHA,
                scope="Native initialized kickoff observation parity only, "
                "not dynamic CUDA gameplay",
                source_sha256=sha(__file__),
            )
            write_json(RESULTS / "full_match_initial_interface_check.json", initial_report)
            assert initial_report["verdict"] == "PASS_INITIAL_ONLY"
            if initial_only:
                print(initial_report, flush=True)
                return
        transition = train.step(action)
        world.begin_decision()
        for _ in range(4):
            bridge.set_actions(match_action)
            world.step(1)
        wp.copy(world.rival2.reset_mask, match.pending_reset)
        reset = (views["pending_reset"] != 0).clone()
        world.apply_interval_resets()
        telemetry.after_resets(world, world.rival2.reset_mask)
        observation = bridge.observation()
        hidden_train.masked_fill_(transition.reset_mask.repeat_interleave(2)[None, :, None], 0)
        hidden_match.masked_fill_(reset.repeat_interleave(2)[None, :, None], 0)
        mismatch_fields = difference.flatten(0, 1).amax(0) != 0
        checks.append(
            dict(
                decision=decision,
                observation_exact=torch.equal(train.observation, observation),
                pre_observation_max_difference=float(difference.max()),
                pre_mismatched_fields=[
                    name for name, bad in zip(OBS_FIELD_NAMES, mismatch_fields, strict=True) if bad
                ],
                actor_logits_exact=torch.equal(actor_train, actor_match),
                actions_exact=torch.equal(action, match_action),
                hidden_exact=torch.equal(hidden_train, hidden_match),
                reset_exact=torch.equal(transition.reset_mask, reset),
                goal_resets=int(reset.sum()),
            )
        )
    all_equal = all(
        row["observation_exact"]
        and row["pre_observation_max_difference"] == 0
        and row["actor_logits_exact"]
        and row["actions_exact"]
        and row["hidden_exact"]
        and row["reset_exact"]
        for row in checks
    )
    goals = sum(row["goal_resets"] for row in checks)
    unchanged = (
        sha(checkpoint) == CHECKPOINT_SHA and tensor_hash(policy.state_dict()) == before_hash
    )
    report = dict(
        utc=utc(),
        verdict="PASS" if all_equal and goals == 1 and unchanged else "FAIL",
        device="cpu",
        checkpoint_sha256=CHECKPOINT_SHA,
        optimizer_steps=0,
        cuda_allocated_bytes=torch.cuda.memory_allocated(),
        native_physics_ticks_per_world=48,
        forced_goal_fixture_resets=goals,
        checkpoint_and_model_unchanged=unchanged,
        initial_kickoff_actions=initial,
        decisions=checks,
        scope="CPU native observation/action/recurrent/goal-reset parity only; "
        "not CUDA parity or gameplay evidence",
        sources={
            str(Path(__file__).relative_to(ROOT)): sha(__file__),
            "benchmarks/inspect_fresh_ground_30hz_trajectories.py": sha(
                ROOT / "benchmarks/inspect_fresh_ground_30hz_trajectories.py"
            ),
        },
    )
    write_json(RESULTS / "full_match_cpu_interface_check.json", report)
    print(report["verdict"], "forced goals:", goals, "exact:", all_equal, flush=True)
    assert report["verdict"] == "PASS"
    assert report["cuda_allocated_bytes"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    try:
        run(initial_only=args.initial_only)
    except Exception:
        write_json(
            RESULTS / "full_match_cpu_interface_failure.json",
            dict(
                utc=utc(),
                verdict="NOT_COMPLETED",
                traceback=traceback.format_exc(),
                active_cuda_training_affected=False,
                physics_code_changed=False,
            ),
        )
        raise
