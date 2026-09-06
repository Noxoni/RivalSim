"""Bounded, no-learning GPU/controller/collector validation; never starts PPO."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/rival2/nexto_native_controller_v1"
CHECKPOINT = ROOT / "checkpoints/rival2/direct_skills_v1/plus_000600.pt"
CHECKPOINT_SHA = "8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2"
SOURCES = (
    "third_party/nexto/native_v5.py", "third_party/nexto/adapter.py",
    "rivalsim/direct_skills_native_nexto.py", "rivalsim/direct_skills_training.py",
    "rivalsim/ssl_entity_mixed_training.py", "rivalsim/direct_skills_v1.py",
    "rivalsim/fresh_ground_30hz.py", "rivalsim/rival2_env.py",
    "rivalsim/ssl_entity_policy.py", "rivalsim/ssl_joint_control_policy.py",
    "benchmarks/run_rival2_ssl_entity_joint_control.py",
    "benchmarks/validate_nexto_native_v5_gpu.py", "tests/test_nexto_native_v5.py",
    "benchmarks/direct_skills_eval_stream.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
    "results/rival2/nexto_native_controller_v1/oracle.json",
    "results/rival2/nexto_native_controller_v1/oracle.npz",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def prepare():
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    write(OUT / "gpu_protocol.json", dict(
        version="NEXTO_NATIVE_V5_GPU_VALIDATION_V1",
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=CHECKPOINT_SHA,
        optimizer_steps=0, backward_calls=0,
        corpus="Existing native controller oracle, existing fixed kickoff starts and direct-skills bank",
        tests=["CUDA oracle 512 ticks x 8 worlds with mid-cadence restore",
               "Real pinned CUDA model, both sampling modes, masked 17-tick exact RNG/control replay",
               "Ten-world deterministic 20-second integration smoke; new native admission semantics",
               "32768-world, 90-decision real collector, native_v5 mode, no optimizer or backward"],
        limits="Bounded correctness smoke, not a new capability benchmark or training campaign. "
               "No all-native observation/physics parity claim. Preserve legacy runners unchanged.",
        nexto_seed=2026090691, sources={p: sha(ROOT / p) for p in SOURCES},
    ))


def equal_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    else:
        assert a == b


def serialized(value):
    import torch
    f = io.BytesIO()
    torch.save(value, f)
    f.seek(0)
    return torch.load(f, weights_only=False)


def run():
    import numpy as np
    import torch
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, summarize
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
    from benchmarks.run_rival2_ssl_entity_joint_control import COLLISION
    from rivalsim.direct_skills_native_nexto import DirectSkillNativeNextoCollector
    from rivalsim.direct_skills_v1 import DirectSkillsEnv, SEED, scenarios
    from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
    from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter, SAMPLING_MODES

    protocol = json.loads((OUT / "gpu_protocol.json").read_text())
    assert not (OUT / "gpu_run_started.json").exists(), "Do not silently repeat a started validation"
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    for p, h in protocol["sources"].items(): assert sha(ROOT / p) == h, p
    for p in (*SOURCES, "results/rival2/nexto_native_controller_v1/gpu_protocol.json"):
        subprocess.check_call(["git", "ls-files", "--error-unmatch", p], cwd=ROOT, stdout=subprocess.DEVNULL)
    write(OUT / "gpu_run_started.json", dict(commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), protocol_sha256=sha(OUT / "gpu_protocol.json")))
    torch.set_num_threads(2)
    result = dict(optimizer_steps=0, backward_calls=0, protocol_sha256=sha(OUT / "gpu_protocol.json"))
    with gpu_lease(), owned_match_stream():
        spec = importlib.util.spec_from_file_location("native_oracle_test", ROOT / "tests/test_nexto_native_v5.py")
        oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
        result["cuda_oracle_world_ticks"] = oracle.replay_oracle("cuda:0", resume_at=87)
        print("CUDA native oracle exact", flush=True)

        runner = CandidateMatchRunner(CHECKPOINT, CHECKPOINT_SHA, entity=True)
        replay = {}
        for mode in SAMPLING_MODES:
            adapter = NextoNativeV5PolicyAdapter(10, device=runner.device, sampling_mode=mode,
                                                seed=protocol["nexto_seed"])
            adapter.set_player_index(runner.nexto_side)
            active = torch.arange(10, device=runner.device) % 3 != 0
            kickoff = torch.arange(10, device=runner.device) % 2 == 0
            for _ in range(13): adapter.tick_action(runner.nexto_state, kickoff, active)
            saved = serialized(adapter.checkpoint_state())
            def sequence():
                records = []
                for tick in range(17):
                    if tick == 5: adapter.activate(~active)
                    if tick == 9: adapter.notify_kickoff(active)
                    mask = ~active if tick % 3 == 0 else active
                    action, index = adapter.tick_action(runner.nexto_state, kickoff, mask)
                    records.append((action.clone(), None if index is None else index.clone()))
                return records, adapter.checkpoint_state()
            a, state_a = sequence()
            adapter.load_checkpoint_state(saved)
            b, state_b = sequence()
            for left, right in zip(a, b, strict=True):
                for x, y in zip(left, right, strict=True): equal_tree(x, y)
            equal_tree(state_a, state_b)
            replay[mode] = dict(exact_controls_indices_state_and_rng=True, physics_calls=17,
                                state_sha256=tensor_hash(state_a["controller"]["tensors"]))
            assert all(not p.requires_grad for p in adapter.actor.parameters())
            del adapter
        result["pinned_model_checkpoint_replay"] = replay
        print("Real CUDA model masked replay exact in both modes", flush=True)

        runner.nexto = NextoNativeV5PolicyAdapter(10, device=runner.device,
            sampling_mode="deterministic_argmax", seed=protocol["nexto_seed"])
        runner.nexto.set_player_index(runner.nexto_side)
        actions, poses = [], []
        started = time.monotonic()
        for _ in range(2400):
            runner.tick()
            actions.append(runner.actions.detach().cpu().numpy().copy())
            if runner.host_tick % 4 == 0:
                poses.append(torch.cat((runner.nexto_state.car_pos.flatten(1),runner.nexto_state.ball_pos),1).cpu().numpy().copy())
        raw = runner.export()["raw"]
        np.savez_compressed(OUT / "gpu_match_smoke.npz", actions=np.stack(actions), poses=np.stack(poses))
        assert np.isfinite(actions).all() and np.isfinite(poses).all()
        assert not bool(raw["goal_overflow"].any())
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        result["match_smoke"] = dict(summary=summarize(raw), physics_ticks=2400,
            mode="deterministic_argmax", wall_seconds=time.monotonic()-started,
            raw={k: v.tolist() for k,v in raw.items()}, trace_sha256=sha(OUT / "gpu_match_smoke.npz"))
        write(OUT / "gpu_partial.json", result)
        print("20-second opt-in production match smoke complete", flush=True)
        del runner; gc.collect(); torch.cuda.empty_cache()

        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = EntityJointControlActorCritic().cuda().eval()
        model.load_state_dict(payload["model"], strict=True)
        before = tensor_hash(model.state_dict())
        del payload
        # No optimizer is constructed, no gradients requested, no model mutation.
        env = DirectSkillsEnv(32768, COLLISION, device="cuda:0", seed=SEED,
                              ssl_foundation_scenarios=scenarios(32768))
        collector = DirectSkillNativeNextoCollector(env, model, seed=SEED,
            nexto_sampling_mode="native_v5", nexto_seed=protocol["nexto_seed"])
        nexto_before = tensor_hash(collector.nexto.actor.state_dict())
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        rollout = collector.collect()
        torch.cuda.synchronize()
        metrics = collector.last_metrics
        assert rollout.horizon == 90 and rollout.num_envs == 32768
        assert metrics["trainable_agent_samples"] == 32768 * 90 * 3 // 2
        assert metrics["nexto_training_sample_count"] * 3 == metrics["trainable_agent_samples"]
        assert tensor_hash(model.state_dict()) == before
        assert tensor_hash(collector.nexto.actor.state_dict()) == nexto_before
        assert all(p.grad is None for p in model.parameters())
        assert not bool(collector.nexto.controller.pending_action[~collector.is_nexto].any())
        assert not bool(collector.nexto.controller.emitted_control[~collector.is_nexto].any())
        saved = serialized(collector.opponent_checkpoint_state())
        # Test exact adapter load in the same live physical context, not an env restore.
        collector.nexto.load_checkpoint_state(saved["native_nexto"])
        equal_tree(collector.nexto.checkpoint_state(), saved["native_nexto"])
        result["full_scale_collector"] = dict(worlds=32768, decisions=90,
            wall_seconds=time.monotonic()-started, metrics=metrics,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            actor_unchanged=True, nexto_unchanged=True, no_gradients=True,
            inactive_world_controls_zero=True, serialized_controller_and_rng_exact=True,
            model_tensor_sha256=before)
        print("32768-world no-learning collector complete", flush=True)
    for p, h in protocol["sources"].items(): assert sha(ROOT / p) == h, p
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    result["checkpoint_unchanged"] = True
    write(OUT / "gpu_results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    (prepare if parser.parse_args().mode == "prepare" else run)()
