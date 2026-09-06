"""No-learning native-play exports and real simulator sequence parity evidence."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from rivalsim.entity_native_bridge import BRIDGE_VERSION, EntityNativeActor, RecurrentPacketScheduler
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash

OUT = ROOT / "results/rival2/entity_native_bridge_v1"
ARTIFACTS = ROOT / "checkpoints/rival2/entity_native_bridge_v1"
SOURCES = {
    "reference600": ("checkpoints/rival2/direct_skills_v1/plus_000600.pt", "8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2"),
    "finishing650": ("checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt", "939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D"),
}
TICKS = 2400  # 20 seconds per layout/side; includes real native goal resets.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path, obj):
    assert not path.exists(), f"Preserve existing evidence: {path}"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def capture(label, checkpoint, digest):
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from benchmarks.evaluate_direct_skills_handbrake_v3 import verify

    verify()
    rows = []
    with gpu_lease(), owned_match_stream():
        class CaptureRunner(CandidateMatchRunner):
            @torch.inference_mode()
            def _update_rival_action(self):
                obs = self.rival_observation[self.batch_index, self.rival_side].cpu().numpy().copy()
                hidden_before = self.hidden.cpu().numpy().copy()
                reset_count = self.hidden_reset_count.cpu().numpy().copy()
                super()._update_rival_action()
                rows.append(dict(observation=obs, source_action=self.rival_action.cpu().numpy().copy(),
                                 hidden_before=hidden_before[0],
                                 hidden_after=self.hidden.cpu().numpy().copy()[0],
                                 reset_count=reset_count, frame=np.asarray(self.host_tick, dtype=np.int64)))
        runner = CaptureRunner(checkpoint, digest, entity=True)
        runner.run_ticks(TICKS)
        assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
        assert runner.hidden_reset_count.sum().item() > 0, "Need actual goal/reset coverage"
        arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        arrays["rival_side"] = runner.rival_side.cpu().numpy().copy()
        resets = np.concatenate((np.ones((1, 10), dtype=bool),
                                np.diff(arrays["reset_count"], axis=0) > 0), axis=0)
        arrays["reset_before"] = resets
        assert np.all(arrays["hidden_before"][resets] == 0)
        assert np.all(np.diff(arrays["frame"]) == 4)
        assert all(np.isfinite(a).all() for a in arrays.values())
        path = OUT / f"{label}_native_sequences.npz"
        assert not path.exists()
        np.savez_compressed(path, **arrays)
        del runner
        gc.collect()
    assert sha(checkpoint) == digest
    return arrays, path


def parity(model, exported, arrays):
    """Independent full sequential CPU trajectories, including batch-one runtime."""
    maxima = dict(export_logit_abs=0., export_hidden_abs=0., gpu_to_cpu_hidden_abs=0.)
    changed_actions = 0
    for world in range(10):
        source_hidden = model.initial_hidden(1, device="cpu")
        export_hidden = source_hidden.clone()
        scheduler = RecurrentPacketScheduler(exported, model.config.context_hidden_dim)
        epoch = 0
        with torch.inference_mode():
            for i, frame in enumerate(arrays["frame"]):
                obs = torch.from_numpy(arrays["observation"][i, world:world+1])
                if arrays["reset_before"][i, world]:
                    source_hidden = torch.zeros_like(source_hidden)
                    export_hidden = torch.zeros_like(export_hidden)
                    epoch += 1
                logits, source_hidden = model.forward_actor(obs, source_hidden)
                action, export_hidden, export_logits = exported(obs, export_hidden)
                torch.testing.assert_close(export_logits, logits, atol=2e-4, rtol=2e-6)
                torch.testing.assert_close(export_hidden, source_hidden, atol=2e-5, rtol=2e-6)
                expected = model.deterministic(logits)
                assert torch.equal(action, expected), (world, i, "CPU export action mismatch")
                changed_actions += int(not np.array_equal(action[0].numpy(), arrays["source_action"][i, world]))
                maxima["export_logit_abs"] = max(maxima["export_logit_abs"], float((export_logits-logits).abs().max()))
                maxima["export_hidden_abs"] = max(maxima["export_hidden_abs"], float((export_hidden-source_hidden).abs().max()))
                maxima["gpu_to_cpu_hidden_abs"] = max(maxima["gpu_to_cpu_hidden_abs"], float(np.abs(source_hidden[0, 0].numpy()-arrays["hidden_after"][i, world]).max()))
                scheduled = scheduler.step(frame=int(frame), identity=epoch, active=True,
                                           observation=lambda: obs, lifecycle=lambda d,r: None)
                assert torch.equal(scheduled, expected[0])
                for held in (1, 2, 3):
                    def forbidden():
                        raise AssertionError("Held tick requested observation")
                    actual = scheduler.step(frame=int(frame)+held, identity=epoch, active=True,
                                            observation=forbidden, lifecycle=lambda d,r: None)
                    assert torch.equal(actual, expected[0])
                torch.testing.assert_close(scheduler.hidden, export_hidden, atol=0, rtol=0)
        assert scheduler.stats["decisions"] == len(arrays["frame"])
        assert scheduler.stats["missed_ticks"] == scheduler.stats["skipped_decisions"] == 0
        assert scheduler.stats["resets"] == int(arrays["reset_before"][:, world].sum())
    assert changed_actions == 0, f"GPU source vs CPU export differing actions: {changed_actions}"
    return dict(**maxima, compared_decisions=int(arrays["observation"].shape[0]*10),
                exact_eight_channel_actions=True, gpu_to_cpu_action_differences=changed_actions,
                real_goal_resets=int(arrays["reset_before"][1:].sum()),
                held_tick_checks=int(arrays["observation"].shape[0]*30),
                all_five_layouts_both_sides=True, packet_domain_parity=False)


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    authority = OUT / "AUTHORITY.md"
    for p in (authority, Path(__file__), ROOT / "rivalsim/entity_native_bridge.py"):
        remote = subprocess.check_output(["git", "show", "origin/main:"+p.relative_to(ROOT).as_posix()])
        assert remote.replace(b"\r\n", b"\n") == p.read_bytes().replace(b"\r\n", b"\n")
    for label, (relative, digest) in SOURCES.items():
        checkpoint = ROOT / relative
        assert sha(checkpoint) == digest
        assert not (OUT / f"{label}.json").exists(), "No silent completed rerun"
        corpus = OUT / f"{label}_native_sequences.npz"
        # An interrupted CPU export may reuse only an identity-bound prior capture.
        capture_identity = OUT / f"{label}_capture.json"
        if corpus.exists():
            recorded = json.loads(capture_identity.read_text())
            assert recorded["checkpoint_sha256"] == digest and recorded["corpus_sha256"] == sha(corpus)
            with np.load(corpus, allow_pickle=False) as z:
                arrays = {k:z[k].copy() for k in z.files}
        else:
            arrays, corpus = capture(label, checkpoint, digest)
            write(capture_identity, dict(checkpoint_sha256=digest, corpus_sha256=sha(corpus),
                                        physics_ticks=TICKS, worlds=10, authority_sha256=sha(authority)))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = EntityJointControlActorCritic().cpu().eval()
        model.load_state_dict(payload["model"], strict=True)
        assert model.config.content_hash == payload["policy_config_sha256"]
        before = tensor_hash(model.state_dict())
        actor = EntityNativeActor(model).cpu().eval()
        script = torch.jit.script(actor)
        destination = ARTIFACTS / f"{label}.ts"
        assert not destination.exists(), "Never overwrite an export"
        torch.jit.save(script, destination)
        exported = torch.jit.load(str(destination), map_location="cpu").eval()
        print(json.dumps(dict(phase="cpu_sequential_parity", candidate=label)), flush=True)
        evidence = parity(model, exported, arrays)
        assert tensor_hash(model.state_dict()) == before == tensor_hash(payload["model"])
        assert sha(checkpoint) == digest
        write(OUT / f"{label}.json", dict(
            format=BRIDGE_VERSION, source=dict(path=relative, sha256=digest,
                model_sha256=before, policy_config_sha256=model.config.content_hash),
            artifact=dict(path=destination.relative_to(ROOT).as_posix(), sha256=sha(destination),
                          bytes=destination.stat().st_size),
            corpus=dict(path=corpus.relative_to(ROOT).as_posix(), sha256=sha(corpus)),
            contracts=dict(physics_hz=120, policy_hz=30, hold_ticks=4, observation_dim=182,
                           observation_schema_sha256=payload["observation_schema_sha256"],
                           action=payload["action_contract"], action_table_sha256=tensor_hash({"table":model.action_table})),
            hidden_shape=[1,1,model.config.context_hidden_dim],
            outputs=["eight_channel_action", "next_hidden", "90_logits"],
            parity=evidence, checkpoint_unchanged=True, optimizer_steps=0,
            installed=False, live_gameplay=False,
            torch_version=torch.__version__,
            compatibility="Scripted using existing repository PyTorch runtime. TorchScript deprecation warning recorded; target RLBot interpreter load still must be tested, not assumed.",
        ))
        print(json.dumps(dict(candidate=label, **evidence)), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
