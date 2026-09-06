"""Offline, one-factor native jump-timer diagnosis; never a deployment transform.

The actual simulator timer is zero during initial Jumping. After that phase,
RLBot's positive remaining dodge_timeout provides an independent timer basis.
Only those supported cases are probed; spent/expired/unknown timers are retained.
No policy, raw packet, production runtime, reward or physics mutation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/entity_native_jump_timer_v1"
SCALE = 1.25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def timer_probe(phase, jumped, doubled, dodged, remaining):
    """Return normalized semantic comparison or None (no guessed hidden timer)."""
    if phase == "Jumping":
        return 0.0, "initial_jump_zero"
    if (phase == "InAir" and jumped and not doubled and not dodged
            and np.isfinite(remaining) and 0 < remaining <= SCALE):
        return (SCALE - remaining) / SCALE, "positive_native_remaining"
    return None


def run():
    import torch
    import rlbot_flatbuffers as flat
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from report_entity_native_comparison import packets, read_records, CHANNELS
    from report_entity_native_reset_v2 import schema
    assert not OUT.exists(), "Completed diagnostic must not be overwritten"
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    manifest = json.loads((ROOT / "results/rival2/entity_native_packet_v1/reference600_manifest.json").read_text())
    names = [x["field"] for x in manifest["fields"]]
    timer_indices = [names.index(p + ".air_time_since_jump") for p in ("self", "opponent")]
    dodge_indices = [names.index(p + ".dodge_available") for p in ("self", "opponent")]
    artifact = ROOT / manifest["artifact"]["path"]
    parent = ROOT / manifest["source"]["path"]
    assert sha(parent) == manifest["source"]["sha256"]
    assert sha(artifact) == manifest["artifact"]["sha256"]
    actor = torch.jit.load(str(artifact), map_location="cpu").eval()
    source_paths = [Path(__file__), parent, artifact,
                    ROOT / "deployment/entity_native_v1/packet_observation.py",
                    ROOT / "rivalsim/kernels/vehicle.py",
                    ROOT / "rivalsim/rival2_env.py"]
    pyi = Path(flat.__file__).with_name("__init__.pyi")
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    OUT.mkdir()
    cases = {}
    for case in ("reference600_blue", "reference600_orange"):
        folder = ROOT / "results/rival2/entity_native_reset_v2" / case
        ds = read_records(folder / "decisions.bin.gz", schema.DECISION)
        observed = ds["observation"].copy()
        modified = observed.copy()
        supported = np.zeros((len(ds), 2), np.uint8)
        remaining = np.full((len(ds), 2), np.nan, np.float32)
        phase_names = {"OnGround": 0, "Jumping": 1, "InAir": 2,
                       "Dodging": 3, "DoubleJumping": 4}
        phases = np.zeros((len(ds), 2), np.uint8)
        flags = np.zeros((len(ds), 2, 3), np.uint8)
        for i, packet in enumerate(packets(folder / "decision_packets.bin.gz")):
            assert i < len(ds) and packet.match_info.frame_num == ds["frame"][i]
            players = sorted(packet.players, key=lambda c: int(c.team) != int(ds["team"][i]))
            assert len(players) == 2 and players[0].team == ds["team"][i]
            for car, player in enumerate(players):
                phase = str(player.air_state).split(".")[-1]
                phases[i, car] = phase_names[phase]
                flags[i, car] = [player.has_jumped, player.has_double_jumped, player.has_dodged]
                remaining[i, car] = player.dodge_timeout
                probe = timer_probe(phase, *flags[i, car], remaining[i, car])
                if probe is not None:
                    value, rule = probe
                    supported[i, car] = 1 if rule == "initial_jump_zero" else 2
                    modified[i, timer_indices[car]] = value
                    modified[i, dodge_indices[car]] = float(
                        not player.has_double_jumped and not player.has_dodged
                        and player.has_jumped and value < 1)
        assert i + 1 == len(ds)
        hidden_base = torch.zeros(1, 1, 256)
        hidden_probe = torch.zeros_like(hidden_base)
        base_actions, paired_actions, recurrent_actions = [], [], []
        previous = None
        with torch.inference_mode():
            for i, reset in enumerate(ds["resets"]):
                if previous != reset:
                    hidden_base.zero_(); hidden_probe.zero_(); previous = reset
                original = torch.from_numpy(observed[i:i+1].copy())
                probe = torch.from_numpy(modified[i:i+1].copy())
                # Same pre-step hidden state isolates immediate input sensitivity.
                paired, _, logits = actor(probe, hidden_base.clone())
                base, hidden_base, base_logits = actor(original, hidden_base)
                recurrent, hidden_probe, recurrent_logits = actor(probe, hidden_probe)
                assert all(torch.isfinite(x).all() for x in
                           (logits, base_logits, recurrent_logits, hidden_base, hidden_probe))
                base_actions.append(base[0].numpy())
                paired_actions.append(paired[0].numpy())
                recurrent_actions.append(recurrent[0].numpy())
        base_actions = np.asarray(base_actions)
        paired_actions = np.asarray(paired_actions)
        recurrent_actions = np.asarray(recurrent_actions)
        assert np.array_equal(base_actions, ds["action"])
        diff = modified - observed
        assert not set(np.flatnonzero(np.any(diff != 0, axis=0))) - set(timer_indices + dodge_indices)
        error_seconds = (observed[:, timer_indices] - modified[:, timer_indices]) * SCALE
        summary = dict(decisions=len(ds), original_action_replay_exact=True,
                       modified_fields=[names[i] for i in timer_indices + dodge_indices],
                       changed_observation_rows=int(np.any(diff != 0, axis=1).sum()),
                       paired_same_hidden_action_changes=int(np.any(base_actions != paired_actions, axis=1).sum()),
                       sequential_action_changes=int(np.any(base_actions != recurrent_actions, axis=1).sum()),
                       paired_per_channel_changes={k:int((base_actions[:,j] != paired_actions[:,j]).sum()) for j,k in enumerate(CHANNELS)},
                       roles={})
        for j, role in enumerate(("self", "opponent")):
            summary["roles"][role] = {}
            for code, label in ((1, "initial_jump_zero"), (2, "positive_native_remaining")):
                selected = supported[:, j] == code
                err = error_seconds[selected, j]
                summary["roles"][role][label] = dict(
                    samples=int(selected.sum()),
                    disagreement_over_1ms=int((np.abs(err) > .001).sum()),
                    mean_signed_seconds=float(err.mean()) if len(err) else None,
                    max_abs_seconds=float(np.abs(err).max()) if len(err) else None)
            summary["roles"][role]["dodge_available_changed"] = int((diff[:, dodge_indices[j]] != 0).sum())
        np.savez_compressed(OUT / (case + ".npz"), frame=ds["frame"], resets=ds["resets"],
                            supported=supported, remaining=remaining, phases=phases, flags=flags,
                            recorded_timer=observed[:, timer_indices], probed_timer=modified[:, timer_indices],
                            recorded_dodge=observed[:, dodge_indices], probed_dodge=modified[:, dodge_indices],
                            original_action=base_actions, paired_action=paired_actions,
                            sequential_action=recurrent_actions)
        summary["hashes"] = {str(p.relative_to(ROOT)):sha(p) for p in
                              (folder/"decisions.bin.gz",folder/"decision_packets.bin.gz",OUT/(case+".npz"))}
        cases[case] = summary
        print(json.dumps({"case":case, **{k:v for k,v in summary.items() if k not in ("roles","hashes")}}),flush=True)
    for path, identity in source_hashes.items(): assert sha(ROOT/path) == identity
    assert not torch.cuda.is_initialized()
    audit = dict(cases=cases, source_hashes=source_hashes,
                 installed_schema=dict(path=str(pyi),sha256=sha(pyi)),
                 timer_scale_seconds=SCALE, phase_codes=phase_names,
                 cpu_only=True, optimizer_steps=0, production_changed=False,
                 interpretation="One-factor offline sensitivity, not a corrected deployed actor or gameplay forecast. Recorded trajectory and original previous actions retained even when counterfactual actions differ. Spent/expired/unsupported timers unchanged; no full native-observation exactness claim. No quality-mask promotion.")
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",newline="\n")


if __name__ == "__main__":
    run()
