"""CPU-only same-layout input comparison and isolated wheel-input intervention.

The intervention changes copied observations, not the simulator or checkpoint.
It establishes first-decision sensitivity, not whole-match outcome causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.rival2_contracts import OBS_FIELD_NAMES  # noqa: E402
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def paired_indices(arrays):
    """Match each post-goal age-zero row to its initial same-side/layout row."""
    initial, later = {}, []
    for decision, world in zip(*np.where(arrays["eligible"] & (arrays["episode_ticks"] == 0))):
        side = int(arrays["rival_side"][decision, world])
        if arrays["match_goal_count"][decision, world] == 0:
            key = (int(arrays["match_starting_layout"][decision, world]), side)
            if key in initial:
                raise ValueError("duplicate initial kickoff")
            initial[key] = (int(decision), int(world), side)
        else:
            key = (int(arrays["lifecycle_layout"][decision, world]), side)
            later.append((key, (int(decision), int(world), side)))
    if not later:
        raise ValueError("no post-goal kickoff observations")
    return [(initial[key], row) for key, row in later]


def restore_wheel_inputs(observations, initial):
    columns = [i for i, name in enumerate(OBS_FIELD_NAMES) if ".wheel_contact." in name]
    if len(columns) != 8:
        raise ValueError("wheel observation contract changed")
    result = observations.clone()
    result[:, columns] = initial[:, columns]
    return result, columns


@torch.inference_mode()
def run(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["exact_saved_raw_summary_hidden_reset_parity"]
    assert manifest["model_checkpoint_unchanged"] and manifest["new_optimizer_steps"] == 0
    archive = directory / "kickoff_reset.npz"
    assert sha(archive) == manifest["archive_sha256"]
    checkpoint = ROOT / f'checkpoints/rival2/direct_skills_v1/plus_{manifest["accepted_updates"]:06d}.pt'
    assert sha(checkpoint) == manifest["checkpoint_sha256"]
    with np.load(archive, allow_pickle=False) as source:
        a = {key: source[key] for key in source.files}
    pairs = paired_indices(a)
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    policy = EntityJointControlActorCritic().eval()
    policy.load_state_dict(parent, strict=True)
    obs = [torch.from_numpy(np.stack([a["observation"][p[index]] for p in pairs])) for index in (0, 1)]
    hidden = [torch.from_numpy(np.stack([a["hidden_before"][p[index][0], :, p[index][1]] for p in pairs], axis=1)) for index in (0, 1)]
    assert all(not bool(h.any()) for h in hidden), "recurrent reset is not zero"
    patched, columns = restore_wheel_inputs(obs[1], obs[0])
    results = [policy.forward_actor(o, h)[0] for o, h in ((obs[0], hidden[0]), (obs[1], hidden[1]), (patched, hidden[1]))]
    actions = [policy.deterministic(logits) for logits in results]
    for i in (0, 1):
        recorded = torch.from_numpy(np.stack([a["rival_action"][p[i][0], p[i][1]] for p in pairs]))
        assert torch.equal(recorded, actions[i]), "CPU argmax differs from recorded native execution"
    assert all(torch.isfinite(x).all() for x in results)
    assert all(torch.equal(v, parent[k]) for k, v in policy.state_dict().items())
    assert sha(checkpoint) == manifest["checkpoint_sha256"]
    changed = (actions[0] != actions[1]).any(1)
    remaining = (actions[0] != actions[2]).any(1)
    delta = (obs[0] - obs[1]).abs()
    report = dict(
        schema="RIVAL2_DIRECT_SKILLS_KICKOFF_INPUT_ISOLATION_V1",
        manifest_sha256=sha(manifest_path), archive_sha256=sha(archive),
        checkpoint_sha256=sha(checkpoint), pairs=len(pairs),
        exact_cpu_recorded_action_parity=True, zero_hidden_before_every_pair=True,
        pairs_with_changed_action=int(changed.sum()),
        wheel_only_intervention_restores_initial_action=int((changed & ~remaining).sum()),
        intervention_remaining_action_differences=int(remaining.sum()),
        max_nonwheel_input_difference=float(delta[:, [i for i in range(182) if i not in columns]].max()),
        max_logit_difference_before=float((results[0] - results[1]).abs().max()),
        max_logit_difference_after=float((results[0] - results[2]).abs().max()),
        per_control_changed_count=(actions[0] != actions[1]).sum(0).tolist(),
        field_differences=[dict(field=OBS_FIELD_NAMES[i], max_abs=float(delta[:, i].max()),
                                changed_pairs=int((delta[:, i] != 0).sum()))
                           for i in range(182) if bool(delta[:, i].any())],
        pair_evidence=[dict(initial=list(p[0]), later=list(p[1]),
                            initial_action=actions[0][i].tolist(), recorded_later_action=actions[1][i].tolist(),
                            wheel_only_counterfactual_action=actions[2][i].tolist()) for i, p in enumerate(pairs)],
        model_checkpoint_unchanged=True, optimizer_steps=0,
        interpretation="Wheel-only copied-observation intervention isolates first-action sensitivity. "
        "Not a new match, measured scoring improvement, or proof that wheels explain every post-goal failure.",
    )
    (directory / "input_isolation.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pair_evidence", "field_differences")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    run(args.directory)
