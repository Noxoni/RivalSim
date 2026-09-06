"""Read-only current checkpoint versus installed legacy RLBot loader, on CPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks import run_direct_skills_finishing_goal_v2 as branch
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,write_json,utc
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic

LIVE=Path('G:/dev/RLBot-Rival')
OUT=ROOT/'results/rival2/direct_skills_finishing_goal_v2/live_loader_readiness.json'
SOURCES=(LIVE/'scripts/export_rival2_unified_v5.py',
         LIVE/'bot/rival2_unified_v5/unified_runtime.py',
         LIVE/'bot/rival2_live/runtime.py')


def source_hashes():
    return {str(p):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest().upper() for p in SOURCES}


def run():
    assert not OUT.exists(), 'Preserve original audit; no overwrite'
    assert sha(branch.SOURCE)==branch.SOURCE_SHA
    sources=source_hashes()
    payload=torch.load(branch.SOURCE,map_location='cpu',weights_only=False)
    before=tensor_hash(payload['model'])
    model=EntityJointControlActorCritic().cpu().eval()
    model.load_state_dict(payload['model'],strict=True)
    assert model.config.content_hash==payload['policy_config_sha256']
    with torch.inference_mode():
        hidden=model.initial_hidden(1,device='cpu')
        logits,next_hidden=model.forward_actor(torch.zeros((1,182)),hidden)
        action=model.deterministic(logits)
    assert tuple(logits.shape)==(1,90) and tuple(action.shape)==(1,8)
    assert torch.isfinite(logits).all() and torch.isfinite(next_hidden).all()
    assert torch.equal(action,model.action_table[logits.argmax(-1)])
    spec=importlib.util.spec_from_file_location('inspected_legacy_unified_export',SOURCES[0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        module.load_checkpoint(ROOT,branch.SOURCE,branch.SOURCE_SHA)
        legacy_result=dict(accepted=True)
    except Exception as exc:
        legacy_result=dict(accepted=False,error_type=type(exc).__name__,error=str(exc))
    assert not legacy_result['accepted'], 'Unexpected compatibility requires review'
    assert source_hashes()==sources and sha(branch.SOURCE)==branch.SOURCE_SHA
    assert before==tensor_hash(model.state_dict())==tensor_hash(payload['model'])
    result=dict(
        utc=utc(),version='RIVAL2_DIRECT_SKILLS_LEGACY_LIVE_LOADER_AUDIT_V1',
        checkpoint=dict(path=branch.SOURCE.relative_to(ROOT).as_posix(),sha256=branch.SOURCE_SHA),
        source_hashes=sources,legacy_export_loader=legacy_result,
        exact_model_cpu_loading=True,logit_shape=list(logits.shape),hidden_shape=list(hidden.shape),
        action_shape=list(action.shape),joint90_argmax_table_exact=True,
        actual_runtime_requirements=dict(physics_hz=120,policy_hz=30,hold_ticks=4,
            decision='Joint90 argmax lookup, not tanh/BCE hybrid decoding',
            recurrent='Advance GRU only at policy decision; hold both GRU/action on other ticks; reset per real episode/kickoff'),
        installed_recurrent_runtime='Requires physics120/policy120/hold1 and legacy hybrid export; not compatible with this checkpoint.',
        installed_feedforward_runtime='Supports held action cadence but invokes model(observation) without recurrent state; also not this policy.',
        observation_scope='No full packet parity test here. Prior individually measured simulator wheel contacts versus aggregate packet ground-contact broadcast remains a separate documented domain gap; unavailable state must not be relabeled exact.',
        optimizer_steps=0,simulator_steps=0,external_files_changed=False,checkpoint_unchanged=True,
        model_unchanged=True,export_created=False,live_deployment=False,live_gameplay_evaluation=False,
        interpretation='Zero-input forward proves loader/output interface only, not meaningful controls or gameplay. A separate entity/joint90 recurrent 30Hz export/runtime and honest packet-observation validation are needed before native RLBot tests. No active training change.',
    )
    write_json(OUT,result)
    print(json.dumps(dict(legacy_export=legacy_result,model_unchanged=True,live_deployment=False)))


if __name__=='__main__':
    torch.set_num_threads(4)
    run()
