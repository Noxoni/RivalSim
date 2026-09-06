"""Fixed parent/latest comparison after the measured kickoff cache correction.

No optimization, candidate selection or replacement of old-method results.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, REGULATION_TICKS, OVERTIME_CAP_TICKS, summarize
from benchmarks.run_rival2_direct_skills_v1 import MATCH_RESET_VERSION, RESULTS, CHECKPOINTS, verify
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from benchmarks.report_rival2_ssl_entity_match_followup import reduce
from rivalsim.fresh_ground_30hz import content_hash


def run():
    package = verify(published=True)
    outputs = RESULTS / 'corrected_reset'
    outputs.mkdir(exist_ok=True)
    inputs = (
        (0, 'plus_000000.pt', 'DCD2ACC6203CF83228B28FCEBCADD487F93ED2B446466F29D883FF77DE523CD4'),
        (451, 'paused_000451.pt', 'B9D1BE5F75D22A9751832982A44B4A50F5BEE521DD4A1EE04C7640101EA8F324'),
    )
    with gpu_lease(), owned_match_stream():
        for offset, name, digest in inputs:
            checkpoint = CHECKPOINTS / name
            assert sha(checkpoint) == digest
            output = outputs / f'full_match_{offset:06d}.json'
            if output.exists():
                saved = json.loads(output.read_text())
                assert saved['checkpoint']['sha256'] == digest
                assert saved['match_reset_version'] == MATCH_RESET_VERSION
                assert saved['runtime_package_sha256'] == content_hash(package)
                assert all(reduce(output)['integrity'].values())
                continue
            runner = CandidateMatchRunner(checkpoint, digest, entity=True)
            elapsed = runner.run_ticks(REGULATION_TICKS).seconds
            for _ in range(OVERTIME_CAP_TICKS // 600):
                if bool(runner.phase_status()['done'].all()):
                    break
                elapsed += runner.run_ticks(600).seconds
            raw = runner.export()['raw']
            assert not raw['goal_overflow'].any()
            assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
            assert sha(checkpoint) == digest
            write_json(output, dict(
                utc=utc(), accepted_updates=offset,
                checkpoint=dict(path=str(checkpoint), sha256=digest, accepted_updates=offset),
                authority_sha256=package['authority_sha256'],
                runtime_package_sha256=content_hash(package), match_reset_version=MATCH_RESET_VERSION,
                summary=summarize(raw), raw={k:v.tolist() for k,v in raw.items()},
                wall_seconds=elapsed, hidden_resets=runner.hidden_reset_count.cpu().tolist(),
                optimizer_steps=0, model_unchanged=True, checkpoint_unchanged=True,
                interpretation='Corrected same-method parent/latest comparison; no checkpoint selection. '
                'Do not compare old-method scores as pure learning changes.',
            ))
            write_json(outputs / f'full_match_{offset:06d}_integrity.json', reduce(output))
            print(json.dumps(dict(offset=offset, summary=summarize(raw))), flush=True)
            del runner
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(8)
    run()
