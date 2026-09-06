"""Real CUDA regression for skill-env -> captured match -> skill-env handoff."""

import gc

import pytest
import torch
import warp as wp

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner
from benchmarks.run_rival2_direct_skills_v1 import COLLISION, PARENT, PARENT_SHA
from rivalsim.direct_skills_v1 import DirectSkillsEnv, scenarios


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_skill_to_match_stream_and_return():
    torch.set_num_threads(4)
    before = torch.cuda.current_stream()
    for _ in range(2):
        for family in range(5):
            env = DirectSkillsEnv(
                8,
                COLLISION,
                device="cuda:0",
                seed=77,
                ssl_foundation_scenarios=scenarios(8, 77, family_only=family),
            )
            tr = env.step(torch.zeros((8, 2, 8), device="cuda:0"))
            assert torch.isfinite(tr.observation).all()
            del tr, env
            gc.collect()
        previous = wp.get_stream("cuda:0")
        with owned_match_stream() as stream:
            runner = CandidateMatchRunner(PARENT, PARENT_SHA, entity=True)
            assert runner.warp_stream is stream and stream.owner
            runner.run_ticks(120)
            assert torch.isfinite(runner.rival_action).all()
            assert torch.isfinite(runner.hidden).all()
            del runner
        assert torch.cuda.current_stream() == before
        assert wp.get_stream("cuda:0") is previous
    # Failure paths also restore stream ownership rather than contaminating PPO.
    with pytest.raises(ValueError, match="sentinel"), owned_match_stream():
        raise ValueError("sentinel")
    assert torch.cuda.current_stream() == before
    assert wp.get_stream("cuda:0") is previous
