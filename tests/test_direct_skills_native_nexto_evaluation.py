import ast
from pathlib import Path

import pytest

from benchmarks import evaluate_direct_skills_native_nexto_v1 as evaluation


def row(wins, goals, concedes):
    return dict(summary=dict(wins=wins, losses=10-wins, goals_for=goals,
                            goals_against=concedes, unresolved=0))


def test_parent_selection_is_frozen_before_results():
    choices = dict(reference600=row(1, 30, 100), finishing650=row(0, 100, 101))
    assert evaluation.select_parent(choices) == "reference600"
    choices["reference600"] = row(0, 35, 100)
    assert evaluation.select_parent(choices) == "finishing650"
    choices["reference600"] = choices["finishing650"]
    assert evaluation.select_parent(choices) == "finishing650"
    choices["reference600"]["summary"]["unresolved"] = 1
    with pytest.raises(ValueError): evaluation.select_parent(choices)


def test_controller_and_complete_match_semantics_are_explicit():
    spec = evaluation.specification()
    assert spec["mode"] == "native_v5" and spec["matches"] == 10
    assert spec["regulation_ticks"] == 36000 and spec["overtime_cap_ticks"] == 14400
    assert spec["optimizer_steps"] == 0
    assert set(spec["candidates"]) == {"reference600", "finishing650"}
    tree = ast.parse(Path(evaluation.__file__).read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("backward", "optimizer") for n in ast.walk(tree))
    text = ast.unparse(tree)
    assert "self.nexto = NextoNativeV5PolicyAdapter" in text
    assert "sampling_mode=MODE" in text
