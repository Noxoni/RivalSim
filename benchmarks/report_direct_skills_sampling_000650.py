"""CPU-only reduction of the three frozen650 sampling seeds, never new rollouts."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.report_rival2_direct_skills_matches import (  # noqa: E402
    checkpoint_integrity,
    match_rows,
    validate_comparability,
)
from benchmarks.report_rival2_ssl_entity_match_followup import reduce  # noqa: E402

RESULTS = ROOT / "results/rival2/direct_skills_v1"
OUT = RESULTS / "sampling_diagnostic_000650"


def text_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def spread(values):
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                values=values)


def build():
    authority = json.loads((OUT / "authority.json").read_text())
    package = json.loads((OUT / "package.json").read_text())
    assert not (OUT / "failure.json").exists(), "Preserve and investigate failed attempts"
    baseline_path = RESULTS / "full_match_000650.json"
    baseline = json.loads(baseline_path.read_text())
    baseline_skill = json.loads((RESULTS / "evaluation_000650.json").read_text())["skills"]["finishing"]
    checkpoint_integrity(baseline)
    baseline_reduction = reduce(baseline_path)
    baseline_rows = match_rows(baseline)
    rows = []
    checks = {}
    for seed in authority["sampling_seeds"]:
        path = OUT / f"seed_{seed}.json"
        source = json.loads(path.read_text())
        checkpoint_integrity(source)
        assert source["checkpoint"]["sha256"] == authority["checkpoint_sha256"]
        assert source["accepted_updates"] == 650 and source["sampling_seed"] == seed
        assert source["action_mode"] == authority["action_mode"]
        assert source["authority_sha256"] == package["authority_sha256"]
        assert source["match_reset_version"] == baseline["match_reset_version"]
        for key in ("match.rival_side", "match.starting_layout"):
            assert source["raw"][key] == baseline["raw"][key]
        skill = source["finishing"]
        assert skill["scenario_sha256"] == baseline_skill["scenario_sha256"]
        assert skill["worlds"] == 64 and skill["unfinished"] == 0
        assert skill["goals_for"] + skill["goals_against"] + skill["timeouts"] == 64
        assert sum(skill["raw"]["goals"]) == skill["goals_for"]
        assert sum(skill["raw"]["concedes"]) == skill["goals_against"]
        assert sum(skill["raw"]["touches"]) == skill["touches"]
        # This MUST be refused as a same-method learning comparison: action mode differs.
        try:
            validate_comparability(baseline, source)
        except ValueError as exc:
            assert "runtime_package_sha256" in str(exc), str(exc)
        else:
            raise AssertionError("Sampled outputs passed greedy learning comparator")
        reduction = reduce(path)
        checks[str(seed)] = reduction["integrity"] | dict(
            same_source_checkpoint=True, identical_physical_cases=True,
            complete_finishing_accounting=True, different_method_correctly_refused=True,
        )
        pairs = []
        for a, b in zip(baseline_rows, match_rows(source), strict=True):
            pairs.append(dict(match=b["match"], side=b["side"], layout=b["layout"],
                              greedy=a, sampled=b,
                              goal_difference_delta=b["goal_difference"] - a["goal_difference"]))
        rows.append(dict(seed=seed, source_text_sha256=text_sha(path),
                         summary=reduction["summary"], contacts=reduction["contacts"],
                         finishing=skill, per_match=pairs))
    assert len(rows) == 3
    aggregates = dict(
        matches={key:spread([r["summary"][key] for r in rows]) for key in
                 ("goals_for", "goals_against", "wins", "touches", "touches_per_minute")},
        finishing={key:spread([r["finishing"][key] for r in rows]) for key in
                   ("goals_for", "goals_against", "timeouts", "touches", "touched_worlds")},
        finishing_on_target_proxy=spread([r["finishing"]["events"]["on_target_touch"] for r in rows]),
        pooled_same_player_followup_fraction=(
            sum(r["contacts"]["same_player_followup"] for r in rows)
            / sum(r["contacts"]["resolved_followups"] for r in rows)),
    )
    return dict(schema="RIVAL2_DIRECT_SKILLS_650_SAMPLING_DIAGNOSTIC_REPORT_V1",
                authority_text_sha256=text_sha(OUT / "authority.json"),
                package_text_sha256=text_sha(OUT / "package.json"),
                checkpoint=baseline["checkpoint"], baseline=baseline_reduction,
                baseline_finishing=baseline_skill, seeds=rows, aggregate=aggregates,
                checks=checks, optimizer_steps=0, model_selection=False,
                inference_mode_changed_only_in_private_diagnostic=True,
                limitations="Three fixed action seeds on ten reused development starts per seed; "
                "not independent match populations or an SSL benchmark. Contact/projection/followup "
                "counts are not scoring, possession duration or intended tactics. This tests whether "
                "sampling rescues this650 model in these cases, not why all training progress differs.")


if __name__ == "__main__":
    result = build()
    output = OUT / "report.json"
    if output.exists():
        assert json.loads(output.read_text()) == result, "Never overwrite different evidence"
    else:
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
