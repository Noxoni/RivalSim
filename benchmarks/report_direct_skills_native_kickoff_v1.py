"""CPU-only reduction of the completed, immutable native kickoff traces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.report_direct_skills_native_nexto_v1 import sha

OUT = ROOT / "results/rival2/direct_skills_native_kickoff_v1"
FILES = {"parent650": "parent650_attempt2", "child10": "child10", "child25": "child25"}


def event_tick(values):
    indices = np.flatnonzero(values)
    return int(indices[0]) + 1 if len(indices) else None


def reduce_case(data):
    rows, race_commands = [], []
    sides = data["rival_side"]
    delta = data["post.touch_count"] - data["pre.touch_count"]
    for w, side in enumerate(sides):
        side = int(side)
        valid = data["pre.goal_count"][:, w] == 0
        first = [event_tick((delta[:, w, c] > 0) & valid) for c in (0, 1)]
        race_end = min((v for v in first if v is not None), default=721) - 1
        race = (np.arange(len(valid)) < race_end) & valid
        car_rows = []
        for c in (int(side), 1-int(side)):
            commands = data["applied_controls"][:, w, c]
            if c == side:
                race_commands.append(commands[race])
            car_rows.append(dict(car=c, is_rival=c == side,
                first_contact_tick=first[c],
                first_native_jumped_tick=event_tick((data["post.has_jumped"][:, w, c] != 0) & race),
                first_native_flipped_tick=event_tick((data["post.has_flipped"][:, w, c] != 0) & race),
                race_mean_speed=float(np.linalg.norm(data["pre.car_vel"][race, w, c], axis=-1).mean()) if race.any() else None,
                race_airborne_ticks=int((data["pre.on_ground"][race, w, c] == 0).sum()),
                race_ticks=int(race.sum())))
        rows.append(dict(world=w, side=int(side), layout=int(data["starting_layout"][w]),
            cars=car_rows, rival_first=(first[int(side)] is not None and
                (first[1-int(side)] is None or first[int(side)] < first[1-int(side)])),
            rival_contact_lag_ticks=None if None in first else first[int(side)] - first[1-int(side)]))
    commands = np.concatenate(race_commands)
    return dict(worlds=rows, aggregate=dict(
        first_contacts_won=sum(r["rival_first"] for r in rows),
        rival_pre_contact_flips=sum(r["cars"][0]["first_native_flipped_tick"] is not None for r in rows),
        nexto_pre_contact_flips=sum(r["cars"][1]["first_native_flipped_tick"] is not None for r in rows),
        rival_race_mean_speed=float(np.mean([r["cars"][0]["race_mean_speed"] for r in rows])),
        nexto_race_mean_speed=float(np.mean([r["cars"][1]["race_mean_speed"] for r in rows])),
        rival_race_full_throttle_fraction=float(np.mean(commands[:, 0] == 1)),
        rival_race_boost_command_fraction=float(np.mean(commands[:, 6] == 1)),
        rival_race_reverse_throttle_fraction=float(np.mean(commands[:, 0] < 0)),
        rival_contact_lag_min_ticks=min(r["rival_contact_lag_ticks"] for r in rows),
        rival_contact_lag_max_ticks=max(r["rival_contact_lag_ticks"] for r in rows)))


def action_difference(parent, child):
    assert np.array_equal(parent["rival_side"], child["rival_side"])
    delta = parent["post.touch_count"] - parent["pre.touch_count"]
    mask = np.zeros(parent["policy_action"].shape[:2], dtype=bool)
    for w in range(mask.shape[1]):
        first = event_tick(delta[:, w].any(-1))
        mask[:, w] = np.arange(mask.shape[0]) * 4 < (first - 1 if first is not None else len(delta))
    difference = np.any(parent["policy_action"] != child["policy_action"], axis=-1)
    return dict(pre_parent_first_contact_different_actions=int(difference[mask].sum()),
        pre_parent_first_contact_decisions=int(mask.sum()),
        full_six_second_different_actions=int(difference.sum()), total_decisions=int(difference.size),
        interpretation="Same opening starts; own closed-loop trajectories can diverge. This is control sequence similarity, not same-observation policy distribution KL.")


def main():
    data, reports, results = {}, {}, {}
    for name, file in FILES.items():
        report_path = OUT / (file + ".json")
        report = json.loads(report_path.read_text())
        assert all(report["checks"].values())
        assert report["optimizer_steps"] == 0
        archive = ROOT / report["archive"]["path"]
        assert sha(archive) == report["archive"]["sha256"]
        with np.load(archive, allow_pickle=False) as source:
            data[name] = dict(source)
        for k, array in data[name].items():
            identity = report["archive"]["arrays"][k]
            assert list(array.shape) == identity["shape"] and str(array.dtype) == identity["dtype"]
            assert hashlib.sha256(array.tobytes()).hexdigest().upper() == identity["sha256"]
            assert np.isfinite(array).all()
        reports[name] = dict(path=report_path.relative_to(ROOT).as_posix(), sha256=sha(report_path),
            archive_sha256=sha(archive), checkpoint=report["case"])
        results[name] = reduce_case(data[name])
    output = dict(version="RIVAL2_NATIVE_KICKOFF_PHYSICAL_REDUCTION_V1", reports=reports,
        results=results, action_differences={n: action_difference(data["parent650"], data[n]) for n in ("child10", "child25")},
        no_optimizer_or_policy_change=True,
        findings="All three use full throttle/boost before first contact, zero reverse throttle, and lose all ten initial first contacts. No Rival native flip occurs before first contact; every Nexto car has one. Rival is slower over the initial race. Child25 changes only12/670 pre-contact action decisions versus parent. These observations isolate an approach/acceleration deficit, not refusal to pursue the ball. They do not prove prescribing a flip is the optimal solution or prove causality for the full-match losses.",
        limitations="Ten fixed development starts only, no goals in the six-second traces so matching goal prefixes are empty. Exact full capture/action/hidden replay checks passed; no independent full-trajectory reference exists beyond the preserved parent operational retry. No named mechanic detector or reward added.")
    encoded = (json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = OUT / "analysis.json"
    if path.exists():
        assert path.read_bytes() == encoded, "Existing analysis differs; do not overwrite"
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    print(json.dumps({k: v["aggregate"] for k,v in results.items()}))
    print(json.dumps(output["action_differences"]))


if __name__ == "__main__":
    main()
