"""Read-only installed Nexto identity and controller-scheduling audit.

No game connection, optimizer, simulator physics or production-source writes.
The executable archive is read as a file; process memory is never inspected.
Exact extracted controller bytecode is exercised with symbolic neural outputs
only to test scheduling, not to claim gameplay or native physics equivalence.
"""
from __future__ import annotations

import __future__
import ast
import hashlib
import json
import marshal
from pathlib import Path
import sys
import types

import numpy as np
import torch
from PyInstaller.archive.readers import CArchiveReader
import rlbot.flat as flat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/native_nexto_integration_audit_v1"
EXE = Path("C:/Users/patri/AppData/Local/RLBot5/bots/bob_build_x86_64-windows/Nexto/x86_64-pc-windows-msvc/nexto.exe")
CHANNELS = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


def sha(data):
    return hashlib.sha256(data).hexdigest().upper()


def find_code(code, name):
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            if value.co_name == name:
                return value
            found = find_code(value, name)
            if found is not None:
                return found
    return None


def method_from_source(path, name):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    node = nodes[0]
    assert not node.decorator_list
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "KICKOFF_LENGTH": 168}
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def source_kickoff(path):
    """Interpret only the actual checked-in upstream controller literal.

    Unlike the old fidelity fixture this never manually duplicates its rows.
    Unspecified SimpleControllerState fields have their standard neutral value.
    """
    tree = ast.parse(path.read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "KICKOFF_CONTROLS" for t in n.targets))

    def value(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -value(n.operand)
        if isinstance(n, ast.List):
            return [value(x) for x in n.elts]
        if isinstance(n, ast.BinOp):
            left, right = value(n.left), value(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Mult):
                return left * right
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"SimpleControllerState", "ControllerState"}:
            assert not n.args
            fields = {k.arg: value(k.value) for k in n.keywords}
            assert set(fields) <= set(CHANNELS)
            return [float(fields.get(c, 0)) for c in CHANNELS]
        raise ValueError("Unexpected kickoff literal AST: " + ast.dump(n))

    result = np.asarray(value(node), np.float32)
    assert result.shape == (168, 8)
    return result


def neural_schedule_probe(native_method, adapter_method, frames=32):
    """Continuous active, no-reset, no-kickoff scheduling with numbered actions."""
    ns = types.SimpleNamespace
    own = ns(team_num=0)
    opponent = ns(team_num=1)
    own.team_num, opponent.team_num = 0, 1
    packet = ns(match_info=ns(frame_num=0, match_phase=flat.MatchPhase.Active), balls=[object()])
    counter = [0]
    records = []
    current_frame = [0]
    control = np.zeros(8, np.float32)
    computes = []

    def act(_obs, beta):
        assert beta == 1
        counter[0] += 1
        computes.append(current_frame[0])
        return np.full(8, counter[0], np.float32), None

    def update_controls(action):
        np.copyto(control, action)

    native = ns(is_toxic=False, prev_tick=0, ticks=8, tick_skip=8, update_action=True,
                index=0, team=0, gamemode=flat.GameMode.Soccar, beta=1,
                stochastic_kickoffs=True, render=False, hardcoded_kickoffs=False,
                action=np.zeros(8), controls=control, update_controls=update_controls,
                game_state=ns(players=[own, opponent], update=lambda p: None),
                obs_builder=ns(build_obs=lambda *args: None), agent=ns(act=act))
    sim_counter = [0]
    sim_computes = []
    sim = ns(num_worlds=1, device=torch.device("cpu"), _cadence_tick=0,
             previous_action=torch.zeros(1, 8), neural_counter=torch.zeros(1, dtype=torch.int64),
             kickoff_index=torch.full((1,), -1, dtype=torch.int64), kickoff_sequence=torch.zeros(168, 8))

    def sim_neural(_state):
        sim_counter[0] += 1
        sim_computes.append(current_frame[0])
        sim.previous_action.fill_(sim_counter[0])
        return sim.previous_action, None

    sim.neural_action = sim_neural
    for frame in range(1, frames + 1):
        packet.match_info.frame_num = frame
        current_frame[0] = frame
        native_action = np.asarray(native_method(native, packet)).copy()
        simulated_action, _ = adapter_method(sim, ns(ball_pos=torch.zeros(1, 3)), torch.zeros(1, dtype=torch.bool))
        records.append([frame, float(native_action[0]), float(simulated_action[0, 0])])
    rows = np.asarray(records)
    return dict(frames=frames, native_compute_frames=computes, simulator_compute_frames=sim_computes,
                differing_emitted_frames=int(np.count_nonzero(rows[:, 1] != rows[:, 2])),
                frame_native_action_marker_simulator_action_marker=records,
                qualification="Numbered synthetic neural outputs isolate the exact controller scheduler. No trained outputs, live controls, gameplay or physics are simulated. Kickoff override disabled in both scheduling arms.")


def run():
    assert sys.version_info[:2] == (3, 12), "Match installed archive Python bytecode"
    assert not OUT.exists(), "Preserve prior audit"
    source_hashes = {str(p): sha(p.read_bytes()) for p in (
        EXE, ROOT/"third_party/nexto/nexto-model.pt", ROOT/"third_party/nexto/adapter.py",
        ROOT/"third_party/nexto/upstream/bot.py", ROOT/"benchmarks/run_nexto_fidelity.py",
        ROOT/"tests/test_nexto_adapter.py", Path(__file__))}
    archive = CArchiveReader(str(EXE))
    model = archive.extract("nexto-model.pt")
    native_code = marshal.loads(archive.extract("bot"))
    method = find_code(native_code, "get_output")
    assert method is not None
    native_method = types.FunctionType(method, {"MatchPhase": flat.MatchPhase, "GameMode": flat.GameMode, "np": np})
    adapter = ROOT/"third_party/nexto/adapter.py"
    source = source_kickoff(ROOT/"third_party/nexto/upstream/bot.py")
    table = method_from_source(adapter, "build_kickoff_sequence")("cpu").numpy()
    delta = table - source
    expected_model = (ROOT/"third_party/nexto/nexto-model.pt").read_bytes()
    assert model == expected_model, "Installed native model differs"
    result = dict(version="RIVAL2_NATIVE_NEXTO_INTEGRATION_AUDIT_V1",
        model=dict(installed_bytes=len(model), installed_sha256=sha(model),
                   pinned_sha256=sha(expected_model), exact_file_match=True),
        kickoff_table=dict(rows=168, mismatched_rows=np.flatnonzero(np.any(delta != 0, axis=1)).tolist(),
            mismatches=[dict(tick=int(i), channel=CHANNELS[j], upstream=float(source[i,j]), simulator=float(table[i,j]))
                        for i,j in np.argwhere(delta != 0)],
            interpretation="Actual vendored upstream literal versus current simulator table. This is not a manual copy used as its own reference. Installed controller bytecode is retained separately, not assumed identical to current upstream source."),
        scheduler=neural_schedule_probe(native_method, method_from_source(adapter, "tick_action")),
        source_hashes=source_hashes,
        installed_bot_bytecode_sha256=sha(archive.extract("bot")),
        native_script_function_names=[c.co_name for c in find_code(native_code,"Nexto").co_consts if isinstance(c,types.CodeType)],
        production_modified=False, optimizer_steps=0, game_connection=False, process_memory_read=False,
        limits="Model identity and specific integration differences only. Not an all-observation equivalence or causal proof of Rival native loss. Neither running native test nor production simulator adapter changed.")
    for p,h in source_hashes.items():
        assert sha(Path(p).read_bytes()) == h
    OUT.mkdir(parents=True)
    (OUT/"audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",newline="\n")
    np.savez_compressed(OUT/"kickoff_tables.npz",actual_vendored_upstream=source,current_simulator=table)
    import dis, io
    stream=io.StringIO()
    dis.dis(native_code,file=stream)
    (OUT/"installed_bot_disassembly.txt").write_text(stream.getvalue(),newline="\n")
    print(json.dumps({"model_exact":True,"kickoff_mismatched_rows":result["kickoff_table"]["mismatched_rows"],
                      "scheduler":result["scheduler"]}))


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()
