"""Bind installed controller code to public source and compare native basis code.

Read-only: bytecode is inspected and only the pure rotation function is called.
Recorded poses are reused; no game connection, model update or new simulation.
"""
from __future__ import annotations
import base64
import json
import marshal
import subprocess
import types

import numpy as np
import torch
import audit_native_nexto_integration as audit

PUBLIC_COMMIT = "0bdb6b49072f6f3829319e68bd6210a0ca4b24a2"


def run():
    target = audit.OUT/"source_binding.json"
    assert not target.exists()
    remote_path = "repos/VirxEC/NectoFamily/contents/nexto/bot.py?ref="+PUBLIC_COMMIT
    response = json.loads(subprocess.check_output(["gh","api",remote_path]))
    public_bytes = base64.b64decode(response["content"])
    public_code = compile(public_bytes,"public_nexto.py","exec",dont_inherit=True)
    archive = audit.CArchiveReader(str(audit.EXE))
    native_code = marshal.loads(archive.extract("bot"))
    comparisons = {}
    for name in ("get_output","maybe_do_kickoff","update_controls"):
        native = audit.find_code(native_code,name)
        public = audit.find_code(public_code,name)
        checks = {key: getattr(native,key)==getattr(public,key) for key in
                  ("co_code","co_names","co_consts","co_varnames","co_freevars","co_cellvars",
                   "co_argcount","co_kwonlyargcount","co_posonlyargcount","co_flags","co_exceptiontable")}
        assert all(checks.values()),(name,checks)
        comparisons[name] = dict(checks=checks,bytecode_sha256=audit.sha(native.co_code))
    # Code is never exec'd as a module. This isolated function only reads arrays.
    pyz = archive.open_embedded_archive("PYZ.pyz")
    observation_code = pyz.extract("nexto_obs")
    rotation_code = audit.find_code(observation_code,"_quats_to_rot_mtx")
    native_rotation = types.FunctionType(rotation_code,{"np":np})
    path = audit.ROOT/"results/rival2/entity_native_gap_v1/native_physical.npz"
    data = np.load(path)
    q = data["car_quat"].reshape(-1,4).astype(np.float64)
    # Same physical quaternion; native helper uses wxyz, simulator uses xyzw.
    native = native_rotation(q[:,[3,0,1,2]])
    forward,up = audit.method_from_source(audit.ROOT/"third_party/nexto/adapter.py","_basis")(torch.from_numpy(q))
    forward_error = np.abs(forward.numpy()-native[:,:,0])
    up_error = np.abs(up.numpy()-native[:,:,2])
    result = dict(public_repository="https://github.com/VirxEC/NectoFamily",public_commit=PUBLIC_COMMIT,
        public_source_url=f"https://github.com/VirxEC/NectoFamily/blob/{PUBLIC_COMMIT}/nexto/bot.py",
        public_blob_sha1=response["sha"],public_source_sha256=audit.sha(public_bytes),
        installed_executable_sha256=audit.sha(audit.EXE.read_bytes()),
        controller_method_comparison=comparisons,
        basis=dict(recorded_quaternions=len(q),source_path=path.relative_to(audit.ROOT).as_posix(),
                   source_sha256=audit.sha(path.read_bytes()),
                   native_function_bytecode_sha256=audit.sha(rotation_code.co_code),
                   forward_max_abs_per_axis=forward_error.max(axis=0).tolist(),
                   up_max_abs_per_axis=up_error.max(axis=0).tolist(),
                   up_mean_abs_per_axis=up_error.mean(axis=0).tolist(),
                   rows_up_error_above_1e5=int(np.count_nonzero(up_error.max(axis=1)>1e-5)),
                   qualification="Component comparison only. The historical vendored upstream helper has a different up-vector sign, but the actual installed v5 helper agrees with the simulator on these recorded physical quaternions to float rounding. Do not report a current up-vector mismatch from historical source alone."),
        sources={str(p.relative_to(audit.ROOT)):audit.sha(p.read_bytes()) for p in
                 (audit.ROOT/"benchmarks/audit_native_nexto_source_binding.py",audit.ROOT/"third_party/nexto/adapter.py")},
        optimizer_steps=0,production_modified=False,game_connection=False,
        interpretation="Installed controller methods match the pinned public port. Model identity, kickoff table and scheduling have separate evidence. No claim of complete observation parity or gameplay causality.")
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps(dict(public_controller_methods_match=True,basis=result["basis"])))


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()
