"""One owned pause inside update50; never starts/resumes a trainer."""

import argparse
import json
import subprocess
import time

import psutil
from request_rival2_long_trace_eval20 import RUN, STOP, in_update, records

OWNER = "long-trace-execution-optimization-u50-20260904"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--py-spy", required=True)
    args = parser.parse_args()
    worker = psutil.Process(args.worker_pid)
    assert "run_rival2_ssl_foundation_v5_long_trace_v1.py" in " ".join(worker.cmdline())
    deadline = json.loads((RUN / "campaign_state.json").read_text())["deadline_unix"]
    while worker.is_running() and time.time() < deadline:
        if STOP.exists():
            raise RuntimeError("Existing stop must not be overwritten")
        before = records()[-1]["accepted_update"]
        if before >= 50:
            raise RuntimeError("Missed update50; do not interrupt a later update")
        if before == 49:
            probe = subprocess.run(
                [args.py_spy, "dump", "--pid", str(worker.pid), "--json", "--nonblocking"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0:
                dump = json.loads(probe.stdout)
                if records()[-1]["accepted_update"] == 49 and in_update(dump):
                    with STOP.open("x") as stream:
                        json.dump({"owner": OWNER, "target_update": 50}, stream)
                    print(
                        "Requested accepted update50 stop; existing evaluation runs first",
                        flush=True,
                    )
                    return
        time.sleep(1)
    raise RuntimeError("Worker stopped or original deadline reached")


if __name__ == "__main__":
    main()
