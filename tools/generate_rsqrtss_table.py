"""Generate the compact pinned-host RSQRTSS lookup used by CUDA source ports."""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate_rsqrtss_table.py PROBE_EXE OUTPUT_PY")
    probe = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    line = subprocess.check_output([str(probe)], text=True).strip()
    prefix = "AMD_RSQRTSS_MANTISSA_HEX="
    if not line.startswith(prefix):
        raise RuntimeError("RSQRTSS probe did not emit the expected table")
    encoded_hex = line[len(prefix) :]
    if len(encoded_hex) != 8192 * 3:
        raise RuntimeError("RSQRTSS probe emitted an invalid table length")
    raw = b"".join(
        int(encoded_hex[index : index + 3], 16).to_bytes(2, "big")
        for index in range(0, len(encoded_hex), 3)
    )
    digest = hashlib.sha256(raw).hexdigest().upper()
    compressed = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    chunks = textwrap.wrap(compressed, 96)
    literal = "\n".join(f'    b"{chunk}"' for chunk in chunks)
    source = f'''"""Generated AMD RSQRTSS estimate table for the pinned v0.3 authority host."""

from __future__ import annotations

import base64
import hashlib
import zlib
from functools import lru_cache

import numpy as np

AMD_RSQRTSS_TABLE_SHA256 = "{digest}"
_AMD_RSQRTSS_TABLE_B85 = (
{literal}
)


@lru_cache(maxsize=1)
def amd_rsqrtss_table() -> np.ndarray:
    raw = zlib.decompress(base64.b85decode(_AMD_RSQRTSS_TABLE_B85))
    if len(raw) != 16384:
        raise RuntimeError("invalid AMD RSQRTSS table length")
    if hashlib.sha256(raw).hexdigest().upper() != AMD_RSQRTSS_TABLE_SHA256:
        raise RuntimeError("AMD RSQRTSS table hash mismatch")
    return np.frombuffer(raw, dtype=">u2").astype(np.uint16)


def amd_rsqrtss_initializer() -> str:
    values = amd_rsqrtss_table()
    return ",".join(f"0x{{value:03X}}u" for value in values)
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8", newline="\n")
    print(f"generated {output} sha256={digest} entries=8192")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
