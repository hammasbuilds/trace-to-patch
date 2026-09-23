"""Show what trace-to-patch does, in one command, with nothing to set up.

    python demo.py

Injects bugs into a real library, runs its suite, and measures how well the
failure output alone points at the line responsible - split by whether the
failure produced a traceback or only a bare assertion. That split is the
finding: the same bug is far easier to locate when Python happens to raise.

The target is `toolz`, a real library vendored under targets/ - not a fixture
built to flatter the tool. Runs with --locate-only and --no-embeddings, so it needs no model and no
GPU. Localisation is the part that can be scored without generation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    print('trace-to-patch: how much is a traceback worth when locating a bug?', flush=True)
    print(flush=True)
    result = subprocess.run(
        [sys.executable, "-m", 'trace_to_patch.cli', *['bench', 'targets/toolz', '--locate-only', '--no-embeddings', '--limit', '5']],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    print(flush=True)
    for line in ['Run the full pipeline, including patching, with:', '    trace-to-patch fix <repo> --test <failing test>', '    trace-to-patch bench <repo>        # add --locate-only to skip the model']:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
