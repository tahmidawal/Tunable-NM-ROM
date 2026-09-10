"""Checksum-collect a completed private replay; delete only after local audit."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PY = "/home/tahmid/Dev/.venv/bin/python"
BASE = "/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907"


def main():
    attempt, job = sys.argv[1:]
    assert attempt.startswith("historical_") and attempt.replace("_", "").isalnum()
    assert job.isdigit()
    remote = f"{BASE}/{attempt}"
    target = HERE / "runs" / attempt
    target.mkdir(parents=True)
    queued = subprocess.check_output(["ssh", "tufts-login", "squeue -h -u tawal01 -o %A"], text=True)
    assert job not in queued.split(), queued
    command = (f"set -e; cd {shlex.quote(remote)}; test -f out/COMPLETE; "
               "find . -type f ! -name COLLECTION.sha256 -print0 | sort -z | "
               "xargs -0 sha256sum > COLLECTION.sha256")
    subprocess.run(["ssh", "tufts-login", command], check=True)
    subprocess.run(["scp", "-rq", f"tufts-login:{remote}/.", str(target)], check=True)
    subprocess.run(["sha256sum", "-c", "COLLECTION.sha256", "--quiet"], cwd=target, check=True)
    print(f"CHECKSUM_LOCAL {target}", flush=True)
    subprocess.run([PY, str(HERE / "audit.py"), str(target)], check=True)
    subprocess.run(["ssh", "tufts-login", f"rm -rf -- {shlex.quote(remote)}; test ! -e {shlex.quote(remote)}"], check=True)
    state = json.loads((HERE / "SUBMISSION.json").read_text())
    state.update(state="complete, checksum-collected, audited", collection_pending=False,
                 remote_deleted=True, local_run=str(target.relative_to(HERE)))
    (HERE / "SUBMISSION.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"COLLECTED_AUDITED_REMOVED {remote}", flush=True)


if __name__ == "__main__":
    main()
