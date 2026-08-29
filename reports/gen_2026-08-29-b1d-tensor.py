"""Build the tables for reports/2026-08-29-b1d-tensor-sample-free-burgers.md by
running the branch's generator (runs/b1dtensor/gen_tables.py on
exp/2026-08-29-b1d-tensor) and keeping the sections named below.  Never
hand-type numbers into the report; rerun this and paste.

    /home/tahmid/Dev/.venv/bin/python reports/gen_2026-08-29-b1d-tensor.py
"""
import os, subprocess, sys
WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder"
RUNS = os.path.join(WT, "runs", "b1dtensor")
KEEP = ["Provenance", "Key table", "Pass criteria", "Stop-reason", "E1 sign audit"]
out = subprocess.run([sys.executable, os.path.join(RUNS, "gen_tables.py")],
                     env={**os.environ, "TABLES_DIR": RUNS}, capture_output=True, text=True, check=True).stdout
sections, cur = [], None
for line in out.splitlines():
    if line.startswith("### "):
        cur = [line] if any(k in line for k in KEEP) else None
        if cur: sections.append(cur)
    elif cur is not None:
        cur.append(line)
print("\n".join("\n".join(s) for s in sections))
