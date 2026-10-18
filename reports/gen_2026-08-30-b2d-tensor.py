"""Build the tables for reports/2026-08-30-b2d-tensor-ladder.md by running the
branch generator (runs/b2dtensor/gen_tables.py on exp/2026-08-29-b2d-tensor) and
keeping the sections listed below.  Never hand-type numbers; rerun and paste.

    /home/tahmid/Dev/.venv/bin/python reports/gen_2026-08-30-b2d-tensor.py
"""
import os, subprocess, sys
WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b2d-tensor/experiments/separable-decoder"
RUNS = os.path.join(WT, "runs", "b2dtensor")
KEEP = ["T-1 ", "T-2 ", "T-3 ", "T-4 ", "T-6 ", "T-7 ", "T-10 ", "T-11 ", "T-12 "]
out = subprocess.run([sys.executable, os.path.join(RUNS, "gen_tables.py")], cwd=WT,
                     env={**os.environ, "TABLES_DIR": RUNS}, capture_output=True, text=True, check=True).stdout
sections, cur = [], None
for line in out.splitlines():
    if line.startswith("### "):
        cur = [line] if any(line.startswith("### " + k) for k in KEEP) else None
        if cur: sections.append(cur)
    elif cur is not None:
        cur.append(line)
print("\n".join("\n".join(s) for s in sections))
