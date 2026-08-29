"""Build the tables for reports/2026-08-29-b1d-tensor-sample-free-burgers.md by
running the branch's generators (runs/b1dtensor/gen_tables.py and gen_ladder.py on
exp/2026-08-29-b1d-tensor) and keeping the sections named below.  Never
hand-type numbers into the report; rerun this and paste.

    /home/tahmid/Dev/.venv/bin/python reports/gen_2026-08-29-b1d-tensor.py
"""
import os, subprocess, sys
WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder"
RUNS = os.path.join(WT, "runs", "b1dtensor")
KEEP = {
    "gen_tables.py": ["Provenance", "Key table", "Pass criteria", "Stop-reason", "E1 sign audit"],
    # cross-job slope sections deliberately excluded (review R5: two-point cross-job ratios are not exponents)
    "gen_ladder.py": ["JOB A — ladder", "Tensor vs oracle inside JOB A", "Slopes from JOB A alone",
                      "JOBs B/C", "Large-N tensor vs oracle", "Large-N gates"],
}
def run(gen):
    out = subprocess.run([sys.executable, os.path.join(RUNS, gen)], cwd=WT,
                         env={**os.environ, "TABLES_DIR": RUNS}, capture_output=True, text=True, check=True).stdout
    sections, cur = [], None
    for line in out.splitlines():
        if line.startswith("###"):
            cur = [line] if any(k in line for k in KEEP[gen]) else None
            if cur: sections.append(cur)
        elif cur is not None:
            cur.append(line)
    return "\n".join("\n".join(s) for s in sections)
print("<!-- four-job result (N=128..1024) -->\n" + run("gen_tables.py"))
print("\n<!-- constant-time verification -->\n" + run("gen_ladder.py"))
