"""Apply the pre-registered stage-1 rules (DESIGN §5, §6, A2, A3) to the pulled bank jobs.

Bar (a): the R=64 bank's development floor vs the parent POD-64 floor (same job, r64),
at every mesh; PASS iff ratio <= 1.5 everywhere. "R needed to match": smallest dedicated
R whose floor <= POD-64 at every mesh. Fixed-frame control: floors > 5 % at every mesh.
Selection: candidate (dedicated bank or ordered prefix) with the fewest columns whose
floor is <= 0.20 % at all meshes, else <= 0.25 %; ties -> dedicated, then lower 96^3
floor. Stop iff no candidate <= 0.25 % at all meshes.

Writes results/bank_selection.json; with --stage copies the selected bank file to
banks/<job>/bank_selected.npz (repo path used by the mesh configs).
Usage: select_bank.py --r64 <run> --r128 <run> --r256 <run> --ff <run> [--stage]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESHES = ("32", "64", "96")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(run):
    return json.loads((HERE / "runs" / run / "output" / "summary.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r64", required=True)
    ap.add_argument("--r128", required=True)
    ap.add_argument("--r256", required=True)
    ap.add_argument("--ff", required=True)
    ap.add_argument("--stage", action="store_true")
    a = ap.parse_args()
    runs = {64: a.r64, 128: a.r128, 256: a.r256}
    S = {R: load(r) for R, r in runs.items()}
    for R, s in S.items():
        if s.get("status") not in ("final", "flagged"):
            raise SystemExit(f"bank R={R} status {s.get('status')}")
    pod = {m: S[64]["floors"][m]["parent_pod_R64"]["evolved_worst"] for m in MESHES}
    repro = {m: S[64]["floors"][m]["parent_pod_R64_reproduction"]["passed"] for m in MESHES}
    floor = {R: {m: S[R]["floors"][m][f"coordnet_R{R}"]["evolved_worst"] for m in MESHES}
             for R in S}
    ratio64 = {m: floor[64][m] / pod[m] for m in MESHES}
    bar_a = dict(r64_floor=floor[64], pod64_floor=pod, ratio=ratio64,
                 passed=bool(all(r <= 1.5 for r in ratio64.values())), pod_reproduced=repro)
    match = [R for R in sorted(S) if all(floor[R][m] <= pod[m] for m in MESHES)]
    bar_a["R_needed_to_match_pod64"] = match[0] if match else "> 256"
    ffs = load(a.ff)
    ff = {m: ffs["floors"][m]["coordnet_fixed_frame_R64"]["evolved_worst"] for m in MESHES}
    nocentre = {R: {m: S[R]["floors"][m]["coordnet_no_centring"]["evolved_worst"] for m in MESHES}
                for R in S}
    controls = dict(fixed_frame_floor=ff, fixed_frame_fails=bool(all(v > 0.05 for v in ff.values())),
                    no_centring_floor=nocentre,
                    no_centring_fails=bool(all(v > 0.05 for d in nocentre.values() for v in d.values())))
    cands = []
    for R, s in S.items():
        for key in s["floors"]["32"]:
            if not key.startswith("coordnet_R"):
                continue
            Rp = int(key[len("coordnet_R"):])
            f = {m: s["floors"][m][key]["evolved_worst"] for m in MESHES}
            cands.append(dict(parent_rank=R, columns=Rp, dedicated=(Rp == R), floor=f,
                              worst=max(f.values()), run=runs[R]))
    def pick(th):
        ok = [c for c in cands if c["worst"] <= th]
        ok.sort(key=lambda c: (c["columns"], not c["dedicated"], c["floor"]["96"]))
        return ok[0] if ok else None
    sel = pick(0.0020)
    tier = "<= 0.20 %"
    if sel is None:
        sel, tier = pick(0.0025), "<= 0.25 % (flagged: unlikely to meet bar b)"
    out = dict(bar_a=bar_a, controls=controls, candidates=cands,
               selected=sel, selection_tier=tier if sel else None,
               stop=sel is None, runs=runs, fixed_frame_run=a.ff)
    if sel and a.stage:
        src = HERE / "runs" / sel["run"] / "output" / "bank_selected.npz"
        dst = HERE / "banks" / sel["run"] / "bank_selected.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        out["staged"] = dict(path=str(dst.relative_to(HERE.parents[1])), sha256=sha(dst),
                             prefix=None if sel["dedicated"] else sel["columns"])
    (HERE / "results" / "bank_selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=1))


if __name__ == "__main__":
    main()
