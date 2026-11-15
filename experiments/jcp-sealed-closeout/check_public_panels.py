"""Development validation only: compare to original decision logic on its eight PUBLIC panels.

Usage: pass the upstream quadrature-plateau source directory. Both input cohorts are
public test cases. Duplicate their rows in memory ONLY to exercise the strict 128/64
shape checks. Synthetic audit receipts below never verify real submission evidence.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import frozen_decisions as fd


def run(directory):
    module = importlib.util.spec_from_file_location("history_plateau", directory / "plateau.py")
    plateau = importlib.util.module_from_spec(module)
    module.loader.exec_module(plateau)
    count = 0
    for run in ("p2d256b","p2d512","p2d1024","p2d2048","p2d4096","p3d65b","p3d129","p3d257c"):
        original = json.loads((directory/"results"/run/"result.json").read_bytes())
        assert original["config"]["cohort"] in ("test64","test32")
        rep = copy.deepcopy(original)
        dim = rep["config"]["dim"]
        co, n = fd.POPULATION[dim]
        half = n//2
        rep["config"].update(cohort=co,cases=list(range(n)))
        rep["cases"] = list(range(n))
        rep["gates"]["G_ref"]["loaded"] = 2*n
        for key in ("rows","fom_rows"):
            rep[key] += [dict(r,case=r["case"]+half) for r in rep[key]]
        rep["references"].update({f"{c+half}|{k}":rep["references"][f"{c}|{k}"] for c in range(half) for k in fd.KINDS})
        fake = "f"*64
        audit = dict(passed=True, dim=dim, mesh=rep["mesh"],job_id=rep["job_id"],result_sha256=fake,
                     checks=dict(G_audit=dict(passed=True),fault_injections=dict(passed=True)))
        actual=fd.classify(rep,rep["config"],audit,fake)
        for kind in fd.KINDS:
            reference=plateau.plateau(original,kind)
            observed=actual["kinds"][kind]
            assert "anchor_pass" in observed, observed.get("reason")
            assert observed["anchor_pass"] == reference["anchor"]["passed"]
            for name in fd.RULES[dim]:
                fam=next(r["family"] for r in rep["config"]["arms"] if r["name"]==name)
                ladder=reference["families"][fam]
                suffix=ladder["rungs"]
                suffix=suffix[next(i for i,r in enumerate(suffix) if r["arm"]==name):]
                expected=fd.combine([r["glob"].upper() for r in suffix])
                if not (reference["demonstrated"] and ladder["tail_demonstrated"]):
                    expected="UNRESOLVED"
                rule=observed["rules"][name]
                assert rule["decision"]==expected, (run,kind,name,rule,expected)
                for item in suffix:
                    assert rule["suffix"][item["arm"]]["decision"]==item["glob"].upper()
                count+=1
    print(f"{count} public historical rule/kind/mesh decisions agreed with original anchor, tails and suffix rungs.")


if __name__=="__main__":
    run(Path(sys.argv[1]))
