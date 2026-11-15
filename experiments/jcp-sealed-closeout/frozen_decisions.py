"""Prospective Burgers four-rule sufficiency classifier (ST primary, S secondary)."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


POPULATION = {2: ("sealed128", 128), 3: ("sealed64", 64)}
MESHES = {2: (256, 512, 1024, 2048, 4096), 3: (65, 129, 257)}
RULES = {2: ("fib987", "gauss45"), 3: ("lat4096", "gl20")}
KINDS = ("ST", "S")
STORE = "3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0"
OUTCOMES = {"accepted", "ic-mismatch", "solver-fail", "unresolved-floor", "peclet-fail",
            "order-fail", "direction-inconsistent"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def combine(states):
    if "FAIL" in states:
        return "FAIL"
    return "PASS" if states and all(x == "PASS" for x in states) else "UNRESOLVED"


def band_interval(lo, hi, ref_lo, ref_hi):
    """Independent adverse endpoints, including the lower 5% edge, A2.3."""
    if not all(math.isfinite(x) for x in (lo, hi, ref_lo, ref_hi)) or ref_lo <= 0:
        return "UNRESOLVED"
    if lo > hi or ref_lo > ref_hi:
        return "UNRESOLVED"
    # Compare the outer and inner bounds without floating-point division near 1.
    if lo >= 0.95 * ref_hi and hi <= 1.05 * ref_lo:
        return "PASS"
    if hi < 0.95 * ref_lo or lo > 1.05 * ref_hi:
        return "FAIL"
    return "UNRESOLVED"


def indexed(rows, key, names, cases):
    indexed_rows = {}
    for row in rows:
        index = (row[key], row["case"])
        require(type(row["case"]) is int and index not in indexed_rows, f"duplicate or invalid {key}/case")
        indexed_rows[index] = row
    expected = {(name, c) for name in names for c in cases}
    require(set(indexed_rows) == expected, f"incomplete or unexpected {key}/case identities")
    return {name: [indexed_rows[name, c] for c in cases] for name in names}


def sizes_and_families(cfg):
    spec = {a["name"]: a for a in cfg["arms"]}
    require(len(spec) == len(cfg["arms"]), "duplicate config arms")
    if cfg["dim"] == 3:
        spec["tensor"] = {"name": "tensor", "rule": "tensor", "family": "tensor"}
    families = defaultdict(list)
    sizes = {}
    for name, setting in spec.items():
        if setting["family"] in ("control", "tensor") or name in cfg["controls"]:
            continue
        match = re.fullmatch(r"(fib|gauss|sobol|lat|gl|sob|kor)([0-9]+)", setting["rule"])
        require(match is not None, f"unknown quadrature rule {name}")
        prefix, size = match.groups()
        sizes[name] = int(size) ** {"gauss": 2, "gl": 3}.get(prefix, 1)
        families[setting["family"]].append(name)
    for ladder in families.values():
        ladder.sort(key=sizes.__getitem__)
        require(len({sizes[x] for x in ladder}) == len(ladder), "ambiguous rung order")
    return spec, families, sizes


def exit_status(row, dim):
    counts = row["exits"] if dim == 2 else row["reasons"]
    keys = ("budget", "tol", "tiny_step", "damping_limit", "stationary") if dim == 2 else ("0", "1", "2", "3", "4")
    require(set(counts) == set(keys) and all(type(x) is int and x >= 0 for x in counts.values()), "invalid exits")
    steps = sum(counts.values())
    bad = counts[keys[0]] + counts[keys[2]] + counts[keys[3]]
    require(steps > 0 and steps == row["steps"] and bad == row["nonconv_steps"], "exit totals disagree")
    require((counts[keys[3]] > 0) is row["hard_fail"] and type(row["finite"]) is bool, "invalid hard or finite flag")
    return bad, steps, counts[keys[3]]


def validate_mesh(rep, cfg, audit, result_sha):
    d = cfg["dim"]
    mesh = cfg["mesh"]
    cohort, n = POPULATION[d]
    cases = list(range(n))
    require(mesh in MESHES[d] and cfg["cohort"] == cohort and cfg["cases"] == cases, "wrong registered population")
    require(rep["config"] == cfg and rep["cases"] == cases and rep["mesh"] == mesh, "frozen config or cases mismatch")
    require(rep["complete"] is True and rep["backend"] == "gpu" and rep["x64"] is True
            and rep["precision"] == "highest", "incomplete panel or precision/backend")
    require(rep["store_version"] == cfg["expected_store_version"] == STORE, "reference version mismatch")
    require(bool(rep["job_id"]) and bool(rep["gpu"]), "missing allocation identity")
    require(audit["passed"] is True and audit["checks"]["G_audit"]["passed"] is True
            and audit["checks"]["fault_injections"]["passed"] is True, "independent audit unavailable or failed")
    require(audit["mesh"] == mesh and audit["dim"] == d and str(audit["job_id"]) == str(rep["job_id"])
            and audit["result_sha256"] == result_sha, "audit not bound to this result")
    gates = rep["gates"]
    for key in ("G_R", "G_coh", "G_ref", "G_lat", "G_score_vs_refstore"):
        require(gates[key]["passed"] is True, f"{key} failed")
    require(gates["G_ref"]["loaded"] == 2*n and gates["G_ref"]["store_version"] == rep["store_version"], "reference coverage")
    if d == 3:
        for key, tol in (("G1_derivative_vs_fd", 1e-6), ("G2_meshnodes_offmesh_vs_tensor", 1e-12),
                         ("G3_jacobian_vs_jacfwd", 1e-12), ("G3_adv_half_Jc", 1e-12)):
            require(number(gates[key]) and gates[key] < tol, f"{key} failed")
        g4 = gates["G4_solver_vs_vendor"]
        require(g4["its_equal"] is True and all(number(g4[k]) and g4[k] <= 1e-12 for k in ("fields", "coefs")),
                "G4 failed")
    spec, families, sizes = sizes_and_families(cfg)
    arms = indexed(rep["rows"], "arm", spec, cases)
    fom_names = [f["name"] for f in cfg["foms"]]
    require(len(set(fom_names)) == len(fom_names), "duplicate config FOMs")
    foms = indexed(rep["fom_rows"], "fom", fom_names, cases)
    require(set(rep["references"]) == {f"{c}|{kind}" for c in cases for kind in KINDS}, "reference identities")
    exits = {}
    for name, rows in arms.items():
        for row in rows:
            require(row["family"] == spec[name]["family"] and row["rule"] == spec[name]["rule"]
                    and (name not in sizes or row["m"] == sizes[name]), f"unexpected arm specification for {name}")
        bad, steps, hard = map(sum, zip(*(exit_status(row, d) for row in rows)))
        nonfinite = [r["case"] for r in rows if not r["finite"]]
        exits[name] = dict(valid=not nonfinite and not hard and 100*bad <= steps, nonfinite_cases=nonfinite,
                           hard_fail_steps=hard, nonconverged_steps=bad, total_steps=steps)
    return cases, spec, families, arms, foms, exits


def stats(rows, kind, cases, refs, is_fom=False):
    values = []
    for i, row in enumerate(rows):
        score = row[kind]
        require(score["eligible"] is refs[i]["eligible"], "reference/score eligibility mismatch")
        value = score["e"]
        if not number(value):
            require(value is None and (not row["converged"] if is_fom else not row["finite"]), "invalid score")
        else:
            ts = score["e_t"]
            require(len(ts) == 5 and all(number(t) for t in ts) and max(ts) == value, "trajectory error mismatch")
        values.append(value)
    all_finite = all(x is not None for x in values)
    selected = [values[i] for i in cases]
    out = dict(n_all=len(values), n_eligible=len(selected), n_finite=sum(x is not None for x in values),
               all_W=max(values) if all_finite else None, all_T=median(values) if all_finite else None)
    if not selected or None in selected:
        return dict(out, W=None, T=None)
    eps = [refs[i]["eps_ref"] for i in cases]
    for key, reduce in (("W", max), ("T", median)):
        out[key] = reduce(selected)
        out[key+"_lo"] = reduce([x-e for x,e in zip(selected,eps)])
        out[key+"_hi"] = reduce([x+e for x,e in zip(selected,eps)])
    return out


def compare_to_anchor(rows, reference_rows, refs, kind):
    by_label = {key: [] for key in
                ("lower_error", "higher_error", "uncertainty_overlap", "missing", "ineligible")}
    for i, ref in enumerate(refs):
        a, b = rows[i][kind]["e"], reference_rows[i][kind]["e"]
        if not ref["eligible"]:
            label = "ineligible"
        elif a is None or b is None:
            label = "missing"
        elif abs(a-b) <= 2*ref["eps_ref"]:
            label = "uncertainty_overlap"
        else:
            label = "lower_error" if a < b else "higher_error"
        by_label[label].append(i)
    return dict(counts={key: len(ids) for key, ids in by_label.items()}, case_ids=by_label)

def one_kind(rep, cfg, kind, evidence):
    cases, spec, families, arms, foms, exits = evidence
    refs = [rep["references"][f"{c}|{kind}"] for c in cases]
    ids = [c for c in cases if refs[c]["eligible"] is True]
    for ref in refs:
        # In the frozen STATUS both dimensions are validated. Per-case crosschecks apply
        # only to validation cohorts, and never to sealed128 or sealed64.
        require(ref["outcome"] in OUTCOMES and ref["eligible"] is (ref["outcome"] == "accepted"),
                "frozen reference outcome/eligibility mismatch")
        if ref["eligible"]:
            require(ref["outcome"] == "accepted" and number(ref["eps_ref"]) and number(ref["n0"]) and ref["n0"] > 0,
                    "eligible reference lacks accepted outcome, finite uncertainty or positive initial norm")
    subjects = {n: stats(rows,kind,ids,refs) for n,rows in arms.items()}
    require(all(type(r["converged"]) is bool for rows in foms.values() for r in rows), "invalid FOM convergence flag")
    fom = {n: dict(stats(rows,kind,ids,refs,True), converged_cases=sum(r["converged"] is True for r in rows))
           for n,rows in foms.items()}
    targets = RULES[cfg["dim"]]
    result = dict(status_counts=dict(Counter(ref["outcome"] for ref in refs)), eligible_cases=ids, n_all=len(refs),
                  excluded_cases=[c for c in cases if not refs[c]["eligible"]], arm_stats=subjects, fom_stats=fom,
                  rules={n: {"decision":"UNRESOLVED"} for n in targets})
    if not ids:
        result["reason"] = "no eligible references"
        return result
    conv, check = cfg["conv_arm"], cfg["anchor_check_arm"]
    a,b = subjects[conv],subjects[check]
    def close_to(subject, anchor):
        return all(number(subject[k]) and number(anchor[k]) and anchor[k] > 0
                   and abs(subject[k]/anchor[k]-1) <= 0.01 for k in ("W","T"))
    distances = [r["dist_conv"] for r in arms[check]]
    anchor_pass = (exits[conv]["valid"] and exits[check]["valid"] and all(number(v) for v in distances)
                   and max(distances) <= 1e-3 and close_to(a,b))
    tails = {fam: len(ladder)>=2 and all(exits[x]["valid"] and close_to(subjects[x],a) for x in ladder[-2:])
             for fam,ladder in families.items()}
    result.update(anchor_pass=anchor_pass, tail_by_family=tails, conv_arm=conv, anchor_check_arm=check)
    for name in targets:
        fam = spec[name]["family"]
        ladder = families[fam]
        suffix = ladder[ladder.index(name):]
        tests = {}
        for n in suffix:
            s = subjects[n]
            if not exits[n]["valid"]:
                tests[n] = {"decision":"FAIL","reason":"invalid rollout"}
            elif any(a.get(k+"_lo") is None or a[k+"_lo"] <= 0 for k in ("W","T")):
                tests[n] = {"decision":"UNRESOLVED","reason":"anchor uncertainty reaches zero"}
            else:
                bands = {k:band_interval(s[k+"_lo"],s[k+"_hi"],a[k+"_lo"],a[k+"_hi"]) for k in ("W","T")}
                tests[n] = dict(bands,decision=combine(list(bands.values())))
        blockers = []
        if not anchor_pass:
            blockers.append("independent anchor failed")
        if sum(tails.values()) < 2:
            blockers.append("fewer than two demonstrated family tails")
        if not tails[fam]:
            blockers.append("selected family tail not demonstrated")
        result["rules"][name] = dict(decision="UNRESOLVED" if blockers else combine([t["decision"] for t in tests.values()]),
                                     blockers=blockers, suffix=tests, paired_vs_conv=compare_to_anchor(arms[name],arms[conv],refs,kind))
    return result


def classify(rep, cfg, audit, result_sha):
    dim, mesh = cfg["dim"], cfg["mesh"]
    base = dict(dim=dim, mesh=mesh, cohort=cfg["cohort"], result_sha256=result_sha, job_id=rep.get("job_id"), kinds={})
    def unresolved(message):
        return dict(reason=message, rules={n:{"decision":"UNRESOLVED"} for n in RULES[dim]})
    try:
        evidence = validate_mesh(rep,cfg,audit,result_sha)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        base["kinds"] = {kind:unresolved(f"incomplete or failed verification: {exc}") for kind in KINDS}
        return base
    base["exit_status"] = evidence[-1]
    for kind in KINDS:
        try:
            base["kinds"][kind] = one_kind(rep,cfg,kind,evidence)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            base["kinds"][kind] = unresolved(f"incomplete or failed {kind} evidence: {exc}")
    return base


def aggregate(panels):
    found = {}
    for panel in panels:
        key = panel["dim"],panel["mesh"]
        require(key not in found and key[0] in MESHES and key[1] in MESHES[key[0]], "unexpected or duplicate mesh")
        found[key] = panel
    out = {}
    for d, names in RULES.items():
        for name in names:
            out[name] = {}
            for kind in KINDS:
                states = {}
                for mesh in MESHES[d]:
                    panel = found.get((d,mesh))
                    states[str(mesh)] = panel["kinds"][kind]["rules"][name]["decision"] if panel else "UNRESOLVED"
                out[name][kind] = dict(by_mesh=states, decision=combine(list(states.values())))
    return out


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest="action",required=True)
    panel=commands.add_parser("panel")
    for flag in ("config","result","audit"):
        panel.add_argument("--"+flag, type=Path, required=True)
    combine_cmd=commands.add_parser("aggregate")
    combine_cmd.add_argument("panels", type=Path, nargs="*")
    args=cli.parse_args()
    if args.action == "aggregate":
        output=aggregate([json.loads(p.read_bytes()) for p in args.panels])
    else:
        raw=args.result.read_bytes()
        receipt=json.loads(args.audit.read_bytes()) if args.audit.exists() else {}
        output=classify(json.loads(raw),json.loads(args.config.read_bytes()),receipt,digest(raw))
        output["config_sha256"]=digest(args.config.read_bytes())
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
