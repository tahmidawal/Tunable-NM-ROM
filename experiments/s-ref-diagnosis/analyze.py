"""Post-hoc diagnosis of the sealed Burgers quadrature S-reference failures (no model re-runs, no decision changes).

Usage: python3 analyze.py <percase.json> <sealed-closeout results dir> <out.json>
Reproduces the frozen band verdicts from per-case scores, then decomposes each verdict into
rule term (subject minus dense anchor), margin term (5% of anchor), reference term (S vs ST anchor error)
and uncertainty term (eps_ref), with counterfactuals.
"""
import json, sys, math
from statistics import median
from pathlib import Path

PANELS = ["sealed2d256", "sealed2d512", "sealed2d1024", "sealed2d2048", "sealed2d4096",
          "sealed3d65", "sealed3d129", "sealed3d257"]
RULES = {2: ("fib987", "gauss45"), 3: ("lat4096", "gl20")}


def band(lo, hi, rlo, rhi):  # identical logic to frozen_decisions.band_interval
    if not all(math.isfinite(x) for x in (lo, hi, rlo, rhi)) or rlo <= 0 or lo > hi or rlo > rhi:
        return "UNRESOLVED"
    if lo >= 0.95 * rhi and hi <= 1.05 * rlo:
        return "PASS"
    if hi < 0.95 * rlo or lo > 1.05 * rhi:
        return "FAIL"
    return "UNRESOLVED"


def stat(vals, eps, red):
    return red(vals), red([v - e for v, e in zip(vals, eps)]), red([v + e for v, e in zip(vals, eps)])


def combine(xs):
    return "FAIL" if "FAIL" in xs else ("PASS" if all(x == "PASS" for x in xs) else "UNRESOLVED")


def verdict(arm, anc, cases, kind, refs, eps_scale=1.0):
    eps = [eps_scale * refs[f"{c}|{kind}"]["eps_ref"] for c in cases]
    out = {}
    for key, red in (("T", median), ("W", max)):
        x = stat([arm[str(c)][kind] for c in cases], eps, red)
        a = stat([anc[str(c)][kind] for c in cases], eps, red)
        out[key] = dict(X=x[0], A=a[0], delta=x[0] - a[0], rel=x[0] / a[0] - 1, margin_abs=0.05 * a[0],
                        eps_halfwidth_X=(x[2] - x[1]) / 2, eps_halfwidth_A=(a[2] - a[1]) / 2,
                        verdict=band(x[1], x[2], a[1], a[2]))
    out["decision"] = combine([out["T"]["verdict"], out["W"]["verdict"]])
    return out


def main():
    pc = json.loads(Path(sys.argv[1]).read_text())
    resdir = Path(sys.argv[2])
    res = {}
    for p in PANELS:
        P = pc[p]
        dec = json.loads((resdir / f"{p}-decision.json").read_text())
        dim, refs, anc_name = P["dim"], P["references"], P["conv_arm"]
        anc = P["arms"][anc_name]["cases"]
        allc = P["cases"]
        elig = {k: [c for c in allc if refs[f"{c}|{k}"]["eligible"]] for k in ("S", "ST")}
        common = [c for c in allc if c in elig["S"] and c in elig["ST"]]
        panel = dict(dim=dim, mesh=P["mesh"], dt_rom=P["dt"], anchor=anc_name, job_id=P["job_id"],
                     n_eligible={k: len(v) for k, v in elig.items()}, n_common=len(common),
                     status={k: {} for k in ("S", "ST")}, rules={})
        for c in allc:
            for k in ("S", "ST"):
                o = refs[f"{c}|{k}"]["outcome"]
                panel["status"][k][o] = panel["status"][k].get(o, 0) + 1
        # reference term on the anchor (per case, over cases eligible for both kinds)
        dref = [anc[str(c)]["ST"] - anc[str(c)]["S"] for c in common]
        panel["anchor_ST_minus_S"] = dict(median=median(dref), min=min(dref), max=max(dref))
        panel["anchor_ratio_S_over_ST_median"] = median([anc[str(c)]["S"] / anc[str(c)]["ST"] for c in common])
        panel["eps"] = {k: dict(median=median([refs[f"{c}|{k}"]["eps_ref"] for c in elig[k]]),
                                max=max([refs[f"{c}|{k}"]["eps_ref"] for c in elig[k]])) for k in ("S", "ST")}
        # argmax output time of the anchor error trajectory
        panel["anchor_argmax_time_index"] = {
            k: {str(i): sum(1 for c in elig[k] if anc[str(c)][k + "_t"].index(max(anc[str(c)][k + "_t"])) == i)
                for i in range(5)} for k in ("S", "ST")}
        for rule in RULES[dim]:
            suffix = list(dec["kinds"]["ST"]["rules"][rule]["suffix"].keys())
            rr = {}
            for n in suffix:
                arm = P["arms"][n]["cases"]
                ent = {}
                for k in ("S", "ST"):
                    v = verdict(arm, anc, elig[k], k, refs)
                    frozen = dec["kinds"][k]["rules"][rule]["suffix"][n]
                    assert v["T"]["verdict"] == frozen["T"] and v["W"]["verdict"] == frozen["W"], (p, rule, n, k)
                    # per-case absolute rule-vs-anchor difference
                    pd = [abs(arm[str(c)][k] - anc[str(c)][k]) for c in elig[k]]
                    v["percase_abs_diff_median"] = median(pd)
                    v["percase_abs_diff_max"] = max(pd)
                    # counterfactual (b): eps -> 0
                    v["cf_eps0"] = verdict(arm, anc, elig[k], k, refs, 0.0)["decision"]
                    # counterfactual (d): common eligible set
                    v["cf_common_set"] = verdict(arm, anc, common, k, refs)["decision"]
                    # counterfactual (d'): all 128 cases, eps 0 (ineligible cases have no accepted eps)
                    v["cf_all_cases_eps0"] = verdict(arm, anc, allc, k, refs, 0.0)["decision"]
                    ent[k] = v
                # counterfactual (a): S rule difference judged against the ST absolute margin
                for key in ("T", "W"):
                    ent["S"][key]["cf_abs_delta_within_ST_margin"] = abs(ent["S"][key]["delta"]) <= ent["ST"][key]["margin_abs"]
                    ent["ST"][key]["cf_abs_delta_within_S_margin"] = abs(ent["ST"][key]["delta"]) <= ent["S"][key]["margin_abs"]
                rr[n] = ent
            panel["rules"][rule] = dict(suffix=rr,
                                        frozen={k: dec["kinds"][k]["rules"][rule]["decision"] for k in ("S", "ST")})
        # FOM proxies for the S-vs-ST reference distance (same-mesh BE/CN at dt 0.005 or 0.01)
        fp = {}
        for f, rows in P["foms"].items():
            if f.startswith(("CC_L", "UB_L", "hist_")):
                fp[f] = {k: median([rows[str(c)][k] for c in common if rows[str(c)][k] is not None]) for k in ("S", "ST")}
        panel["fom_medians_common"] = fp
        # eligibility descriptives (anchor errors of S-eligible vs S-ineligible vs ST-ineligible cases)
        groups = {"S_eligible": elig["S"], "S_ineligible": [c for c in allc if c not in elig["S"]],
                  "ST_ineligible": [c for c in allc if c not in elig["ST"]]}
        panel["eligibility_groups"] = {
            g: dict(n=len(cs), **({} if not cs else dict(
                anchor_eS_median=median([anc[str(c)]["S"] for c in cs]),
                anchor_eST_median=median([anc[str(c)]["ST"] for c in cs]),
                eps_S_median=median([refs[f"{c}|S"]["eps_ref"] for c in cs]))))
            for g, cs in groups.items()}
        panel["both_ineligible"] = len([c for c in allc if c not in elig["S"] and c not in elig["ST"]])
        # smallest symmetric relative margin at which each target rule's whole suffix would pass (eps as frozen)
        for rule in RULES[dim]:
            for k in ("S", "ST"):
                need = 0.0
                for n, ent in panel["rules"][rule]["suffix"].items():
                    for key in ("T", "W"):
                        q = ent[k][key]
                        need = max(need, (abs(q["delta"]) + q["eps_halfwidth_X"] + q["eps_halfwidth_A"]) / q["A"])
                panel["rules"][rule].setdefault("margin_needed", {})[k] = need
        res[p] = panel
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
    print("verdicts reproduced for all panels; wrote", sys.argv[3])


if __name__ == "__main__":
    main()
