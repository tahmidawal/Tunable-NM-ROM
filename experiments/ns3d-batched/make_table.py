"""Tables from results/<run>/batched.json (no hand-typed numbers).
Usage: make_table.py results/b64/batched.json [more ...] > results/table.md"""
import json, sys


def g(c, k="median_ms"):
    return c.get(k) if c else None


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def best(cs, prefix, tag):
    cand = {k: v for k, v in cs.items() if k.split("|", 1)[0].startswith(prefix) and k.split("|", 1)[1] == tag
            and v.get("median_ms") is not None}
    if not cand:
        return None, None
    k = min(cand, key=lambda x: cand[x]["median_ms"])
    return k.split("|")[0], cand[k]


def mark(c):
    return f" (chunk {c['chunk']})" if c and c.get("chunked") else ""


def summary(paths):
    """Headline: per mesh/arm/B, best ROM e2e vs best solver solve-only, per-query ms and ratio."""
    print("| mesh | GPU | arm | ROM worst % | solver (worst %) | B | ROM ms/query | solver ms/query | ROM queries/s "
          "| solver queries/s | ROM s/1000 | solver s/1000 | ratio | ratio vs ROM solve-only |")
    print("|" + "---|" * 14)
    for p in paths:
        r = json.load(open(p)); n = r["mesh"]; gpu = r["gpu"].split(",")[0].replace("NVIDIA ", "")
        for arm, a in r["arms"].items():
            m = a["matched"]; tag = f"{m['scheme']}|{m['n']}|{m['steps']}"
            for B, b in sorted(r["batches"].items(), key=lambda x: int(x[0])):
                cs = b["callables"]; Bi = int(B)
                rk, rc = best(cs, "ROM_VMAP", arm); sk, sc = best(cs, "ROM_SHARED", arm)
                if sc and (rc is None or sc["median_ms"] < rc["median_ms"]):
                    rk, rc = sk, sc
                fk, fc = best(cs, "SOLVE_", tag); lat = cs.get(f"ROM_LAT|{arm}")
                if rc is None: continue
                print(f"| {n} | {gpu} | {arm} | {a['fresh_worst'] * 100:.2f} | {m['scheme']} {m['n']}^3 x{m['steps']} ({m['cohort_worst'] * 100:.2f}) "
                      f"| {Bi} | {rc['median_ms'] / Bi:.2f}{mark(rc)} | " + (f"{fc['median_ms'] / Bi:.1f}{mark(fc)}" if fc else "-")
                      + f" | {rc['samples_per_s']:.0f} | " + (f"{fc['samples_per_s']:.2f}" if fc else "-")
                      + f" | {rc['ms_per_1000'] / 1e3:.2f} | " + (f"{fc['ms_per_1000'] / 1e3:.0f}" if fc else "-")
                      + " | " + (f"{fc['median_ms'] / rc['median_ms']:.1f}" if fc else "-")
                      + " | " + (f"{fc['median_ms'] / lat['median_ms']:.0f}" if fc and lat and lat.get("median_ms") else "-") + " |")


def main(paths):
    for p in paths:
        r = json.load(open(p)); n = r["mesh"]
        print(f"\n## Mesh {n}^3 — job {r['job_id']}, {r['gpu']}, status {r.get('status')}\n")
        if r["failures"]:
            print("Failures: " + "; ".join(r["failures"]) + "\n")
        print("| arm | worst err (ROM / solver) | solver config | B | ROM e2e ms (variant) | ROM samples/s | ROM s per 1000 "
              "| ROM mem GB | solver solve ms (variant) | solver samples/s | solver s per 1000 | solver mem GB "
              "| ratio solve/e2e | ratio e2e/e2e | ratio solve/solve |")
        print("|" + "---|" * 15)
        for arm, a in r["arms"].items():
            m = a["matched"]; tag = f"{m['scheme']}|{m['n']}|{m['steps']}"
            for B, b in sorted(r["batches"].items(), key=lambda x: int(x[0])):
                cs = b["callables"]; Bi = int(B)
                rk, rc = best(cs, "ROM_VMAP", arm); sk, sc = best(cs, "ROM_SHARED", arm)
                if sc and (rc is None or sc["median_ms"] < rc["median_ms"]):
                    rk, rc = sk, sc
                fk, fc = best(cs, "SOLVE_", tag); ek, ec = best(cs, "E2E_", tag)
                lat = cs.get(f"ROM_LAT|{arm}")
                if rc is None:
                    continue
                rm = rc["mem"]["batch_need_gb"] if rc.get("mem") else None
                fm = fc["mem"]["total_gb"] if fc and fc.get("mem") else None
                print(f"| {arm} | {a['fresh_worst'] * 100:.2f} / {m['cohort_worst'] * 100:.2f} % | {m['scheme']} {m['n']}^3 x{m['steps']} "
                      f"| {Bi} | {fmt(rc['median_ms'])} ({rk.split('_')[1].lower()}){mark(rc)} | {fmt(rc['samples_per_s'], 1)} "
                      f"| {fmt(rc['ms_per_1000'] / 1e3)} | {fmt(rc['mem']['total_gb'] if rc.get('mem') else None, 1)} "
                      f"| {fmt(g(fc))}" + (f" ({fk.split('_')[1].lower()}){mark(fc)}" if fc else "")
                      + f" | {fmt(g(fc, 'samples_per_s'), 1)} | {fmt(g(fc, 'ms_per_1000') / 1e3 if fc else None)} | {fmt(fm, 1)} "
                      f"| {fmt(fc['median_ms'] / rc['median_ms'] if fc else None)} "
                      f"| {fmt(ec['median_ms'] / rc['median_ms'] if ec else None)} "
                      f"| {fmt(fc['median_ms'] / lat['median_ms'] if (fc and lat and lat.get('median_ms')) else None)} |")
        print("\nAll variants (median ms per batch call; chunk if chunked):\n")
        names = sorted({k for b in r["batches"].values() for k in b["callables"]})
        Bs = sorted(r["batches"], key=int)
        print("| callable | " + " | ".join(f"B={B}" for B in Bs) + " |")
        print("|---|" + "---|" * len(Bs))
        for nm in names:
            row = []
            for B in Bs:
                c = r["batches"][B]["callables"].get(nm)
                row.append("-" if not c or c.get("median_ms") is None else fmt(c["median_ms"]) + mark(c))
            print(f"| {nm} | " + " | ".join(row) + " |")
        print("\nAgreement with the unchanged batch-1 code (max |diff| / max |ref| over checked slots):\n")
        for B, chk in r["checks"].items():
            vals = [v.get("max_rel_diff") for v in chk.values() if v.get("max_rel_diff") is not None]
            errs = [k for k, v in chk.items() if "error" in v]
            print(f"- B={B}: max {max(vals):.2e} over {len(vals)} callables" + (f"; errors: {errs}" if errs else ""))
        if r["cohort"]:
            print("\nCohort agreement (all cases, B=32 callables):\n")
            for k, v in r["cohort"].items():
                print(f"- {k}: max |err diff| {v['max_abs_err_diff']:.1e}, worst {v['worst'] * 100:.3f} %, ok {v['ok']}")
        if r["rematch"]:
            print("\nRe-match at B=32 (solve-only, shortlist configurations):\n")
            for arm, a in r["arms"].items():
                tags = {f"{s[0]}|{s[1]}|{s[2]}" for s in a["shortlist"]}
                cand = {k: v for k, v in r["rematch"].items() if k.split("|", 1)[1] in tags}
                if not cand: continue
                k = min(cand, key=lambda x: cand[x]["median_ms"])
                rom = min((v["median_ms"] for kk, v in r["batches"]["32"]["callables"].items()
                           if kk in (f"ROM_VMAP|{arm}", f"ROM_SHARED|{arm}") and v.get("median_ms")), default=None)
                print(f"- {arm}: fastest shortlisted {k} {cand[k]['median_ms']:.1f} ms; ratio vs ROM e2e "
                      + (f"{cand[k]['median_ms'] / rom:.2f}" if rom else "-"))
        print(f"\nProcess peak device memory {r.get('process_peak_gb', 0):.1f} GB; elapsed {r['elapsed_seconds'] / 60:.1f} min.")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        main(sys.argv[1:])
