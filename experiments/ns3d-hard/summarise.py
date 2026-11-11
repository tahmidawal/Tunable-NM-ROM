"""Headline table from saved compare.json files (no number is typed by hand).

Usage: summarise.py <label>=<compare.json> [<label>=<compare.json> ...] > table.md
"""
import json, sys


def cell(c):
    if not c:
        return "none"
    return f"{c['ratio']:.2f}x ({c['scheme']} {c['n']}^3 x{c['steps']}, {c['ms']:.1f} ms, {100 * c['cohort_worst']:.2f} %)"


def main():
    for item in sys.argv[1:]:
        label, path = item.split("=", 1)
        rep = json.load(open(path))
        print(f"\n### {label}: {rep['cases']} cases, job {rep['job_id']}, commit {str(rep['source_commit'])[:9]}, "
              f"{rep['gpu']}, status {rep.get('status', 'incomplete')}, failures {rep['failures']}\n")
        print("| mesh | arm | setting | worst err | median err | band | ROM ms (min-max) | any-mesh solve-only | "
              "any-mesh end-to-end | same-mesh solve-only | same-mesh end-to-end | dev-chosen solve-only |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n, m in rep["meshes"].items():
            for name, e in m["arms"].items():
                lab = rep.get("arm_labels") or {}
                if name.startswith("interp_"):
                    name = lab.get("interp", "interp_") + name[len("interp_"):]
                elif not name.startswith("pod_"):
                    name = lab.get("main", "") + name
                cfg = e["config"]
                setting = ", ".join(f"{k}={cfg[k]}" for k in ("k", "width", "M", "steps") if k in cfg)
                st = e["stats"]
                if not st["finite"]:
                    print(f"| {n} | {name} | {setting} | non-finite | | | | | | | | |"); continue
                ms = e.get("ms")
                mss = f"{ms['median_ms']:.2f} ({ms['min_ms']:.2f}-{ms['max_ms']:.2f})" if ms else "not timed"
                print(f"| {n} | {name} | {setting} | {100 * st['evolved_worst']:.2f} % | {100 * st['evolved_median']:.2f} % | "
                      f"{e['band']} | {mss} | {cell(e.get('any_mesh_solve'))} | {cell(e.get('any_mesh_e2e'))} | "
                      f"{cell(e.get('same_mesh_solve'))} | {cell(e.get('same_mesh_e2e'))} | {cell(e.get('dev_chosen_solve'))} |")
        for n, m in rep["meshes"].items():
            t = m.get("timing")
            if t:
                print(f"\nmesh {n}: timing attempts {m['timing_attempts']}, gate passed {t['gate_passed']}, "
                      f"ROM agreement {t['agreement_rom']:.1e}, solver agreement {t['agreement_solver']:.1e}")


if __name__ == "__main__":
    main()
