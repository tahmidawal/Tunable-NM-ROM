"""Dev selection for the rank lane (PROTOCOL A5.2). Pure JSON, no GPU.

Usage: freeze_rank.py <out_dir> <mesh>=<romdev.json>[+<romdev.json>...]:<parent frozen.json> ...
Writes <out_dir>/frozen_<mesh>.json and prints the floors, the contingency test and the choice.
"""
import json, sys
from pathlib import Path

KEYS = ("kind", "rank", "k", "width", "M", "steps", "iters")


def select(rows, slack):
    ok = [r for r in rows if r["stats"]["finite"]]
    if not ok:
        return None
    best = min(r["stats"]["evolved_worst"] for r in ok)
    near = [r for r in ok if r["stats"]["evolved_worst"] <= slack * best]
    return min(near, key=lambda r: (r["steps"] * r["iters"], r["M"], r.get("k", 0), r.get("width", 0)))


def cfg(r):
    return {k: r[k] for k in KEYS if k in r} | dict(dev_worst=r["stats"]["evolved_worst"],
                                                    dev_median=r["stats"]["evolved_median"])


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for item in sys.argv[2:]:
        mesh, rest = item.split("=", 1); files, parent = rest.split(":")
        reps = [json.load(open(f)) for f in files.split("+")]
        rows = [r for rep in reps for r in rep.get("rollouts", [])]
        arms = {}
        pf = json.load(open(parent))
        for name in ("head", "head_fast"):
            h = pf[name]
            c = dict(kind="head", rank=64, k=h["k"], width=256, M=h["M"], steps=h["steps"], iters=h["iters"],
                     dev_worst=h["dev_worst"], dev_median=h["dev_median"])
            if name == "head_fast" and all(c[k] == arms["r64_head"][k] for k in KEYS):
                continue
            arms[f"r64_{name}"] = c
        s = pf["spans"]["64"]
        arms["r64_span"] = dict(kind="span", rank=64, M=s["M"], steps=s["steps"], iters=s["iters"],
                                dev_worst=s["dev_worst"], dev_median=s["dev_median"])
        ranks = sorted({r["rank"] for r in rows})
        print(f"\n## mesh {mesh}")
        for rep in reps:
            for r, sec in rep["ranks"].items():
                fl = sec["floors"]; sp = fl["span"]["evolved_worst"]
                heads = {k: v["evolved_worst"] for k, v in fl.items() if k != "span"}
                trig = bool(heads) and all(v > 1.15 * sp for v in heads.values())
                print(f"rank {r}: dev span floor worst {100 * sp:.3f} % median {100 * fl['span']['evolved_median']:.3f} %; "
                      + "; ".join(f"head {k} floor worst {100 * v['evolved_worst']:.3f} % median {100 * v['evolved_median']:.3f} %"
                                  for k, v in fl.items() if k != "span")
                      + f"; contingency (all heads > 1.15 x span floor): {trig}")
        for r in ranks:
            hr = [x for x in rows if x["rank"] == r and x["kind"] == "head"]
            sr = [x for x in rows if x["rank"] == r and x["kind"] == "span"]
            h = select(hr, 1.05); f = select(hr, 1.25); s = select(sr, 1.05)
            if h: arms[f"r{r}_head"] = cfg(h)
            if f and any(f[k] != h[k] for k in KEYS if k in h): arms[f"r{r}_head_fast"] = cfg(f)
            if s: arms[f"r{r}_span"] = cfg(s)
        for name, c in arms.items():
            print(f"{name}: " + ", ".join(f"{k}={c[k]}" for k in KEYS if k in c)
                  + f" -> dev worst {100 * c['dev_worst']:.3f} % median {100 * c['dev_median']:.3f} %")
        tmp = out / f"frozen_{mesh}.json"
        tmp.write_text(json.dumps(dict(family=pf["family"], mesh=int(mesh), sources=files.split("+"), parent=parent,
                                       arms=arms), indent=1))


if __name__ == "__main__":
    main()
