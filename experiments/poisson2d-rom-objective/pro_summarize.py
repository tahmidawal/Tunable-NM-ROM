"""Markdown tables from the study JSONs.  Usage: python pro_summarize.py <runs_dir>"""
import glob
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"


def f(x):
    return "—" if x is None else f"{x:.2e}"


def obj_table(path):
    r = json.load(open(path))
    m = r["manifest"]
    print(f"\n### {os.path.basename(path)}  (pkl {m['pkl']}, K={m['pkl_config']['K_LAT']}, N={m['pkl_config']['N']}, "
          f"stages {m['ns']}, budget {m['gn_iters']}, hard_bc {m['hard_bc']}, complete={r['complete']})")
    print("oracle (data-misfit LM, same init & budget): " +
          ", ".join(f"{k} {f(v['rel_l2_mean'])}" for k, v in r["oracle"].items()))
    if r.get("encoder"):
        print(f"encoder plug-in (no solve): {f(r['encoder']['plugin_rel_l2_mean'])}")
    inits = list(r["oracle"].keys())
    print("\n| objective | modes | " + " | ".join(f"{i}: ROM mean (med) / oracle" for i in inits) +
          " | obj(z_LM)<=obj(z_or) | acc/att |")
    print("|---|---|" + "---|" * len(inits) + "---|---|")
    by = {}
    for row in r["rows"]:
        by.setdefault(row["objective"], {})[row["init"]] = row
    for o, d in by.items():
        cells = []
        for i in inits:
            row = d.get(i)
            cells.append("—" if row is None else
                         f"{f(row['rom_rel_l2_mean'])} ({f(row['rom_rel_l2_med'])}) / {f(row['oracle_rel_l2_mean'])}")
        anyrow = next(iter(d.values()))
        le = "/".join(str(d[i]["n_obj_lm_le_oracle"]) if i in d else "—" for i in inits)
        acc = "/".join(f"{d[i]['lm_accepted_med']:.0f}:{d[i]['lm_attempts_med']:.0f}" if i in d else "—" for i in inits)
        print(f"| {o} | {anyrow.get('n_modes_retained') or 'all'} | " + " | ".join(cells) + f" | {le} | {acc} |")


def colloc_table(path):
    r = json.load(open(path))
    m = r["manifest"]
    print(f"\n### {os.path.basename(path)}  (pkl {m['pkl']}, K={m['pkl_config']['K_LAT']}, budget {m['gn_iters']}, "
          f"hard_bc {m['hard_bc']}, bc_beta {m.get('bc_beta')}, complete={r['complete']})")
    print("oracle: " + ", ".join(f"{k} {f(v)}" for k, v in r["oracle"].items()))
    inits = m["inits"]
    print("\n| objective | scheme | m | " + " | ".join(f"{i}: ROM mean (med)" for i in inits) + " | acc/att | EQ info |")
    print("|---|---|---|" + "---|" * len(inits) + "---|---|")
    by = {}
    for row in r["rows"]:
        by.setdefault((row["objective"], row["scheme"], row["m"]), {})[row["init"]] = row
    for (o, s, mm), d in by.items():
        cells = ["—" if i not in d else f"{f(d[i]['rom_rel_l2_mean'])} ({f(d[i]['rom_rel_l2_med'])})" for i in inits]
        acc = "/".join(f"{d[i]['lm_accepted_med']:.0f}:{d[i]['lm_attempts_med']:.0f}" if i in d else "—" for i in inits)
        eq = next(iter(d.values())).get("eq_info")
        eqs = "" if not eq else f"supp {eq['support']}+{eq['padded']} rn {eq['rnorm_final']:.1e}/{eq['b_norm']:.1e}"
        print(f"| {o} | {s} | {mm} | " + " | ".join(cells) + f" | {acc} | {eqs} |")


def train_table(path):
    r = json.load(open(path))
    c = r["config"]
    print(f"\n### {os.path.basename(path)}  K={c['K_LAT']} N={c['N']} hard_bc={c['hard_bc']}: train {f(r['train_global_rel'])}/"
          f"{f(r['train_mean_rel_l2'])}, val LM-inferred " +
          ", ".join(f"{k} {f(v)}" for k, v in r["val_lm_inferred_mean_rel_l2"].items()) +
          f", boundary block {r['boundary_block_train_med']:.1e}, n_freq {r['n_freq']}")


for p in sorted(glob.glob(os.path.join(RUNS, "**", "*.json"), recursive=True)):
    b = os.path.basename(p)
    if b.startswith("obj_"):
        obj_table(p)
    elif b.startswith("colloc_"):
        colloc_table(p)
    elif b.startswith("autodec_"):
        train_table(p)
