"""Plot audited initial-fit improvement and same-job complete query frontier."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("audit", type=Path); p.add_argument("--out", type=Path, required=True)
args = p.parse_args(); a = json.loads(args.audit.read_text())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
rows = a["reconstruction"]; x = np.arange(len(rows))
colors = ["#555555" if row["model"] == "frozen" else "#376ba2" if row["model"].startswith("original") else "#b35423" for row in rows]
ax[0].bar(x, [100*row["initial_worst"] for row in rows], color=colors, alpha=.85, label="worst case")
ax[0].scatter(x, [100*row["initial_median"] for row in rows], color="black", marker="D", s=20, label="median")
ax[0].set_xticks(x, [row["model"].replace("_seed", "\n") for row in rows], fontsize=8)
ax[0].set(ylabel="Initial field relative error (%)", title="Matched optimization; different coverage")
ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.2)
for n in sorted(set(row["intervals"] for row in a["rollout_groups"])):
    for row in a["rollout_groups"]:
        if row["intervals"] != n: continue
        color = "#367044" if row["model"] == "fom" else "#555555" if row["model"] == "frozen" else "#b35423"
        marker = "o" if n == min(g["intervals"] for g in a["rollout_groups"]) else "s"
        ax[1].scatter(1000*row["median_query_seconds"], 100*row["error_worst"], color=color, marker=marker,
                      s=35, alpha=.85)
ax[1].set(xlabel="Complete query median (ms)", ylabel="Worst rollout current-relative error (%)",
          title="Same-job full input/output costs")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].grid(alpha=.2)
from matplotlib.lines import Line2D
legend = [Line2D([], [], color=c, marker="o", linestyle="", label=l) for c, l in
          [("#367044", "direct DST FOM"), ("#555555", "frozen head"), ("#b35423", "expanded coverage")]]
for n, marker in zip(sorted(set(row["intervals"] for row in a["rollout_groups"])), ("o", "s")):
    legend.append(Line2D([], [], color="black", marker=marker, linestyle="", label=f"{n} solver intervals"))
ax[1].legend(handles=legend, fontsize=8)
fig.suptitle("Fixed heat bank: expanded coverage improves accuracy; direct FOM remains faster", fontsize=12)
args.out.parent.mkdir(parents=True, exist_ok=True)
for suffix in ("png", "svg"): fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=180)
args.out.with_suffix(".json").write_text(json.dumps(dict(source_audit=str(args.audit),
    audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
    note="Restricted development cohort; circle/square distinguish the two solver meshes; both declared tolerances retained."), indent=2)+"\n")
