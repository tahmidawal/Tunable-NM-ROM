"""Independent NumPy reconstruction audit of stored ROM field observations.

This audits coefficient/gauge/output correspondence at the saved observation
mesh. It does not independently regenerate the fine-grid truth trajectories.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main(record):
    cluster = record/"cluster"
    native = cluster/"out/pilot"
    result = json.loads((native/"result.json").read_text())
    cfg = result["config"]
    rows = []
    for bc in cfg["boundaries"]:
        xx = np.linspace(0, 1, cfg["saved_intervals"]+1)
        if bc == "dirichlet":
            xx = xx[1:-1]
        xy = np.stack(np.meshgrid(xx, xx, indexing="ij"), -1).reshape(-1, 2)
        with np.load(cluster/"in"/bc/"bank_parameters.npz") as p, np.load(cluster/"in"/bc/"coordinates.npz") as co:
            phase = 2*np.pi*xy@p["frequency"].T
            h = np.concatenate((2*xy-1, np.sin(phase), np.cos(phase)), -1)
            for layer in ("l1", "l2"):
                h = h@p[f"p/{layer}/w"]+p[f"p/{layer}/b"]
                h = h/(1+np.exp(-h))
            raw = h@p["p/out/w"]+p["p/out/b"]
            if bc == "dirichlet":
                raw *= (np.sin(np.pi*xy[:, 0])*np.sin(np.pi*xy[:, 1]))[:, None]
            original_g = np.linalg.solve(co["qr_r"].T, raw.T).T
        for n in cfg["meshes"]:
            with np.load(native/f"mesh_{bc}_{n}.npz") as mesh:
                g = np.linalg.solve(mesh["coordinate_transform"].T, original_g.T).T
            for case in cfg["validation_indices"]:
                for dt in cfg["rom_dts"]:
                    name = f"{bc}_{n}_{case}_0_rom_{dt}"
                    with np.load(native/(name+".npz")) as f:
                        du = float(np.max(abs((f["coefficients"]@g.T).reshape(f["u"].shape)-f["u"])))
                        dv = float(np.max(abs((f["velocity_coefficients"]@g.T).reshape(f["v"].shape)-f["v"])))
                    rows.append(dict(invocation_id=name, maximum_displacement_difference=du, maximum_velocity_difference=dv))
    query_audits = []
    for row in result["invocations"]:
        components = sum(v for k,v in row["seconds"].items() if k != "complete_query")
        nodes = (row["intervals"]-1)**2 if row["boundary"] == "dirichlet" else (row["intervals"]+1)**2
        query_audits.append(dict(invocation_id=row["invocation_id"], timing_sum_difference=abs(components-row["seconds"]["complete_query"]),
            output_bytes_match=row["output_bytes"] == 2*8*len(result["observations"])*nodes))
    passed = all(max(r["maximum_displacement_difference"], r["maximum_velocity_difference"]) < 1e-10 for r in rows)
    passed &= all(r["timing_sum_difference"] < 1e-12 and r["output_bytes_match"] for r in query_audits)
    audit = dict(passed=passed, scope=__doc__, reconstruction=rows, query_accounting=query_audits)
    (record/"analysis/saved-field-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    if not passed:
        raise RuntimeError("Saved physical field or query accounting mismatch")
    print(json.dumps(dict(passed=passed, reconstructed_outputs=len(rows), queried_outputs=len(query_audits))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    main(parser.parse_args().record)
