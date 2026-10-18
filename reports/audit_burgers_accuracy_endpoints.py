"""Second CPU implementation of worst Burgers trajectories and selected gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from audit_burgers_stationarity import head, metric, sample
from audit_poisson_online_tuning import features

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/accuracy08"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    run = run.resolve()
    out = run/"archive/out"
    result_path = out/"result.json"
    d = json.loads(result_path.read_text())
    owner_path = run/"AUDIT.json"
    owner = json.loads(owner_path.read_text())
    assert d["complete"] and owner["passed"]
    assert owner["result_sha256"] == sha(result_path)
    assert d["backend"] == "gpu" and d["x64"] and d["matmul_precision"] == "highest"
    assert d["final_cohort_unopened"]
    checkpoints = {"frozen": run/"archive/in/checkpoint.pkl", "trained": out/"trained_checkpoint.pkl"}
    assert sha(checkpoints["frozen"]) == d["checkpoint_sha256"]
    assert sha(checkpoints["trained"]) == d["trained_checkpoint_sha256"]
    if d.get("coverage_training_performed"):
        checkpoints["trained576"] = checkpoints.pop("trained")
        checkpoints["trained4608"] = out/"coverage_training/trained_checkpoint.pkl"
        assert sha(checkpoints["trained4608"]) == d["coverage_trained_checkpoint_sha256"]
    evidence = []
    for model, checkpoint in checkpoints.items():
        p = pickle.loads(checkpoint.read_bytes())["params"]
        for n in d["config"]["meshes"]:
            op = dict(np.load(out/f"operators_{model}_L{n}.npz"))
            solvers = sorted({r["solver"] for r in d["invocations"] if r.get("model") == model and r["intervals"] == n})
            assert solvers
            for solver in solvers:
                name = model+"_"+solver
                rows = [r for r in d["invocations"] if r["intervals"] == n and r["name"] == name]
                row = max(rows, key=lambda r: r["error"]["fixed_initial_max"])
                case = row["case"]
                f = np.load(out/row["artifact"])["fields"]
                assert f.dtype == np.float64 and np.isfinite(f).all()
                assert hashlib.sha256(f.tobytes()).hexdigest() == row["field_sha256"]
                reference = next(r for r in d["reference"] if r["case"] == case
                                 and r["intervals"] == d["config"]["reference_mesh"]
                                 and r["dt"] == d["config"]["reference_dt"])
                stride = max(d["config"]["meshes"])//n
                truth = np.load(out/reference["artifact"])["fields"][:, ::stride, ::stride]
                norm = np.linalg.norm(truth[0])
                errors = np.linalg.norm((f-truth).reshape(len(f), -1), axis=1)/norm
                np.testing.assert_allclose(errors, row["error"]["fixed_initial_per_time"], atol=1e-12, rtol=1e-12)
                ij = np.random.default_rng(911103).integers(1, n, (113, 2))
                bank = features(p, ij/n)
                predicted = np.array([bank@head(p, np.asarray(z)) for z in row["latent_states"]])
                saved = f[:, ij[:, 0], ij[:, 1]]
                parity = float(np.linalg.norm(predicted-saved)/np.linalg.norm(saved))
                assert parity < 2e-10
                cx, cy, width, amp, nu = d["physical_cases"][case]
                x, y = np.meshgrid(np.arange(n+1)/n, np.arange(n+1)/n, indexing="ij")
                u0 = amp*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
                u0[[0, -1], :] = 0; u0[:, [0, -1]] = 0
                states = np.asarray(row["internal_latents"])
                h0, j0 = head(p, states[0], True)
                target = op["cold_Q"].T@(sample(u0, op["cold_xy"], n)*op["cold_w"])
                initial = metric(op["cold_R"]@h0-target, op["cold_R"]@j0)["normalized_gradient"]
                assert abs(initial-row["ic_normalized_stationarity"]) < 2e-9
                checks = [dict(step=0, cpu=initial, posthoc=row["ic_normalized_stationarity"],
                               charged=row.get("charged_ic_normalized_stationarity"))]
                selected = sorted({1, len(states)-1, int(np.argmax(row["normalized_stationarity"]))+1})
                dt = row["dt"]
                for i in selected:
                    h, j = head(p, states[i], True)
                    u = np.einsum("msr,r->ms", op["G5"], h)
                    du = np.einsum("msr,rk->msk", op["G5"], j)
                    c, xp, xm, yp, ym = u.T
                    dc, dxp, dxm, dyp, dym = du.transpose(1, 0, 2)
                    diff = np.where(c > 0, 2*c-xm-ym, xp+yp-2*c)
                    ddiff = np.where((c > 0)[:, None], 2*dc-dxm-dym, dxp+dyp-2*dc)
                    adv, dadv = n*c*diff, n*(dc*diff[:, None]+c[:, None]*ddiff)
                    ah, aj = op["A"]@h, op["A"]@j
                    previous = op["A"]@head(p, states[i-1])
                    scale = 1+dt*nu*op["lam"]
                    residual = (ah-previous+dt*(op["Pq"].T@adv+nu*op["lam"]*ah))/scale
                    jac = (aj+dt*(op["Pq"].T@dadv+nu*op["lam"][:, None]*aj))/scale[:, None]
                    g = metric(residual, jac)["normalized_gradient"]
                    post = row["normalized_stationarity"][i-1]
                    assert abs(g-post) < 2e-9
                    charged = row.get("charged_normalized_stationarity")
                    checks.append(dict(step=i, cpu=g, posthoc=post, charged=charged[i-1] if charged is not None else None))
                evidence.append(dict(model=name, intervals=n, case=case, cohort=row["cohort"],
                                     worst_fixed_initial_error=float(max(errors)), decoder_relative_error=parity,
                                     field_sha256=row["field_sha256"], checkpoint_sha256=sha(checkpoint),
                                     selected_gradient_checks=checks))
                print(name, n, case, max(errors), flush=True)
    return dict(passed=True, source_result_sha256=sha(result_path), owner_audit_sha256=sha(owner_path),
                source_script_sha256=sha(Path(__file__)),
                helper_sha256={p.name: sha(p) for p in (ROOT/"reports/audit_burgers_stationarity.py", ROOT/"reports/audit_poisson_online_tuning.py")},
                source_result_path=str(result_path.relative_to(ROOT)), job_id=d["job_id"], source_commit=d["commit"],
                scope="Worst trajectory per model/mesh: all output errors and decoded samples; initial, first, last and largest recorded-gradient weak states. Full owner audit remains required. No new solve or timing.",
                checked_trajectories=len(evidence), endpoints=evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/2026-09-11-burgers-accuracy.coordinator-audit.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.run), indent=2)+"\n")
