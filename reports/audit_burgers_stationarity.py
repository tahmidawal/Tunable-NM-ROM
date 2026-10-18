"""CPU diagnosis of saved Burgers iterates; no new PDE solves or timings."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/iterative06"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head(p, z, derivative=False):
    x = z
    jac = np.eye(len(z))
    if "hB" in p:
        a = 2 * np.pi * (z @ p["hB"])
        x = np.concatenate((z, np.sin(a), np.cos(a)))
        jac = np.concatenate((jac, np.cos(a)[:, None] * (2*np.pi*p["hB"].T),
                              -np.sin(a)[:, None] * (2*np.pi*p["hB"].T)))
    for i, (w, b) in enumerate(p["h"]):
        x = x @ w + b
        if derivative:
            jac = w.T @ jac
        if i < len(p["h"]) - 1:
            s = 1 / (1 + np.exp(-x)) if np.iscomplexobj(x) else expit(x)
            if derivative:
                jac = (s + x*s*(1-s))[:, None] * jac
            x = x*s
    value = x + z @ p["h_lin"]
    return (value, jac + p["h_lin"].T) if derivative else value


def metric(residual, jac):
    gradient = jac.T @ residual
    return dict(residual_norm=float(np.linalg.norm(residual)),
                gradient_norm=float(np.linalg.norm(gradient)),
                normalized_gradient=float(np.linalg.norm(gradient) /
                    max(np.linalg.norm(jac)*np.linalg.norm(residual), 1e-300)))


def sample(field, xy, n):
    x = xy*n
    lo = np.minimum(np.floor(x).astype(int), n-1)
    t = x-lo
    i, j = lo.T
    a, b = t.T
    return ((1-a)*(1-b)*field[i,j] + a*(1-b)*field[i+1,j]
            + (1-a)*b*field[i,j+1] + a*b*field[i+1,j+1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/2026-09-11-burgers-stationarity-diagnostic.json")
    args = ap.parse_args()
    record = args.record
    run = record / "archive"
    result = run / "out/result.json"
    checkpoint = run / "in/checkpoint.pkl"
    audit = record / "AUDIT.json"
    d = json.loads(result.read_text())
    assert json.loads(audit.read_text())["passed"]
    assert d["complete"] and d["backend"] == "gpu" and d["x64"]
    assert d["checkpoint_sha256"] == sha(checkpoint)
    p = pickle.loads(checkpoint.read_bytes())["params"]
    sources = {str(x.relative_to(ROOT)): sha(x) for x in (result, checkpoint, audit, Path(__file__))}
    rows = []
    derivative_errors = []
    weak_derivative_errors = []
    residual_errors = []
    sign_distances = []
    physical = np.array(d["physical_cases"])
    for n in d["config"]["meshes"]:
        opath = run / f"out/operators_L{n}.npz"
        sources[str(opath.relative_to(ROOT))] = sha(opath)
        with np.load(opath) as f:
            op = {k:f[k] for k in f.files}
        for old in d["invocations"]:
            if old["intervals"] != n or old["name"] != "nmrom" or old["rep"] != 0:
                continue
            case = old["case"]
            states = np.array(old["internal_latents"])
            h0, j0 = head(p, states[0], True)
            # Independent complex-step check of the analytic head derivative.
            complex_jac = np.column_stack([
                head(p, states[0].astype(complex) + 1e-25j*np.eye(len(states[0]))[i]).imag/1e-25
                for i in range(len(states[0]))])
            derivative_errors.append(float(np.linalg.norm(j0-complex_jac)/np.linalg.norm(complex_jac)))
            cx, cy, width, amp, nu = physical[case]
            xx, yy = np.meshgrid(np.arange(n+1)/n, np.arange(n+1)/n, indexing="ij")
            u0 = amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*width**2))
            u0[[0,-1],:] = 0
            u0[:,[0,-1]] = 0
            target = op["cold_Q"].T @ (sample(u0, op["cold_xy"], n)*op["cold_w"])
            r = op["cold_R"]@h0-target
            rows.append(dict(intervals=n, case=case, stage="initial", step=0,
                             stop_reason=old["ic_reason"], **metric(r, op["cold_R"]@j0)))
            dt = old["dt"]
            previous = op["A"]@h0
            for i, z in enumerate(states[1:], 1):
                h, dh = head(p, z, True)
                us = np.einsum("msr,r->ms", op["G5"], h)
                dus = np.einsum("msr,rk->msk", op["G5"], dh)
                c, xp, xm, yp, ym = us.T
                dc, dxp, dxm, dyp, dym = dus.transpose(1,0,2)
                diff = np.where(c>0, 2*c-xm-ym, xp+yp-2*c)
                ddiff = np.where((c>0)[:,None], 2*dc-dxm-dym, dxp+dyp-2*dc)
                adv = n*c*diff
                dadv = n*(dc*diff[:,None]+c[:,None]*ddiff)
                ah = op["A"]@h
                dah = op["A"]@dh
                scale = 1+dt*nu*op["lam"]
                r = (ah-previous+dt*(op["Pq"].T@adv+nu*op["lam"]*ah))/scale
                jac = (dah+dt*(op["Pq"].T@dadv+nu*op["lam"][:,None]*dah))/scale[:,None]
                if i == 1:
                    def weak_complex(zz):
                        hh = head(p, zz)
                        uu = np.einsum("msr,r->ms", op["G5"], hh)
                        cc, xpp, xmm, ypp, ymm = uu.T
                        aa = n*cc*np.where(c>0, 2*cc-xmm-ymm, xpp+ypp-2*cc)
                        ahh = op["A"]@hh
                        return (ahh-previous+dt*(op["Pq"].T@aa+nu*op["lam"]*ahh))/scale
                    check_jac = np.column_stack([
                        weak_complex(z.astype(complex)+1e-25j*np.eye(len(z))[j]).imag/1e-25
                        for j in range(len(z))])
                    weak_derivative_errors.append(float(np.linalg.norm(jac-check_jac)/np.linalg.norm(check_jac)))
                previous = ah
                rows.append(dict(intervals=n, case=case, stage="evolution", step=i,
                                 stop_reason=old["stop_reasons"][i-1], **metric(r,jac)))
                residual_errors.append(abs(np.linalg.norm(r)-old["residuals"][i-1]))
                sign_distances.append(float(np.min(abs(c))))
    assert max(derivative_errors) < 1e-11
    assert max(weak_derivative_errors) < 1e-10
    assert max(residual_errors) < 2e-10
    assert all(np.isfinite(r["normalized_gradient"]) for r in rows)
    groups = []
    for n in d["config"]["meshes"]:
        for stage in ("initial", "evolution"):
            selected = [r for r in rows if r["intervals"] == n and r["stage"] == stage]
            v = np.array([r["normalized_gradient"] for r in selected])
            groups.append(dict(intervals=n, stage=stage, count=len(v),
                               median=float(np.median(v)), maximum=float(v.max()),
                               at_or_below_1e_6=int(np.sum(v<=1e-6))))
    out = dict(status="independent CPU diagnostic of archived GPU endpoints",
               source_job=d["job_id"], source_commit=d["commit"], source_hashes=sources,
               definition="norm(J.T r)/(norm(J, Frobenius)*norm(r)); 1e-6 is a diagnostic threshold, not a retroactive change to original stopping contract",
               scope="One retained deterministic repetition per mesh/case; initial fit and every saved evolution endpoint; no new solver or timing",
               caveat="Upwind derivative uses recorded local sign branch; distances to zero are retained. Initial source is reconstructed from recorded parameters with floating-point platform variation.",
               head_jacobian_complex_step_max_relative_error=max(derivative_errors),
               weak_jacobian_complex_step_max_relative_error=max(weak_derivative_errors),
               archived_weak_residual_max_abs_difference=max(residual_errors),
               minimum_upwind_center_abs=min(sign_distances), groups=groups, rows=rows)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ("rows", "source_hashes")}, indent=2))


if __name__ == "__main__":
    main()
