"""Task 1: how fine must an accuracy-matched pseudo-spectral FOM be, and how large
must a centred POD bank be, for one candidate family?  See PROTOCOL.md section 1.

Outputs (in --out): characterise.json (everything), log on stdout.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ns3d-grok"))
import hard_fom as H
from hard_fom import F
import diag_floor as DF
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
log = DF.log


def run_ref(u0, nu, n, dt, horizon, scheme="cnab2", frames=5):
    steps = DF.nsteps_for(dt, horizon)
    solver = F.make_solver(dt, steps, steps // frames, scheme=scheme)
    return solver(jnp.asarray(u0), float(nu), F.geometry(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-n", type=int, default=192)
    ap.add_argument("--ref-dt", type=float, default=2.5e-4)
    ap.add_argument("--check-n", type=int, default=256)
    ap.add_argument("--probe-cases", type=int, default=6)
    ap.add_argument("--check-cases", type=int, default=2)
    ap.add_argument("--meshes", default="16,24,32,48,64,96,128")
    ap.add_argument("--steps", default="5,10,20,40,80,160,320,640")
    ap.add_argument("--pod-n", type=int, default=96)
    ap.add_argument("--pod-dt", type=float, default=5e-4)
    ap.add_argument("--pod-train", type=int, default=128)
    ap.add_argument("--pod-held", type=int, default=16)
    ap.add_argument("--pod-rank", type=int, default=128)
    ap.add_argument("--seed-train", type=int, default=2026111101)
    ap.add_argument("--seed-probe", type=int, default=2026111102)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = H.FAMILIES[a.family]; T = fam["horizon"]
    report = dict(family=a.family, family_def=fam, args=vars(a), backend=jax.default_backend(),
                  device=str(jax.devices()[0]))
    if a.family == "A":   # exact reproduction of the paper's family
        p0 = F.parameters(7, 3); p1 = H.parameters(7, 3, fam)
        d = float(np.max(np.abs(F.initial_raw(24, p0[1]) - H.initial_raw(24, p1[1], fam))))
        report["paper_family_reproduction"] = dict(param_diff=float(np.max(np.abs(p0 - p1))), field_diff=d)
        assert np.max(np.abs(p0 - p1)) == 0 and d < 1e-14
    probe = H.parameters(a.seed_probe, max(a.probe_cases, a.pod_held), fam)
    N = a.ref_n
    meshes = [int(x) for x in a.meshes.split(",")]
    steps_list = [int(x) for x in a.steps.split(",")]

    # ---------------- reference + self-convergence -------------------------------
    ladder = {}
    ref_checks = []; descr = []
    solvers = {}
    for ci in range(a.probe_cases):
        p = probe[ci]; nu = float(p[-1])
        u0 = H.initial(N, p, fam)
        t0 = time.time()
        ref = run_ref(u0, nu, N, a.ref_dt, T); ref.block_until_ready()
        t_ref = time.time() - t0
        ref_hat = F.fft(ref)
        ref_tot2 = jnp.sum(jnp.abs(ref_hat) ** 2, axis=(1, 2, 3, 4))
        u0n2 = float(ref_tot2[0])
        # descriptors
        k2 = F.geometry(N)[1]
        ens = np.asarray(jnp.sum(k2 * jnp.sum(jnp.abs(ref_hat) ** 2, axis=1), axis=(1, 2, 3)))
        en = np.asarray(ref_tot2)
        kmag = np.sqrt(np.asarray(k2)) / (2 * np.pi)
        e_last = np.asarray(jnp.sum(jnp.abs(ref_hat[-1]) ** 2, axis=0))
        tail = {str(c): float(np.sqrt(e_last[kmag >= c].sum() / e_last.sum())) for c in (8, 10, 16, 21, 32, 42, 48)}
        umax = [float(jnp.max(jnp.sqrt(jnp.sum(ref[i] ** 2, axis=0)))) for i in (0, ref.shape[0] - 1)]
        descr.append(dict(case=ci, nu=nu, umax_first_last=umax, energy_ratio=float(en[-1] / en[0]),
                          enstrophy_ratio_max=float(ens.max() / ens[0]), enstrophy_ratio_last=float(ens[-1] / ens[0]),
                          final_tail_fraction_beyond_k=tail, ref_seconds_incl_compile=t_ref))
        log(f"case {ci} nu={nu:.2e} ref {t_ref:.1f}s umax={umax} ens_max={ens.max()/ens[0]:.3f} tail={tail}")
        if ci < a.check_cases and not a.smoke:
            chk = {}
            # finer mesh and half step
            M = a.check_n
            fine = run_ref(H.initial(M, p, fam), nu, M, a.ref_dt / 2, T)
            fh = F.fft(fine); ft2 = jnp.sum(jnp.abs(fh) ** 2, axis=(1, 2, 3, 4))
            chk["ref_vs_finer_mesh_half_dt"] = [float(x) for x in H.full_errors(ref, fh, ft2, float(ft2[0]))]
            del fine, fh
            rk = run_ref(u0, nu, N, a.ref_dt, T, scheme="rk4")
            chk["ref_vs_rk4_same_mesh"] = [float(x) for x in H.full_errors(rk, ref_hat, ref_tot2, u0n2)]
            del rk
            ref_checks.append(dict(case=ci, **chk)); log(f"ref check {chk}")
        # ---------------- FOM ladder -------------------------------------------
        for n in meshes:
            geom = F.geometry(n)
            u0_n = np.asarray(F.ifft(H.truncate_hat(ref_hat[0], n)))   # best available IC
            u0_n = jnp.asarray(u0_n)
            for scheme in ("cnab2", "rk4"):
                for s in steps_list:
                    if scheme == "rk4" and s > 320: continue
                    key = f"{scheme}|{n}|{s}"
                    if key not in solvers:
                        solvers[key] = F.make_solver(T / s, s, s // 5, scheme=scheme)
                        w = solvers[key](u0_n, nu, geom); w.block_until_ready()   # compile
                    best = np.inf
                    for _ in range(3):
                        t0 = time.perf_counter()
                        pred = solvers[key](u0_n, nu, geom); pred.block_until_ready()
                        best = min(best, time.perf_counter() - t0)
                    if bool(jnp.all(jnp.isfinite(pred))):
                        err = H.full_errors(pred, ref_hat, ref_tot2, u0n2)
                    else:
                        err = np.full(6, np.inf)
                    ladder.setdefault(key, dict(scheme=scheme, n=n, steps=s, errors=[], seconds=[]))
                    ladder[key]["errors"].append([float(x) for x in err])
                    ladder[key]["seconds"].append(best)
            log(f"case {ci} mesh {n} done: " + " ".join(
                f"{s}:{max(ladder[f'cnab2|{n}|{s}']['errors'][-1][1:]):.2e}" for s in steps_list))
        del ref, ref_hat
    rows = []
    for key, v in ladder.items():
        e = np.asarray(v["errors"]); w = e[:, 1:].max(axis=1)
        rows.append(dict(scheme=v["scheme"], n=v["n"], steps=v["steps"], evolved_worst=float(w.max()),
                         evolved_median=float(np.median(w)), seconds_median=float(np.median(v["seconds"])),
                         seconds_min=float(np.min(v["seconds"])), per_case_worst=[float(x) for x in w]))
    cheapest = {}
    for target in (0.002, 0.01, 0.03):
        for stat in ("evolved_worst", "evolved_median"):
            ok = [r for r in rows if r[stat] <= target]
            b = min(ok, key=lambda r: r["seconds_median"]) if ok else None
            cheapest[f"{stat}<={target}"] = b
            log(f"cheapest {stat}<={target}: " + (f"{b['scheme']} n={b['n']} steps={b['steps']} err={b[stat]:.3e} t={b['seconds_median']*1e3:.1f}ms" if b else "none"))
        # smallest mesh able to reach target with any step count
        ok = sorted({r["n"] for r in rows if r["evolved_worst"] <= target})
        cheapest[f"min_mesh_worst<={target}"] = ok[0] if ok else None
    report.update(reference=dict(n=N, dt=a.ref_dt, checks=ref_checks), descriptors=descr,
                  ladder=rows, cheapest=cheapest)
    (out / "characterise.json").write_text(json.dumps(report, indent=1))

    # ---------------- centred POD floor vs rank -----------------------------------
    n = a.pod_n; geom = F.geometry(n)
    steps = DF.nsteps_for(a.pod_dt, T)
    solver = F.make_solver(a.pod_dt, steps, steps // 5)
    train = H.parameters(a.seed_train, a.pod_train, fam)
    def centred_frames(params):
        fr = np.empty((len(params), 6, 3, n, n, n)); 
        for i, p in enumerate(params):
            x = np.asarray(solver(jnp.asarray(H.initial(n, p, fam)), float(p[-1]), geom))
            for t in range(6):
                fr[i, t] = DF.center_field(x[t])[0]
            if i % 32 == 0: log(f"pod data {i}/{len(params)}")
        return fr
    tr = centred_frames(train)
    snaps = tr.reshape(len(train) * 6, -1)
    basis, evals, orth = DF.pod_basis(snaps, min(a.pod_rank, snaps.shape[0]), 96)
    del snaps
    held = centred_frames(probe[:a.pod_held])
    ranks = [r for r in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if r <= basis.shape[1]]
    def floor(frames):
        flat = frames.reshape(-1, frames.shape[2] * n ** 3)
        coeff = np.asarray(jnp.asarray(flat) @ jnp.asarray(basis))
        tot = np.sum(flat * flat, axis=1)
        u0 = np.sqrt(tot.reshape(len(frames), 6)[:, :1])
        res = {}
        for r in ranks:
            e = np.sqrt(np.maximum(tot - np.sum(coeff[:, :r] ** 2, axis=1), 0)).reshape(len(frames), 6) / u0
            res[str(r)] = dict(evolved_worst=float(e[:, 1:].max()), evolved_median=float(np.median(e[:, 1:].max(axis=1))),
                               initial_worst=float(e[:, 0].max()), all_worst=float(e.max()),
                               all_median=float(np.median(e.max(axis=1))))
        return res
    report["pod_floor"] = dict(n=n, dt=a.pod_dt, train_cases=len(train), held_cases=int(a.pod_held),
                               orth=orth, singular_values=[float(np.sqrt(max(x, 0))) for x in evals],
                               held_out=floor(held), train_first16=floor(tr[:16]))
    for r in ranks:
        log(f"pod floor rank {r}: held-out {report['pod_floor']['held_out'][str(r)]}")
    (out / "characterise.json").write_text(json.dumps(report, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
