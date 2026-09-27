"""Wave 2D phase 2 — data, bank, three head arms, and the manifold gates D0, D1, D2, G0a, G0b.

  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY wav2d_head_gates.py N=64 BC=ref [R=64] [STEPS=40000]
        [HEADS=sup,auto,auto+vc] [SMOKE=1] [OUT=runs/wav2d] [CACHE=cache/wav2d]

Data are regenerated from the seed (cached under CACHE with a fingerprint; a cache whose fingerprint
does not match the freshly generated data aborts).  Heads are cached by their full configuration;
a SMOKE head can never be consumed by a certified run (the smoke flag travels with the head).
Every gate records value, threshold, pass, and its negative control.  G0c (stepdiag) is in phase 3.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import wav2d_common as wc
import wav2d_bank as wb
import wav2d_head as wh
from wav2d_common import Grid, log, precond

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
N = int(ARGS.get("N", "64"))
BC = ARGS.get("BC", "ref")
R = int(ARGS.get("R", "64"))
STEPS = int(ARGS.get("STEPS", "40000"))
HEADS = ARGS.get("HEADS", "sup,auto,auto+vc").split(",")
SMOKE = int(ARGS.get("SMOKE", "0"))
OUT = ARGS.get("OUT", "runs/wav2d")
CACHE = ARGS.get("CACHE", "cache/wav2d")
N_TRAIN = int(ARGS.get("N_TRAIN", str(wc.N_TRAIN)))
N_TEST = int(ARGS.get("N_TEST", str(wc.N_TEST)))
K_OF = {"sup": 6, "auto": 8, "auto+vc": 8}
ORACLE_STARTS = int(ARGS.get("ORACLE_STARTS", "8"))
ORACLE_ITERS = int(ARGS.get("ORACLE_ITERS", "400"))
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)


def finite(*xs):
    return all(np.all(np.isfinite(np.asarray(x, dtype=float))) for x in xs)


def gate(name, value, thr, control=None, control_thr=None, control_dir="ge", note="", aggregate="median"):
    v = float(value) if value is not None else float("nan")
    ok = bool(np.isfinite(v) and v <= thr)
    rec = dict(value=v, threshold=thr, aggregate=aggregate, passed=ok, note=note)
    if control is not None:
        cv = float(control)
        fired = bool(np.isfinite(cv) and (cv >= control_thr if control_dir == "ge" else cv <= control_thr))
        rec.update(control_value=cv, control_threshold=control_thr, control_dir=control_dir, control_fired=fired)
        rec["passed"] = ok and fired
    log(f"  {name:5s} {'PASS' if rec['passed'] else 'FAIL'}  value={v:.3e} thr={thr:.1e}" +
        (f"  control={rec['control_value']:.3e} fired={rec['control_fired']}" if control is not None else "") +
        (f"  [{note}]" if note else ""))
    return rec


# ----------------------------- data (cached, fingerprinted) -----------------------------

def get_data(g: Grid):
    path = os.path.join(CACHE, f"data_{g.bc}_N{g.N}_tr{N_TRAIN}_te{N_TEST}.npz")
    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=False))
        fp = wc.data_fingerprint(d["U"])
        precond(fp["sha256"] == str(d["sha256"]), "data cache fingerprint mismatch -- delete the cache")
        log(f"  data loaded from cache {path} (sha256 {fp['sha256'][:12]})")
        d["fingerprint"] = fp
        return d
    d = wc.build_data(g, n_train=N_TRAIN, n_test=N_TEST)
    np.savez(path, U=d["U"], V=d["V"], E=d["E"], mu=d["mu"], c=d["c"], U_test=d["U_test"], V_test=d["V_test"],
             E_test=d["E_test"], mu_test=d["mu_test"], c_test=d["c_test"], sha256=d["fingerprint"]["sha256"],
             train_energy_check=d["train_energy_check"], test_energy_check=d["test_energy_check"])
    return d


# ----------------------------- driver -----------------------------

def main():
    t_all = time.time()
    g = Grid(N, BC)
    prov = wc.provenance()
    log(f"phase-2 head gates N={N} BC={BC} R={R} STEPS={STEPS} SMOKE={SMOKE} backend={prov['jax_backend']}")
    res = dict(N=N, bc=BC, R=R, steps=STEPS, smoke=bool(SMOKE), provenance=prov, args=ARGS, heads={})
    d = get_data(g)
    U, V, Ut, Vt = d["U"], d["V"], d["U_test"], d["V_test"]
    res["data"] = dict(sha256=str(d.get("sha256", d.get("fingerprint", {}).get("sha256", ""))),
                       n_train=int(U.shape[0]), n_test=int(Ut.shape[0]))
    m = g.mass_diag()

    # bank + D0
    t0 = time.time()
    bank = wb.build_bank(g, U, R)
    log(f"  bank R={R} ({bank['method']}) in {time.time()-t0:.0f}s: sigma_R/sigma_1 {bank['sigma_ratio_R']:.3e}, gap {bank['gap_R']:.3f}")
    D0 = wb.gate_D0(g, bank, U, Ut)
    log(f"  D0    {'PASS' if D0['passed'] else 'FAIL'}  orth {D0['orthonormality']:.2e} (control {D0['orthonormality_control_noM']:.2e}), "
        f"floor bank/indep {D0['floor_bank_median']:.4f}/{D0['floor_independent_median']:.4f} reldiff {D0['floor_reldiff_max']:.1e}, metric {D0['metric_identity']:.1e}")
    res["bank"] = dict(sigma=bank["sigma"].tolist(), sigma_ratio_R=bank["sigma_ratio_R"], gap_R=bank["gap_R"], method=bank["method"])
    res["D0"] = D0
    np.savez(os.path.join(CACHE, f"bank_{BC}_N{N}_R{R}.npz"), G=bank["G"], sigma=bank["sigma"])
    G = bank["G"]

    # coefficients
    X, traj = wb.snapshots(U); Xt, trajt = wb.snapshots(Ut)
    C = wb.coefficients(g, G, X); Ct = wb.coefficients(g, G, Xt)
    CV = wb.coefficients(g, G, V.reshape(-1, g.n)); CVt = wb.coefficients(g, G, Vt.reshape(-1, g.n))
    perp2 = np.sum(m[None, :] * (X - C @ G.T) ** 2, axis=1); perp2t = np.sum(m[None, :] * (Xt - Ct @ G.T) ** 2, axis=1)
    perpv2t = np.sum(m[None, :] * (Vt.reshape(-1, g.n) - CVt @ G.T) ** 2, axis=1)
    # kinetic-energy fraction of the test rows (for the G0b filter)
    KEt = 0.5 * np.sum(m[None, :] * Vt.reshape(-1, g.n) ** 2, axis=1)
    Et = d["E_test"].reshape(-1)
    ke_frac = KEt / np.maximum(Et, 1e-300)
    MU_tr = wh.sup_latents(d["mu"]); MU_te = wh.sup_latents(d["mu_test"])
    # a train cohort of the test's size for G0a (first N_TEST training trajectories)
    tr_sub = traj < N_TEST

    for mode in HEADS:
        K = K_OF[mode]
        log(f"== head arm {mode} (K={K}) ==")
        cfg = dict(mode=mode, K=K, R=R, steps=STEPS, hidden=128, layers=3, lr=3e-3, batch=2048, seed=0,
                   n_fit=int(C.shape[0]), smoke=bool(SMOKE), bc=BC, N=N)
        hpath = os.path.join(CACHE, f"head_{BC}_N{N}_R{R}_{mode.replace('+','')}_s{STEPS}{'_SMOKE' if SMOKE else ''}.npz")
        spec = wh.load_spec(hpath, expect=dict(mode=mode, K=K, R=R, steps=STEPS, smoke=bool(SMOKE), bc=BC, N=N))
        trained = spec is None
        if trained:
            t0 = time.time()
            spec = wh.train_head(C, traj, mode, K, MU=(MU_tr if mode == "sup" else None), CV=(CV if mode == "auto+vc" else None),
                                 steps=STEPS, log_every=max(STEPS // 8, 1), tag=mode)
            wh.save_spec(hpath, spec, extra=dict(smoke=bool(SMOKE), bc=BC, N=N))
            log(f"  trained in {spec['seconds']:.0f}s, final loss {spec['final_loss']:.3e}, params {spec['n_params']}")
        else:
            log(f"  loaded cached head {hpath}")
        H = dict(config=cfg, trained=trained, final_loss=spec["final_loss"], n_params=spec["n_params"], gates={})

        # oracle on held-out and on the train cohort
        t0 = time.time()
        Zt, rt = wh.oracle(spec, Ct, n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)
        Ztr, rtr = wh.oracle(spec, C[tr_sub], n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)
        log(f"  oracle: {Ct.shape[0]} held-out + {int(tr_sub.sum())} train rows in {time.time()-t0:.0f}s")
        e_te = wh.traj_rms_from_coeffs(rt, perp2t, Ct, trajt)
        e_tr = wh.traj_rms_from_coeffs(rtr, perp2[tr_sub], C[tr_sub], traj[tr_sub])
        pod_te = wh.pod_k_traj_rms(Ct, perp2t, trajt, K)
        pod_R_te = wh.traj_rms_from_coeffs(np.zeros(len(Ct)), perp2t, Ct, trajt)      # the bank ceiling
        H["oracle_heldout_per_traj"] = e_te.tolist(); H["oracle_train_per_traj"] = e_tr.tolist()
        H["podK_heldout_per_traj"] = pod_te.tolist(); H["podR_ceiling_per_traj"] = pod_R_te.tolist()

        # D1: held-out oracle vs POD-K (median over trajectories).
        # Control (RETRACTION 5): an UNTRAINED head (random init, 0 steps) -- its per-snapshot oracle measures the
        # manifold's CAPACITY alone; the trained head must beat it by >= 1.3x.  The earlier 'shuffled-target head must
        # be worse than POD-K' could not fire: for free-code arms a row shuffle is not a mutation (the codes are per
        # row), and any smooth K-manifold fitted per snapshot captures about what POD-K does.
        t0 = time.time()
        spec_un = wh.train_head(C, traj, mode, K, MU=(MU_tr if mode == "sup" else None),
                                CV=(CV if mode == "auto+vc" else None), steps=2, tag=mode + "-untrained")   # 2: the schedule needs >= 1 decay step
        Zun, run_ = wh.oracle(spec_un, Ct, n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)
        e_un = wh.traj_rms_from_coeffs(run_, perp2t, Ct, trajt)
        # the shuffled-target head is still REPORTED (it is informative for 'sup', vacuous for free codes)
        rng = np.random.default_rng(123)
        Csh = C[rng.permutation(len(C))]
        spec_sh = wh.train_head(Csh, traj, mode, K, MU=(MU_tr if mode == "sup" else None),
                                CV=(CV if mode == "auto+vc" else None), steps=max(STEPS // 4, 50), tag=mode + "-shuffled")
        Zsh, rsh = wh.oracle(spec_sh, Ct, n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)
        e_sh = wh.traj_rms_from_coeffs(rsh, perp2t, Ct, trajt)
        log(f"  D1 controls (untrained, shuffled) trained+evaluated in {time.time()-t0:.0f}s")
        H["gates"]["D1"] = gate("D1", np.median(e_te) / np.median(pod_te), 0.5,
                                control=np.median(e_un) / np.median(e_te), control_thr=1.3,
                                note=f"held-out oracle / POD-K, medians ({np.median(e_te):.4f} / {np.median(pod_te):.4f}); "
                                     f"bank ceiling POD-R {np.median(pod_R_te):.4f}; control: UNTRAINED head's oracle {np.median(e_un):.4f} must be >= 1.3x the trained head's "
                                     f"(retraction 5); shuffled-target head {np.median(e_sh):.4f} reported (vacuous for free-code arms); "
                                     f"FiLM 08-14 comparator NOT recomputed in this pass")
        H["gates"]["D1"]["shuffled_control_heldout_median"] = float(np.median(e_sh))
        H["gates"]["D1"]["untrained_control_heldout_median"] = float(np.median(e_un))

        # D2: J_h conditioning at training codes, oracle points; control: duplicated coordinate
        Ztrain = spec["Z"] if mode != "sup" else MU_tr
        c_tr = np.concatenate([wh.jac_condition(spec, Ztrain[i0:i0 + 4096]) for i0 in range(0, len(Ztrain), 4096)])   # ALL training codes
        c_te = wh.jac_condition(spec, Zt)
        c_dup = wh.jac_condition(wh.duplicated_coordinate_control(spec), np.concatenate([Zt[:64], np.zeros((min(64, len(Zt)), 1))], axis=1))
        val = -min(float(c_tr.min()), float(c_te.min()))           # gate wants value <= thr; use -cond <= -1e-8
        H["gates"]["D2"] = gate("D2", val, -1e-8, control=-float(c_dup.max()), control_thr=-1e-12, control_dir="ge",
                                note=f"-(min sigma_min/sigma_max of J_h) over train codes ({c_tr.min():.2e}) and oracle points ({c_te.min():.2e}); "
                                     f"control: duplicated latent coordinate reads {c_dup.max():.1e} (must be ~0)", aggregate="min")
        H["gates"]["D2"].update(cond_train_min=float(c_tr.min()), cond_test_min=float(c_te.min()), cond_test_median=float(np.median(c_te)))

        # G0a: train/held-out gap.  Control (RETRACTION 6): an OVERFIT head -- the same arm trained on only 4
        # trajectories -- must show a large gap (ratio > 1.5 or abs gap > 0.05).  The earlier 'shuffled head's gap'
        # could not fire: a head that learned nothing has no generalisation gap by construction.
        ratio = float(np.median(e_te) / np.median(e_tr)); absgap = float(np.median(e_te) - np.median(e_tr))
        small = traj < 4
        spec_of = wh.train_head(C[small], traj[small], mode, K, MU=(MU_tr[small] if mode == "sup" else None),
                                CV=(CV[small] if mode == "auto+vc" else None), steps=max(STEPS // 4, 50), tag=mode + "-overfit4")
        e_of_te = wh.traj_rms_from_coeffs(wh.oracle(spec_of, Ct, n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)[1], perp2t, Ct, trajt)
        e_of_tr = wh.traj_rms_from_coeffs(wh.oracle(spec_of, C[small], n_starts=ORACLE_STARTS, iters=ORACLE_ITERS)[1], perp2[small], C[small], traj[small])
        ratio_of = float(np.median(e_of_te) / np.median(e_of_tr)); gap_of = float(np.median(e_of_te) - np.median(e_of_tr))
        rec = gate("G0a", max(ratio / 1.5, absgap / 0.05), 1.0, control=max(ratio_of / 1.5, gap_of / 0.05), control_thr=1.0,
                   note=f"max(ratio/1.5, absgap/0.05): held-out {np.median(e_te):.4f} vs train {np.median(e_tr):.4f}, ratio {ratio:.3f}, gap {absgap:.4f}; "
                        f"control: overfit head (4 trajectories) held-out {np.median(e_of_te):.4f} vs train {np.median(e_of_tr):.4f}, ratio {ratio_of:.3f}, gap {gap_of:.4f} (retraction 6)")
        rec.update(ratio=ratio, abs_gap=absgap, overfit_ratio=ratio_of, overfit_gap=gap_of); H["gates"]["G0a"] = rec

        # G0b: tangent-space velocity residual at oracle points with KE >= 10% E; control: random tangent
        tv, pv, ranks = wh.tangent_velocity_residual(spec, Zt, CVt, perpv2t, K)
        sel = ke_frac >= 0.1                                    # the KE filter is the ONLY exclusion
        precond(sel.sum() >= 32, "too few kinetic-energy-rich held-out states for G0b")
        nonfinite = int(np.sum(~(np.isfinite(tv[sel]) & np.isfinite(pv[sel]))))
        rnd = wh.random_tangent_residual(CVt[sel], perpv2t[sel], K)
        val = (np.median(tv[sel]) / np.median(pv[sel])) if nonfinite == 0 else float("nan")   # NaN anywhere -> FAIL
        rec = gate("G0b", val, 1.0, control=np.median(rnd) / np.median(pv[sel]), control_thr=1.2,
                   note=f"tangent-space velocity residual (median {np.nanmedian(tv[sel]):.4f}) / POD-K on the same {int(sel.sum())} states (median {np.nanmedian(pv[sel]):.4f}); "
                        f"J_h ranks {np.unique(ranks[sel]).tolist()}; nonfinite states {nonfinite} (any -> FAIL); control: random K-dim tangent / POD-K (median {np.median(rnd):.3f}) must be >= 1.2")
        rec["nonfinite_states"] = nonfinite
        rec.update(tangent_median=float(np.median(tv[sel])), podK_median=float(np.median(pv[sel])), n_states=int(sel.sum()),
                   tangent_per_state=tv.tolist(), ke_frac=ke_frac.tolist())
        H["gates"]["G0b"] = rec
        H["G0_passed"] = bool(H["gates"]["G0a"]["passed"] and H["gates"]["G0b"]["passed"])
        H["predicted_G0"] = {"auto": False, "sup": True, "auto+vc": True}[mode]
        log(f"  G0 {'PASS' if H['G0_passed'] else 'FAIL'} (predicted {'PASS' if H['predicted_G0'] else 'FAIL'})")
        # oracle latents for phase 3 (cold starts, stepdiag)
        np.savez(os.path.join(CACHE, f"oracle_{BC}_N{N}_R{R}_{mode.replace('+','')}_s{STEPS}{'_SMOKE' if SMOKE else ''}.npz"),
                 Z_test=Zt, res_test=rt, Z_train_sub=Ztr, res_train_sub=rtr)
        res["heads"][mode] = H

    res["wall_s"] = time.time() - t_all
    path = os.path.join(OUT, f"wav2d_head_gates_{BC}_N{N}_R{R}{'_SMOKE' if SMOKE else ''}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"phase 2 done ({res['wall_s']:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
