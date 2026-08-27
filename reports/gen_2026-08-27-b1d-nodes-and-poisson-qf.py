"""Tables for reports/2026-08-27-b1d-node-screening-and-poisson-qf.md.

Usage:  python gen_2026-08-27-b1d-nodes-and-poisson-qf.py

Reads the run JSONs pulled from the b1dqf cluster namespace into the
2026-08-27-b1d-poissonqf worktree and prints the report's tables.  Every
number comes from the JSONs; nothing is typed by hand.
"""
import glob
import json
import os

RUNS = os.path.join(
    "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees",
    "2026-08-27-b1d-poissonqf", "experiments", "separable-decoder",
    "runs", "b1dqf")

ARMS = ["oracle", "base_gen", "base_tight", "nodes_tight",
        "base_half", "nodes_half"]
BASE_OF = {"nodes_tight": "base_tight", "nodes_half": "base_half"}


def e(v):
    return "—" if v is None else f"{v:.3e}"


def c(v):
    return "—" if v is None else f"{v:.4f}"


def load_b1d():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "b1d_n*", "out",
                                           "sep_b1d_*.json"))):
        d = json.load(open(p))
        assert d.get("complete"), f"incomplete run: {p}"
        out[d["config"]["N"]] = d
    return out


def load_qf():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "qf_n*", "out",
                                           "sep_poisson_qf_*.json"))):
        d = json.load(open(p))
        assert d.get("complete"), f"incomplete run: {p}"
        out[d["config"]["N"]] = d
    return out


def t_b1(runs):
    print("**T-B1 — 1D Burgers: all arms, held-out rungs and test rollouts "
          "(8 fresh trajectories, one seed).**\n")
    print("| N | arm | m | EQ rel fit | held (b) | held (c1) | (c1) cos | "
          "rollout err | vs base | vs oracle | ms/traj |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for N in sorted(runs):
        d = runs[N]
        v = d["variants"]
        for arm in ARMS:
            if arm not in v:
                continue
            r = v[arm]
            ho = r["heldout"]
            base = BASE_OF.get(arm)
            vb = ("—" if base is None else
                  f"{100*(r['rollout_err_mean']-v[base]['rollout_err_mean'])/v[base]['rollout_err_mean']:+.1f}%")
            vo = ("—" if arm == "oracle" else
                  f"{100*(r['rollout_err_mean']-v['oracle']['rollout_err_mean'])/v['oracle']['rollout_err_mean']:+.1f}%")
            print(f"| {N} | {arm} | {r['m']} | {e(r.get('eq_rel_fit'))} | "
                  f"{e(ho['b'])} | {e(ho['c1'])} | {c(ho['c1_cos'])} | "
                  f"{e(r['rollout_err_mean'])} | {vb} | {vo} | "
                  f"{r['rollout_ms_median']:.0f} |")


def t_b2(runs):
    print("\n**T-B2 — 1D Burgers: where the error budget lives per N "
          "(decoder floor vs quadrature).**\n")
    print("| N | held recon | oracle rollout | base m=M rollout | "
          "quadrature share at m=M | learned-nodes recovery |")
    print("|---|---|---|---|---|---|")
    for N in sorted(runs):
        v = runs[N]["variants"]
        orc = v["oracle"]["rollout_err_mean"]
        bt = v["base_tight"]["rollout_err_mean"]
        nt = v["nodes_tight"]["rollout_err_mean"]
        gap = bt - orc
        recov = "—" if gap <= 0 else f"{100*(bt-nt)/gap:.0f}%"
        print(f"| {N} | {e(runs[N]['held_recon_rel'])} | {e(orc)} | {e(bt)} "
              f"| {100*gap/bt:+.1f}% | {recov} |")


def t_p1(runs):
    print("\n**T-P1 — Poisson 2D: the three residual paths through the same "
          "solver (held-out cohort, tau=1e-3; per-source medians of "
          "persisted reps).**\n")
    print("| N | path | m | solve ms | err rel-L2 | jac evals | censored |")
    print("|---|---|---|---|---|---|---|")
    for N in sorted(runs):
        for row in runs[N]["rows"]:
            if row["cohort"].startswith("held") and row["tau"] == 1e-3:
                print(f"| {N} | {row['method']} | {row['m']} | "
                      f"{row['time_ms']:.2f} | {e(row['err_rel_l2'])} | "
                      f"{row['jac_evals']:.1f} | "
                      f"{row['censored_frac']*100:.0f}% |")


def t_p2(runs):
    print("\n**T-P2 — Poisson 2D: residual/gradient fidelity vs the "
          "full-grid reference, and one-time setup cost.**\n")
    print("| N | path | b (resid) at solutions | (c1) grad at solutions | "
          "(c1) cos min | setup s |")
    print("|---|---|---|---|---|---|")
    for N in sorted(runs):
        d = runs[N]
        su = d["setup"]
        for p in ("eq", "qf"):
            s = d["rungs"][f"summary_{p}"]
            setup = (su["eq_nnls_fit_s"] + su["eq_bank_and_table_s"]
                     if p == "eq" else su["qf_b_matmul_s"])
            print(f"| {N} | {p} | {e(s['b_resid_sol_mean'])} | "
                  f"{e(s['c1_grad_sol_mean'])} | {s['c1_cos_sol_min']:.4f} | "
                  f"{setup:.1f} |")


def t_p3(runs):
    print("\n**T-P3 — Poisson 2D: gate Q (quadrature-free == full grid, "
          "machine precision).**\n")
    print("| N | random states: resid rel max | grad rel max | "
          "at solutions: resid rel max | fails |")
    print("|---|---|---|---|---|")
    for N in sorted(runs):
        g = runs[N]["gates"]
        print(f"| {N} | {e(g['Q_random']['resid_rel_max'])} | "
              f"{e(g['Q_random']['grad_rel_max'])} | "
              f"{e(g['Q_solutions']['resid_rel_max'])} | "
              f"{g['Q_solutions']['n_fail']} |")


def load_scale():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "b1ds_n*", "out",
                                           "sep_b1d_scale_*.json"))):
        d = json.load(open(p))
        assert d.get("complete"), f"incomplete run: {p}"
        out[d["config"]["N"]] = d
    return out


def t_s1(runs):
    print("\n**T-S1 — 1D Burgers scaling ladder: accuracy per arm "
          "(rollout err, 8 fresh trajectories, one seed).**\n")
    print("| N | held recon | oracle | NNLS m=32 | learned m=32 (vs base) | "
          "NNLS m=16 | learned m=16 (vs base) |")
    print("|---|---|---|---|---|---|---|")
    for N in sorted(runs):
        v = runs[N]["variants"]

        def vb(a, b):
            return (f"{e(v[a]['rollout_err_mean'])} "
                    f"({100*(v[a]['rollout_err_mean']-v[b]['rollout_err_mean'])/v[b]['rollout_err_mean']:+.1f}%)")
        print(f"| {N} | {e(runs[N]['held_recon_rel'])} | "
              f"{e(v['oracle']['rollout_err_mean'])} | "
              f"{e(v['base_tight']['rollout_err_mean'])} | "
              f"{vb('nodes_tight', 'base_tight')} | "
              f"{e(v['base_half']['rollout_err_mean'])} | "
              f"{vb('nodes_half', 'base_half')} |")


def t_s2(runs):
    print("\n**T-S2 — 1D Burgers: ROM online cost per trajectory "
          "(learned m=32 arm; on-device instrument, medians of persisted "
          "reps).  The latent solve is the part that must be flat in N; "
          "IC projection and full decode are O(N) one-time ends.**\n")
    print("| N | IC fit ms | latent solve ms | full decode ms | e2e ms | "
          "oracle-arm solve ms (O(N) residual, for contrast) |")
    print("|---|---|---|---|---|---|")
    for N in sorted(runs):
        v = runs[N]["variants"]
        nt = v["nodes_tight"]
        print(f"| {N} | {nt['ic_ms_median']:.2f} | "
              f"{nt['roll_ms_median']:.2f} | {nt['dec_ms_median']:.2f} | "
              f"{nt['e2e_ms_median']:.2f} | "
              f"{v['oracle']['roll_ms_median']:.2f} |")


def t_s3(runs):
    print("\n**T-S3 — 1D Burgers: ROM vs the tridiagonal tolerance-Newton "
          "FOM, same job, same GPU.  NOT iso-accuracy: the FOM is far more "
          "accurate than the ROM everywhere (its error is shown).**\n")
    print("| N | ROM e2e ms (learned m=32) | ROM err | FOM ms (ntol 1e-3) | "
          "FOM err | FOM ms (ntol 1e-8) | ROM/FOM(1e-3) |")
    print("|---|---|---|---|---|---|---|")
    for N in sorted(runs):
        v = runs[N]["variants"]["nodes_tight"]
        fom = {f["ntol"]: f for f in runs[N]["fom"]}
        f3, f8 = fom[1e-3], fom[1e-8]
        print(f"| {N} | {v['e2e_ms_median']:.2f} | "
              f"{e(v['rollout_err_mean'])} | {f3['ms_median']:.2f} | "
              f"{e(f3['err_mean'])} | {f8['ms_median']:.2f} | "
              f"{v['e2e_ms_median']/f3['ms_median']:.1f}x |")


def main():
    b1d = load_b1d()
    qf = load_qf()
    scale = load_scale()
    if b1d:
        t_b1(b1d)
        t_b2(b1d)
    if qf:
        t_p1(qf)
        t_p2(qf)
        t_p3(qf)
    if scale:
        t_s1(scale)
        t_s2(scale)
        t_s3(scale)


if __name__ == "__main__":
    main()
