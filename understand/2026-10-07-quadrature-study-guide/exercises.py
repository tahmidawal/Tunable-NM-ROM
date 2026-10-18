"""Exercises for the quadrature study guide.

Run:  /home/tahmid/Dev/.venv/bin/python exercises.py
It writes every figure (figs/) and every number table (tables/) that guide.tex uses,
so the guide never contains a hand-typed number. CPU only, about a minute.

Rules come from hari_quadrature.py, an unchanged copy of nmrom/quadrature.py from
Hari's package (quadrature-study-2026-09-30).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import qmc

import hari_quadrature as hq

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
TAB = os.path.join(HERE, "tables")
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
COL = {"gauss": "#1f77b4", "lattice": "#d4a017", "sobol": "#7f7f7f", "smolyak": "#d62728",
       "mc": "#9467bd", "uniform": "#2ca02c"}


def sci(v):
    """LaTeX scientific notation, generated (never typed)."""
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def write_table(name, header, rows):
    with open(os.path.join(TAB, name), "w") as f:
        f.write("\\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ step 1: 1D basics
def gauss_1d(f, p):
    x, w = hq.gauss_legendre_1d(p)
    return np.sum(w * f(x))


def midpoint_1d(f, n):
    x = (np.arange(n) + 0.5) / n
    return np.mean(f(x))


def mc_1d(f, n):
    return np.mean(f(rng.uniform(size=n)))


def step1():
    f_poly = lambda x: 7 * x ** 5 - 3 * x ** 2 + 1                     # degree 5
    exact_poly = 7 / 6 - 1 + 1
    rows = []
    for p in (1, 2, 3, 4):
        rows.append([f"{p}", sci(abs(gauss_1d(f_poly, p) - exact_poly) + 0.0)])
    write_table("t1_gauss_poly.tex", ["Gauss points $p$", "error on a degree-5 polynomial"], rows)

    f_smooth = lambda x: np.exp(x)                                      # does not vanish at the ends
    f_vanish = lambda x: np.sin(np.pi * x) ** 2 * np.exp(x)             # vanishes to 2nd order at ends
    ns = np.array([4, 8, 16, 32, 64, 128])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, f, title in [(axs[0], f_smooth, r"$f(x)=e^x$ (does not vanish at the ends)"),
                         (axs[1], f_vanish, r"$f(x)=\sin^2(\pi x)\,e^x$ (vanishes at the ends)")]:
        ex = integrate.quad(f, 0, 1, epsabs=1e-15, epsrel=1e-15)[0]
        eg = [abs(gauss_1d(f, n) - ex) for n in ns]
        eu = [abs(midpoint_1d(f, n) - ex) for n in ns]
        em = [np.mean([abs(mc_1d(f, n) - ex) for _ in range(50)]) for n in ns]
        ax.loglog(ns, np.maximum(eg, 1e-17), "o-", c=COL["gauss"], label="Gauss")
        ax.loglog(ns, np.maximum(eu, 1e-17), "s-", c=COL["uniform"], label="equal spacing (1D lattice)")
        ax.loglog(ns, em, "^-", c=COL["mc"], label="Monte Carlo (mean of 50)")
        ax.set_title(title, fontsize=9); ax.set_xlabel("number of points"); ax.set_ylim(1e-17, 1)
    axs[0].set_ylabel("absolute error"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s1_1d.pdf")); plt.close(fig)

    rows = []
    for f, name in [(f_smooth, "$e^x$"), (f_vanish, r"$\sin^2(\pi x)e^x$")]:
        ex = integrate.quad(f, 0, 1, epsabs=1e-15, epsrel=1e-15)[0]
        e1, e2 = abs(midpoint_1d(f, 32) - ex), abs(midpoint_1d(f, 64) - ex)
        rows.append([name, sci(e1), sci(e2), f"{e1 / max(e2, 1e-300):.1f}"])
    write_table("t1_uniform_ratio.tex", ["integrand", "error, 32 pts", "error, 64 pts", "ratio"], rows)


# ------------------------------------------------------------------ step 2: oscillation
def step2():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rows = []
    for a, c in [(1, "#1f77b4"), (10, "#ff7f0e"), (25, "#2ca02c")]:
        f = lambda x, a=a: np.sin(a * np.pi * x) * np.exp(-((x - 0.4) / 0.25) ** 2)
        ex = integrate.quad(f, 0, 1, limit=400, epsabs=1e-15, epsrel=1e-15)[0]
        ps = np.arange(2, 61, 2)
        er = np.array([abs(gauss_1d(f, p) - ex) for p in ps]) / max(abs(ex), 1e-3)
        ax.semilogy(ps, np.maximum(er, 1e-16), "o-", ms=3, c=c, label=f"$a={a}$")
        first = ps[np.argmax(er < 1e-6)] if np.any(er < 1e-6) else None
        rows.append([f"{a}", f"{first}" if first else "--"])
    ax.set_xlabel("Gauss points $p$"); ax.set_ylabel("relative error")
    ax.set_title(r"$\int_0^1 \sin(a\pi x)\,e^{-((x-0.4)/0.25)^2}dx$", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_oscillation.pdf")); plt.close(fig)
    write_table("t2_onset.tex", ["test frequency $a$", "first $p$ with error $<10^{-6}$"], rows)


# ------------------------------------------------------------------ 2D integrand like ours
def bump(x, y):
    """Illustrative smooth field vanishing on the walls (stands in for u = G(x) c)."""
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-((x - 0.45) ** 2 + (y - 0.4) ** 2) / 0.05)


def bump_grad(x, y, e=1e-6):
    return ((bump(x + e, y) - bump(x - e, y)) / (2 * e), (bump(x, y + e) - bump(x, y - e)) / (2 * e))


def integrand(X, a, b):
    x, y = X[:, 0], X[:, 1]
    u = bump(x, y)
    ux, uy = bump_grad(x, y)
    return np.sin(a * np.pi * x) * np.sin(b * np.pi * y) * u * (ux + uy)


def reference(a, b):
    X, W = hq.gauss_tensor(300)
    return np.sum(W * integrand(X, a, b))


def scale(a, b):
    """Size of the integrand, int |f|; errors are divided by this, not by |int f|, which can
    nearly cancel for high-frequency tests and make relative errors meaningless."""
    X, W = hq.gauss_tensor(300)
    return np.sum(W * np.abs(integrand(X, a, b)))


def apply(rule, a, b):
    X, W = rule
    return np.sum(W * integrand(X, a, b))


def step3_points():
    rules = [("Gauss $14\\times14$", hq.gauss_tensor(14), COL["gauss"]),
             ("Fibonacci lattice", hq.fibonacci_lattice(13), COL["lattice"]),
             ("Sobol (scrambled)", hq.sobol(256), COL["sobol"]),
             ("Smolyak level 6", hq.smolyak(6), COL["smolyak"]),
             ("Monte Carlo", hq.uniform_mc(256), COL["mc"])]
    fig, axs = plt.subplots(1, 5, figsize=(12, 2.9))
    for ax, (name, (X, W), c) in zip(axs, rules):
        s = 4 + 600 * np.abs(W) / np.abs(W).max() * (0.05 if "Smolyak" not in name else 0.02)
        neg = W < 0
        ax.scatter(X[~neg, 0], X[~neg, 1], s=s[~neg] if "Gauss" in name or "Smolyak" in name else 4, c=c)
        if neg.any():
            ax.scatter(X[neg, 0], X[neg, 1], s=s[neg], facecolors="none", edgecolors="k", lw=0.6)
        ax.set_title(f"{name}\n$m={len(X)}$", fontsize=9); ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s3_points.pdf")); plt.close(fig)


def ladders():
    return {
        "Gauss": (COL["gauss"], [hq.gauss_tensor(p) for p in (8, 12, 16, 24, 32, 48, 64)]),
        "Fibonacci lattice": (COL["lattice"], [hq.fibonacci_lattice(k) for k in (11, 12, 13, 14, 15, 16, 17, 18)]),
        "Sobol": (COL["sobol"], [hq.sobol(2 ** j) for j in range(7, 14)]),
        "Smolyak": (COL["smolyak"], [hq.smolyak(l) for l in range(3, 10)]),
        "Monte Carlo": (COL["mc"], [hq.uniform_mc(2 ** j) for j in range(7, 14)]),
    }


def step4_convergence():
    L = ladders()
    tests = [(3, 4, "low frequency test $\\psi_{3,4}$"), (12, 12, "mixed high frequency test $\\psi_{12,12}$")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    summary = {}
    for ax, (a, b, title) in zip(axs, tests):
        ref, sc = reference(a, b), scale(a, b)
        for name, (c, rules) in L.items():
            ms = [len(r[0]) for r in rules]
            er = [abs(apply(r, a, b) - ref) / sc for r in rules]
            ax.loglog(ms, np.maximum(er, 1e-16), "o-", ms=3, c=c, label=name)
            summary[(a, b, name)] = (ms, er)
        ax.axhline(0.116, ls="--", c="k", lw=0.8)
        ax.set_title(title, fontsize=9); ax.set_xlabel("number of points $m$")
    axs[0].set_ylabel(r"error $/\int|f|$"); axs[1].legend(fontsize=8, loc="lower left")
    axs[1].text(1.2e4, 0.16, "0.116", fontsize=7, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s4_convergence.pdf")); plt.close(fig)

    def at(ms, er, target):
        i = int(np.argmin(np.abs(np.log(np.array(ms) / target))))
        return ms[i], er[i]
    rows = []
    for name in L:
        r = [name]
        for a, b in [(3, 4), (12, 12)]:
            m, e = at(*summary[(a, b, name)], 4096)
            r.append(f"{m}"); r.append(sci(e))
        rows.append(r)
    write_table("t4_at4096.tex", ["rule", "$m$", r"error, $\psi_{3,4}$", "$m$", r"error, $\psi_{12,12}$"], rows)


def step5_shifts():
    a, b = 6, 5
    ref, sc = reference(a, b), scale(a, b)
    rows = []
    for k in (13, 15, 17):
        ers = [abs(apply(hq.fibonacci_lattice(k, seed=s), a, b) - ref) / sc for s in range(32)]
        rows.append([f"{hq._fib(k)}", sci(np.median(ers)), sci(np.max(ers))])
    write_table("t5_shifts.tex", ["lattice points $m$", "median over 32 shifts", "worst over 32 shifts"], rows)


def step6_tent():
    a, b = 6, 5
    ref, sc = reference(a, b), scale(a, b)
    rows = []
    for k in (13, 15, 17):
        e0 = abs(apply(hq.fibonacci_lattice(k), a, b) - ref) / sc
        e1 = abs(apply(hq.fibonacci_lattice(k, tent=True), a, b) - ref) / sc
        rows.append([f"{hq._fib(k)}", sci(e0), sci(e1)])
    write_table("t6_tent.tex", ["$m$", "plain lattice", "with tent transform"], rows)


# ------------------------------------------------------------------ step 7: mesh vs continuum
def step7_mesh_gap():
    a, b = 3, 4
    ref = reference(a, b)
    rows = []
    Ns, gaps = [], []
    for N in (32, 64, 128, 256, 512):
        h = 1.0 / N
        g = np.arange(1, N) * h
        Xg, Yg = np.meshgrid(g, g, indexing="ij")
        full = np.linspace(0, 1, N + 1)
        Xf, Yf = np.meshgrid(full, full, indexing="ij")
        U = bump(Xf, Yf)
        Ui = U[1:-1, 1:-1]
        bx = (U[1:-1, 1:-1] - U[:-2, 1:-1]) / h; fx = (U[2:, 1:-1] - U[1:-1, 1:-1]) / h
        by = (U[1:-1, 1:-1] - U[1:-1, :-2]) / h; fy = (U[1:-1, 2:] - U[1:-1, 1:-1]) / h
        Dx = np.where(Ui > 0, bx, fx); Dy = np.where(Ui > 0, by, fy)
        psi = np.sin(a * np.pi * Xg) * np.sin(b * np.pi * Yg)
        mesh_sum = h * h * np.sum(psi * Ui * (Dx + Dy))
        gap = abs(mesh_sum - ref) / abs(ref)
        Ns.append(N); gaps.append(gap)
        rows.append([f"{N}", sci(gap)] + ([f"{gaps[-2] / gap:.2f}"] if len(gaps) > 1 else ["--"]))
    write_table("t7_gap.tex", ["mesh $N$", "mesh sum vs true integral", "ratio to previous"], rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(Ns, gaps, "o-", c="#d62728", label="upwind mesh sum")
    ax.loglog(Ns, gaps[0] * Ns[0] / np.array(Ns), "--", c="k", lw=0.8, label=r"slope $-1$ ($O(h)$)")
    ax.set_xlabel("mesh points per axis $N$"); ax.set_ylabel("relative difference"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7_gap.pdf")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    step1(); step2(); step3_points(); step4_convergence(); step5_shifts(); step6_tent(); step7_mesh_gap()
    print("figures:", sorted(os.listdir(FIG)))
    print("tables:", sorted(os.listdir(TAB)))
