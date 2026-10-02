"""Supplement: CBC-constructed rank-1 lattices (and the P_2 merit of every lattice vector used)
against the 3D continuum reference; appends rows to results/burgers3d_quad_study.json."""
import sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.dim3 import *
from nmrom.rom import scalar_params
pde = BURGERS3D
ckpt = pickle.load(open("data/burgers3d_model.pkl", "rb")); ckpt["train_params"] = pickle.load(open("data/burgers3d_train_params.pkl", "rb"))
k, R = ckpt["k"], ckpt["R"]; M_acc, M_fast = 4 * (k + 64), 4 * k
log = open("results/burgers3d_quad_lattice.log", "w")
def say(*a): print(*a, flush=True); print(*a, file=log, flush=True)
rng = np.random.default_rng(0); sel = rng.choice(len(ckpt["codes"]), 96, replace=False)
states = [ckpt["codes"][i] for i in sel]; state_p = [scalar_params(ckpt["train_params"][int(ckpt["case"][i])]) for i in sel]
mm0 = MeshModel3(ckpt, pde, 64, M_acc)
ref = make_offmesh_evaluator3(mm0, *gauss_tensor3(64), "ref64")
def rho_over_states(ev):
    nl, nlr = jax.jit(ev["nl"]), jax.jit(ref["nl"]); out = []
    for c, ps in zip(states, state_p):
        a, b = np.asarray(nl(jnp.asarray(c), ps)), np.asarray(nlr(jnp.asarray(c), ps))
        out.append((np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300), np.linalg.norm(a[:M_fast] - b[:M_fast]) / max(np.linalg.norm(b[:M_fast]), 1e-300)))
    return np.array(out)
rows = []
say("== lattice generating vectors and P_2 merits")
for n in [1024, 2048, 4096, 8192, 16384, 32768]:
    t0 = time.time(); z, P = cbc_lattice_vector3(n); say(f"  n={n}: CBC z={z.tolist()} P2={P:.3e} ({time.time()-t0:.0f}s);  'kuo' z mod n={(KUO_Z % n).tolist()} P2={p2_merit3(KUO_Z, n):.3e}")
    for form in ["point", "flux"]:
        for name, (X, w) in [("cbc_lattice", rank1_lattice3(n, z)), ("cbc_lattice_tent", rank1_lattice3(n, z, tent=True))]:
            if form == "flux" and name != "cbc_lattice": continue
            r = rho_over_states(make_offmesh_evaluator3(mm0, X, w, name, form=form))
            rows.append(dict(pde="burgers3d", target="continuum", N=0, rule=name, param=n, m=n, form=form, rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()),
                             rho_acc_p90=float(np.quantile(r[:, 0], .9)), rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()), wmin=float(w.min()), wsum=float(w.sum()), z=z.tolist(), P2=P))
            say(f"  [{form}] {name:17s} m={n:6d}  rho(M={M_acc}) med {rows[-1]['rho_acc_med']:.2e} max {rows[-1]['rho_acc_max']:.2e} | rho(M={M_fast}) med {rows[-1]['rho_fast_med']:.2e} max {rows[-1]['rho_fast_max']:.2e}")
for n in [4093, 16381]:
    z, P = korobov_search3(n); say(f"  korobov n={n}: z={z.tolist()} P2={P:.3e}")
json.dump(rows, open("results/burgers3d_quad_study_lattice.json", "w"), indent=1)
say("done")
