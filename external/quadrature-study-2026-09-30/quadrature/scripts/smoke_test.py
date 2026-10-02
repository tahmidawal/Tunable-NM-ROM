"""Quick end-to-end smoke test of the package (random model, tiny mesh). usage: python scripts/smoke_test.py burgers"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.pdes import PDES
from nmrom.fom import solve_case
from nmrom.grid import interior_coords
from nmrom.bank import init_bank, orthonormalise
from nmrom.head import init_head, head
from nmrom import quadrature as Q
from nmrom.rom import MeshModel, ReducedProblem, make_dense_evaluator, make_offmesh_evaluator, rollout_errors, scalar_params
pde = PDES[sys.argv[1] if len(sys.argv) > 1 else "burgers"]; N = 64; R, k = 32, 8
print("devices:", jax.devices())
X = interior_coords(N)
bank = orthonormalise(init_bank(jax.random.PRNGKey(0), R), X)
hp = init_head(jax.random.PRNGKey(1), k, R)
Z = np.random.default_rng(0).normal(size=(40, k)); codes = np.asarray(jax.vmap(lambda z: head(hp, z))(jnp.asarray(Z)))
ckpt = dict(R=R, k=k, bank={a: np.asarray(b) for a, b in bank.items()}, head={a: np.asarray(b) for a, b in hp.items()},
            Z=Z, codes=codes, C=np.linalg.qr(np.random.default_rng(1).normal(size=(R, R)))[0], t=np.zeros(40), case=np.zeros(40, int))
p = pde.sample_params(np.random.default_rng(5)); ckpt["train_params"] = [p]
t = time.time(); fom = solve_case(pde, N, p); print(f"FOM {N}^2: {time.time()-t:.1f}s")
case = dict(p=p, same_grid={N: dict(u_out=fom["u_out"])}, refined=dict(u_out=fom["u_out"], N=N))
mm = MeshModel(ckpt, pde, N, 4 * k)
for ev in [make_dense_evaluator(mm), make_offmesh_evaluator(mm, *Q.gauss_tensor(24), "gl24"), make_offmesh_evaluator(mm, *Q.fibonacci_lattice(14), "fib")]:
    rp = ReducedProblem(mm, 0, ev); t = time.time(); roll = rp.rollout(case); err = rollout_errors(mm, roll, case)
    print(f"  {ev['label']:6s} m={ev['m']:5d}: {time.time()-t:.1f}s  its/step {np.mean(roll['its']):.1f} exits {np.bincount(roll['exits'], minlength=4)} err {err['err_same_grid'].max():.3f}")
print("smoke test ok")
