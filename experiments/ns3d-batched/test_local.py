"""Tiny local check (synthetic operators, n = 16/24): the shared-operator ROM and the
real-FFT solver reproduce the unchanged codes, batched and unbatched."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batched as BT
from batched import jax, jnp, F, HR, H, P

rng = np.random.default_rng(0)
n, R, k, M, steps, iters = 16, 12, 4, 40, 10, 3
# band-limited bank (|k| < n/2, as every real bank: built from truncated fields)
_raw = rng.standard_normal((R, 3, n, n, n))
_bl = np.real(np.fft.ifftn(np.fft.fftn(_raw, axes=(-3, -2, -1)) * np.asarray(H.keep_mask(n)), axes=(-3, -2, -1)))
G = np.linalg.qr(_bl.reshape(R, -1).T)[0]
A = rng.standard_normal((M, R)); T = 1e-2 * rng.standard_normal((M, R, R))
lam = np.abs(rng.standard_normal(M)) * 10; Dd = 1e-1 * rng.standard_normal((3, M, R))
p = dict(W1=0.3 * rng.standard_normal((k, 32)), b1=np.zeros(32), W2=0.1 * rng.standard_normal((32, 32)),
         b2=np.zeros(32), W3=0.01 * rng.standard_normal((32, R)), b3=0.1 * rng.standard_normal(R),
         Ws=rng.standard_normal((k, R)))
p = {kk: jnp.asarray(v) for kk, v in p.items()}
Z = jnp.asarray(rng.standard_normal((20, k))); Hc = HR.head_apply(p, Z); Hn = jnp.sum(Hc * Hc, 1)
C = jnp.zeros((R, 0)); dt = 0.2 / steps
run = HR.make_head_run(dt, steps, steps // 5, n, R, k, 0, iters=iters, damping=1e-6, ic_iters=12)
e2e_b, enc_b, lat_b, dec_b = BT.make_rom_parts(dt, steps, steps // 5, n, R, k, iters, 1e-6, 12)
args = [jnp.asarray(x) for x in (G, A, T, lam, Dd)]
u0s = jnp.asarray(np.stack([(G @ (rng.standard_normal(R))).reshape(3, n, n, n) for _ in range(3)]))
nus = jnp.asarray([1e-3, 2e-3, 5e-4])
ref = jnp.stack([run(u0s[i], nus[i], *args, p, C, Hc, Hn, Z) for i in range(3)])
got = jax.jit(e2e_b)(u0s, nus, args[0], *args[1:], p, Hc, Hn, Z)
vm = jax.jit(jax.vmap(run, in_axes=(0, 0) + (None,) * 10))(u0s, nus, *args, p, C, Hc, Hn, Z)
w0, c0 = jax.jit(enc_b)(u0s, args[0], p, Hc, Hn, Z)
W, cs = jax.jit(lat_b)(w0, c0, nus, *args[1:], p)
parts = jax.jit(dec_b)(w0, c0, W, cs, args[0], p)
print("finite", bool(jnp.all(jnp.isfinite(ref))), "scale", float(jnp.max(jnp.abs(ref))))
print("rom shared vs ref", BT.relmax(got, ref), " vmap vs ref", BT.relmax(vm, ref), " parts vs e2e", BT.relmax(parts, got))

fam = H.FAMILIES["B"]; par = H.parameters(5, 3, fam)
for m in (24, 32):
    u = jnp.asarray(np.stack([H.initial(m, q, fam) for q in par]))
    for scheme, s in (("cnab2", 20), ("rk4", 10)):
        f0 = F.make_solver(0.2 / s, s, s // 5, scheme=scheme); f1 = BT.make_solver_r(0.2 / s, s, s // 5, scheme)
        r0 = jnp.stack([f0(u[i], float(par[i, -1]), F.geometry(m)) for i in range(3)])
        r1 = jax.jit(jax.vmap(f1, in_axes=(0, 0, None)))(u, jnp.asarray(par[:, -1]), BT.geometry_r(m))
        r2 = jax.jit(jax.vmap(f0, in_axes=(0, 0, None)))(u, jnp.asarray(par[:, -1]), F.geometry(m))
        print(m, scheme, "rfft vs ref", BT.relmax(r1, r0), " vmap vs ref", BT.relmax(r2, r0))

# rfft shift vs complex shift
f = jnp.asarray(H.truncate_field(jnp.asarray(np.stack([H.initial(48, q, fam) for q in par[:1]])[0]), 32))
off = jnp.asarray([3.7, -1.2, 10.4])
print("rfft shift vs complex shift", BT.relmax(BT.shift_field_r(f, off), BT.SR.shift_field(f, off)))
e2r = BT.make_rom_parts(dt, steps, steps // 5, n, R, k, iters, 1e-6, 12, rfft_shift=True)[0]
gotr = jax.jit(e2r)(u0s, nus, args[0], *args[1:], p, Hc, Hn, Z)
print("rom sharedR vs ref", BT.relmax(gotr, ref))
