"""Cost model of one reduced NS query (ns2d_rom.make_query / make_lm / make_weak), fitted to
ns304's measured per-rung times and checked out-of-sample on the POD-LSPG subjects; then the
per-fix estimates. CPU only. Writes tables.md beside this file. Nothing is hand-typed."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = ('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/'
       'artifacts/ns304/result.json')
BP = ('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-panel/experiments/b-panel/'
      'artifacts/bpn301/result.json')
r = json.load(open(RES))
inv = r['invocations']
NSTEPS = 500
M, R, K = 2176, 512, 32
BYTES = 8
out = []
P = lambda s='': out.append(s)


def agg(name):
    t = [i for i in inv if i['subject'] == name and i['timed']]
    return dict(ms=float(np.median([i['seconds'] for i in t])) * 1e3,
                it=float(np.median([i['iterations_total'] for i in t])) / NSTEPS,
                we=max(i['worst_evolved'] for i in t if i['rep'] == 3))


subs = [s for s in r['subjects']]
neural = [s for s in subs if s['arm'] == 'neural']
pod = [s for s in subs if s['arm'] == 'pod']

# ---- pass counts per step, straight from ns2d_rom.py:
#   step(): r0 = fun(w) [1 residual], re = fun(we) [1 residual]          (lines 307-308)
#   step_lm -> evaluate(z0): fun + jacfwd(fun)                          (lines 224-227, 232)
#   body, per LM iteration: fun(zn) [1 residual] + cond(accept, evaluate(zn)) [fun + jacfwd]  (243-245)
n_res = lambda it: 3 + 2 * it        # residual passes / step  (the primal inside jacfwd assumed CSE'd)
n_jac = lambda it: 1 + it            # jacfwd passes / step

# ---- fit a (ms per streaming pass over the (M,R,R) tensor) and c (ms per tangent column of jacfwd)
# model: ms/step = n_res*a + n_jac*max(a, c*T)  -> for T >= 32 the compute term dominates a
# (verified after the fit: c*32 >= a), so ms/step = n_res*a + n_jac*c*T is linear in (a, c).
rows, y = [], []
for s in neural:
    g = agg(s['name']); T = s['unknowns']; it = g['it']
    rows.append([n_res(it), n_jac(it) * T]); y.append(g['ms'] / NSTEPS)
(a, c), *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(y), rcond=None)
gb = M * R * R * BYTES / 1e9
bw = gb / a * 1e3
tflops = 2 * M * R * R / (c * 1e-3) / 1e12
P('## A. Fitted constants (A100 80GB PCIe `pax105`, f64, from the six neural rungs)')
P()
P(f'- tensor (M,R,R) = ({M},{R},{R}) f64 = **{gb:.2f} GB**')
P(f'- `a` = one streaming pass over the tensor (`Q @ cm`, `ns2d_rom.py:214`): **{a:.2f} ms** '
  f'-> effective bandwidth {bw/1e3:.2f} TB/s (A100-80GB-PCIe spec 1.94 TB/s)')
P(f'- `c` = one tangent column of `jax.jacfwd(fun)` (`ns2d_rom.py:226`, a batched (R,R)@(R,T) matmul over M): '
  f'**{c:.4f} ms/column** -> {tflops:.1f} TFLOP/s f64 (A100 f64 tensor-core peak 19.5)')
P(f'- crossover: jacfwd is compute-bound once T > a/c = {a / c:.0f} columns; every rung (T >= 32) is on the compute side')
P()
P('## B. Check against the measured neural rungs (model uses the MEASURED iteration count of each rung)')
P()
P('| rung | T=K+q | LM it/step | residual passes/step | jacfwd passes/step | model ms/step | measured ms/step | model/measured | measured s/query |')
P('|---|---|---|---|---|---|---|---|---|')
for s in neural:
    g = agg(s['name']); T = s['unknowns']; it = g['it']
    pred = n_res(it) * a + n_jac(it) * max(a, c * T)
    meas = g['ms'] / NSTEPS
    P(f"| {s['name']} | {T} | {it:.2f} | {n_res(it):.1f} | {n_jac(it):.1f} | {pred:.1f} | {meas:.1f} | {pred / meas:.2f} | {g['ms'] / 1e3:.1f} |")
P()
P('Share of the per-step time in the jacfwd term (model): ' + ', '.join(
    f"q={s['q']}: {n_jac(agg(s['name'])['it']) * max(a, c * s['unknowns']) / (n_res(agg(s['name'])['it']) * a + n_jac(agg(s['name'])['it']) * max(a, c * s['unknowns'])) * 100:.0f} %"
    for s in neural))
P()
P('## C. Out-of-sample check: POD-LSPG subjects (tensor (M,k,k); jacfwd = batched (k,k)@(k,k) -> 2Mk^3 FLOP)')
P()
P('| subject | k | LM it/step | tensor GB | pass ms | jacfwd ms | model ms/step | measured ms/step | model/measured |')
P('|---|---|---|---|---|---|---|---|---|')
lat = None
for s in pod:
    g = agg(s['name']); k = s['unknowns']; it = g['it']
    gbk = M * k * k * BYTES / 1e9
    ak = gbk / bw * 1e3
    jk = max(ak, 2 * M * k * k * k / (tflops * 1e12) * 1e3)
    pred = n_res(it) * ak + n_jac(it) * jk
    meas = g['ms'] / NSTEPS
    if k == 32:
        lat = meas - pred
    P(f"| {s['name']} | {k} | {it:.2f} | {gbk:.3f} | {ak:.3f} | {jk:.2f} | {pred:.2f} | {meas:.2f} | {pred / meas:.2f} |")
P()
P(f'The k=32 row is pure launch/loop latency: measured minus model = **{lat:.2f} ms/step** '
  f'(~{lat * 1e3 / 35:.0f} us per kernel over the ~35 kernels a step launches). Used below as the floor `lat`.')
P()

# ---- FOM
fom = {n: agg(n) for n in ('fom_ntol0.001', 'fom_ntol1e-11')}
P('## D. The full-order solve in the same job')
P()
P(f"- FOM ntol 1e-3 (1 Newton/step, BiCGStab with exact FFT Helmholtz preconditioner): **{fom['fom_ntol0.001']['ms']:.0f} ms / query = "
  f"{fom['fom_ntol0.001']['ms'] / NSTEPS:.2f} ms/step**, worst evolved error {fom['fom_ntol0.001']['we']:.1e}")
P(f"- converged FOM ntol 1e-11 (2 Newton/step): {fom['fom_ntol1e-11']['ms']:.0f} ms / query")
P(f"- one FOM residual = 3 FFT2 of 256^2 + 9-point stencils: ~{3 * 5 * 65536 * 16 / 1e6:.0f} MFLOP and ~{6 * 65536 * 8 / 1e6:.0f} MB of traffic; "
  f"one ROM residual = {2 * M * R * R / 1e9:.2f} GFLOP and {gb:.2f} GB -> the ROM residual is ~{2 * M * R * R / (3 * 5 * 65536 * 16):.0f}x the FLOPs and ~{gb * 1e3 / (6 * 65536 * 8 / 1e6):.0f}x the bytes of the FOM residual.")
P()

# ---- per-fix estimates at q=0 and q=128 (A100), per query = 500 steps unless stated
P('## E. Per-fix estimates (A100 constants above; per query = 500 steps; `lat` floor added to every design)')
P()


def design(Mx, Rx, it, passes_res, passes_jac, jac_mode, dtype_bytes=8, m=None, stencil=18, T=32, steps=NSTEPS):
    """ms per query. jac_mode: 'jacfwd' (batched tensor matmul), 'analytic' (one pass + M*R*T), 'eq' (sampled)."""
    if m is None:
        gbx = Mx * Rx * Rx * dtype_bytes / 1e9
        ax = gbx / bw * 1e3
        if jac_mode == 'jacfwd':
            jx = max(ax, 2 * Mx * Rx * Rx * T / (tflops * 1e12) * 1e3 * (dtype_bytes / 8))
        else:
            jx = ax + 2 * Mx * Rx * T / (tflops * 1e12) * 1e3
    else:
        by = (2 * 9 * m * Rx + Mx * m) * dtype_bytes / 1e9
        ax = max(by / bw * 1e3, 0.02)
        jx = ax + 2 * (24 * m * Rx + m * Rx * T + Mx * m * T) / (tflops * 1e12) * 1e3   # analytic sampled Jacobian
    solve = 2 * T ** 3 / 3 / (tflops * 1e12) * 1e3 + 0.05
    per_step = passes_res * ax + passes_jac * jx + lat + (passes_jac * solve)
    return per_step * steps, per_step, ax, jx


it0 = agg('neural_q0')['it']
rows = [
    ('as measured (q=0, M=2176, jacfwd, 2.08 it/step)', design(2176, 512, it0, n_res(it0), n_jac(it0), 'jacfwd')),
    ('F1 analytic Jacobian only (reuse Q@cm; 1 pass gives r and J)', design(2176, 512, it0, n_res(it0), n_jac(it0), 'analytic')),
    ('F1 + cap 1 LM it/step, keep the 2-residual extrapolation test', design(2176, 512, 1, n_res(1), n_jac(1), 'analytic')),
    ('F1 + cap 1 it + always take the extrapolated start (no test)', design(2176, 512, 1, 1, 2, 'analytic')),
    ('F2 M=4(K+q)=128 test modes, everything else as written', design(128, 512, it0, n_res(it0), n_jac(it0), 'jacfwd')),
    ('F2 + F1 (M=128, analytic J)', design(128, 512, it0, n_res(it0), n_jac(it0), 'analytic')),
    ('F2 + F1 + cap 1 it, no extrapolation test', design(128, 512, 1, 1, 2, 'analytic')),
    ('F6 f32 tensor only (M=2176, jacfwd, 2.08 it) -- stopping rule must change', design(2176, 512, it0, n_res(it0), n_jac(it0), 'jacfwd', dtype_bytes=4)),
    ('F4 EQ m=2048 nodes at M=2176, analytic J, 2.08 it/step', design(2176, 512, it0, n_res(it0), n_jac(it0), 'eq', m=2048)),
    ('F4 EQ m=2048 at M=2176, analytic J, cap 1 it, no test', design(2176, 512, 1, 1, 2, 'eq', m=2048)),
    ('F4 EQ m=4096 at M=2176, analytic J, cap 1 it, no test', design(2176, 512, 1, 1, 2, 'eq', m=4096)),
    ('F4+F2 EQ m=512 at M=128, analytic J, cap 1 it, no test  (the Burgers q=0 recipe)', design(128, 512, 1, 1, 2, 'eq', m=512)),
    ('F4+F2 EQ m=1024 at M=128, analytic J, 2 it/step with test', design(128, 512, 2, n_res(2), n_jac(2), 'eq', m=1024)),
]
P('| design (q=0, K=32, R=512) | per residual/pass ms | per Jacobian ms | ms/step | s/query | vs FOM 0.42 s |')
P('|---|---|---|---|---|---|')
for name, (tot, per, ax, jx) in rows:
    P(f'| {name} | {ax:.3f} | {jx:.3f} | {per:.2f} | {tot / 1e3:.2f} | {tot / fom["fom_ntol0.001"]["ms"]:.2f}x |')
P()
P('Same designs at q=128 (T=160, M=4(K+q)=640 where F2 applies):')
P()
it1 = agg('neural_q128')['it']
rows = [
    ('as measured (M=2176, jacfwd, 2.04 it)', design(2176, 512, it1, n_res(it1), n_jac(it1), 'jacfwd', T=160)),
    ('F1 analytic J', design(2176, 512, it1, n_res(it1), n_jac(it1), 'analytic', T=160)),
    ('F2 M=640 + F1', design(640, 512, it1, n_res(it1), n_jac(it1), 'analytic', T=160)),
    ('F4 EQ m=2048 at M=2176, F1, 2 it', design(2176, 512, it1, n_res(it1), n_jac(it1), 'eq', m=2048, T=160)),
    ('F4+F2 EQ m=2048 at M=640, F1, cap 1 it, no test', design(640, 512, 1, 1, 2, 'eq', m=2048, T=160)),
]
P('| design (q=128) | pass ms | Jacobian ms | ms/step | s/query | vs FOM |')
P('|---|---|---|---|---|---|')
for name, (tot, per, ax, jx) in rows:
    P(f'| {name} | {ax:.3f} | {jx:.3f} | {per:.2f} | {tot / 1e3:.2f} | {tot / fom["fom_ntol0.001"]["ms"]:.2f}x |')
P()
P(f'Latency floor used: {lat:.2f} ms/step = {lat * NSTEPS / 1e3:.2f} s/query at 500 steps -- that alone is '
  f'{lat * NSTEPS / fom["fom_ntol0.001"]["ms"]:.2f}x the FOM. Nothing on this GPU with this loop structure beats the FOM by more than '
  f'~{fom["fom_ntol0.001"]["ms"] / (lat * NSTEPS):.1f}x at 500 steps unless the per-step kernel count also drops.')
P()

# ---- Burgers comparison rows
b = json.load(open(BP))
binv = b['invocations']
P('## F. Burgers panel (bpn301, job 3789570, A100 80GB PCIe, 50 steps of dt=0.005, L=256, K=16, R=512) per-step costs')
P()
P('| arm | M | m | T | LM it/step (median) | ms/query | ms/step |')
P('|---|---|---|---|---|---|---|')
for n in ('q0_M64_dense_g1em06', 'q0_M256_dense_g1em06', 'q128_M576_dense_g1em06', 'q256_M1088_dense_g1em06',
          'q0_M64_eqcert_g1em06', 'q0_M64_eqcert_g0p001', 'q0_M64_eqcert_g1em06_fastL4', 'q128_M576_eqcert_g1em06',
          'q256_M1088_eqcert_g1em06', 'pod32_M128_dense', 'pod512_M2048_dense', 'nt1e-3_dt005', 'fft_tight'):
    t = [i for i in binv if i['name'] == n and i['rep'] > 0]
    if not t:
        continue
    ms = float(np.median([i['gpu_seconds'] for i in t])) * 1e3
    it = float(np.median([i.get('median_iterations') or -1 for i in t]))
    P(f"| {n} | {t[0].get('M')} | {t[0].get('m')} | {t[0].get('solved_dimension') or t[0].get('k') or ''} | {it:.1f} | {ms:.0f} | {ms / 50:.2f} |")
P()
open(os.path.join(HERE, 'tables.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
