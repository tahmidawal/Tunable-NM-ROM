"""CPU development check of fscore (not a verdict): L = 256, r = 128, fib4181, dev6 case 0.
Precontraction equivalence (C1), arm 2 vs t2core GAL CN (C2), ETDRK4 linear exactness (C3), certificates on clean runs
and on the mutants (C1m, C2m, C13-C15, C18)."""
import os, sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')
import numpy as np
import fscore as F
import p1common as P
from p1common import Q, T2
import jax, jax.numpy as jnp
import fsrun as R

L, Rp, rule = int(os.environ.get('L', 256)), int(os.environ.get('RP', 128)), os.environ.get('RULE', 'fib4181')
t0 = time.time()
bank = P.Bank(L, [0, 128] if Rp == 128 else [0, 128, 256, 384])
st = R.SettingM(bank, 'R', Rp, rule, roles={'qp': 'fib6765' if Rp == 128 else 'gauss160'})
nl = st.q['GAL']['nl']
dg = st.data[('main', 'GAL')]
tab, chk = F.precontract(dg)
print('checks', json.dumps(chk), round(time.time() - t0, 1), flush=True)
cases = P.case_list({})
ph = cases[0][2]
u0 = jnp.asarray(Q.e.initial(L, ph)); nu = float(ph[4]); n0 = float(np.linalg.norm(np.asarray(u0)[::L // 256, ::L // 256]))
w0, scale = st.q['GAL']['init'](u0, st.cold)
rng = np.random.default_rng(20261021)
z0 = np.asarray(tab['VtRm'] @ w0)
Zs = [z0] + [rng.standard_normal(Rp) * np.linalg.norm(z0) / np.sqrt(Rp) for _ in range(5)]
print('C1 rhs agreement', F.rhs_agreement(Zs, nu, dg, tab, nl))
tab_bad, _ = F.precontract(dg, mutation='wrong_congruence')
print('C1m rhs agreement (must be large)', F.rhs_agreement(Zs, nu, dg, tab_bad, nl))
dec = lambda W: P.restrict(st.q['GAL']['decode'](W, dg), L)
runs = {}
for kind, dt in (('cn', .005), ('etd', .0125), ('etd', .025), ('ros', .005)):
    ev = jax.jit(F.make_evolve(kind, F.adv_pre, F.jac_pre))
    t1 = time.time(); W, s = ev(w0, jnp.float64(nu), tab, F.sched(dt)); jax.block_until_ready(W); t2 = time.time()
    W, s = ev(w0, jnp.float64(nu), tab, F.sched(dt)); jax.block_until_ready(W); t3 = time.time()
    runs[(kind, dt)] = (np.asarray(W), s)
    print(kind, dt, 'compile+run', round(t2 - t1, 2), 'run', round(t3 - t2, 3), 'growth', float(s['growth']), 'iters', int(np.sum(np.asarray(s['it'])[:int(s['steps'])])), flush=True)
# C2: arm 2 vs t2core GAL CN, both tight
evg = jax.jit(T2.make_evolve(nl, Q.tested_jac('point', L), 'GAL', Rp, st.trust))
Wg, sg = evg(w0, nu, scale, dg, T2.sched('CN', .005, 0., 1e-13))
ev = jax.jit(F.make_evolve('cn', F.adv_pre, F.jac_pre))
Wc, sc = ev(w0, jnp.float64(nu), tab, F.sched(.005, level='tight'))
print('C2 dist arm2 vs t2core GAL CN (tight)', P.fdist(dec(Wc), dec(Wg), n0))
# certificates on clean runs
cc, ce, cr = F.make_cert_cn(nl), F.make_cert_etd(nl), F.make_cert_ros(nl)
W, s = runs[('cn', .005)]; k = int(s['steps'])
rn, nA, gam = (np.asarray(x) for x in cc(jnp.asarray(np.asarray(s['Z'])[:k + 1]), jnp.float64(nu), dg, tab['V'], jnp.float64(.005), k))
print('cn cert worst rn/(tol nA)', float(np.max(rn / (1e-10 * nA))), 'gamma max', float(np.max(gam)))
for dt in (.0125, .025):
    W, s = runs[('etd', dt)]; k = int(s['steps'])
    co = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(tab['D']), dt).items()}
    print('etd cert', dt, float(np.max(np.asarray(ce(jnp.asarray(np.asarray(s['Z'])[:k + 1]), dg, tab['V'], co, k)))))
W, s = runs[('ros', .005)]; k = int(s['steps'])
jr, lr = cr(jnp.asarray(np.asarray(s['Z'])[:k + 1]), jnp.float64(nu), dg, tab['V'], tab, jnp.float64(.005), k)
print('ros cert jac', float(np.max(np.asarray(jr))), 'lin', float(np.max(np.asarray(lr))))
# mutants
for kind, dt, mut in (('etd', .0125, 'f3_scale'), ('etd', .0125, 'f2_as_f1'), ('etd', .0125, 'wrong_nu'), ('ros', .005, 'half_update'),
                      ('cn', .005, 'newton0'), ('etd', .0125, 'neg_nu'), ('etd', .0125, 'late_out')):
    ev = jax.jit(F.make_evolve(kind, F.adv_pre, F.jac_pre, mutation=mut))
    W, s = ev(w0, jnp.float64(nu), tab, F.sched(dt)); k = int(s['steps']); Z = jnp.asarray(np.asarray(s['Z'])[:k + 1])
    if kind == 'etd':
        co = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(tab['D']), dt).items()}
        m = float(np.max(np.asarray(ce(Z, dg, tab['V'], co, k))))
    elif kind == 'ros':
        m = float(np.max(np.asarray(cr(Z, jnp.float64(nu), dg, tab['V'], tab, jnp.float64(dt), k)[1])))
    else:
        rn, nA, _ = (np.asarray(x) for x in cc(Z, jnp.float64(nu), dg, tab['V'], jnp.float64(dt), k)); m = float(np.mean(rn > 1e-10 * nA))
    zo = np.asarray(s['z_out']); Zn = np.asarray(s['Z'])
    print('mutant', kind, mut, 'certificate', m, 'growth', float(s['growth']), 'out_vs_states', float(np.max(np.abs(zo[1:] - Zn[[k // 5 * j for j in range(1, 6)]]))), flush=True)
print('total', round(time.time() - t0, 1))
