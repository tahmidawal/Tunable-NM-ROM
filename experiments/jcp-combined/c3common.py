"""jcp-combined 3D shared setup (c3run.py modes esc / grid / bud): banks (W1024 from lane C1, M2 from the 3D lane),
rule sets (vendored rules.npz + C1 rules_extra.npz, sha256-pinned), lean tables, off-mesh rule blocks, the three solver
families and the cohort / refined reference.

Solvers (all end-to-end: L2 projection, stepping, decode of the six output fields):
  fs      fs3.make_fs_query: fixed-sweep LSPG LMM, K sweeps after a_eff adaptive steps (K=1 + BE = deployed)
  vendor  offmesh.make_fsc_rule: the deployed fixed-sweep BE (gate G-FS1)
  ad      t2core.make_evolve (generic adaptive LM to the stopping test on every step), LSPG or GAL, at a tolerance level
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402  (vendored, unchanged)
from offmesh import C, TB  # noqa: E402
from qpanel import lattice65_index  # noqa: E402
import t2core as T2  # noqa: E402
import fs3  # noqa: E402

DT0 = .01
BANKS = {'W1024': (HERE / 'inputs' / 'model_W1024' / 'bank.pkl',
                   '86094f6884e55643240197e560f7dc1924bac3885aed79b20faf2e54446aa036'),
         'M2': (HERE / 'vendor' / 'quad3d' / 'inputs' / 'model_M2' / 'bank.pkl',
                '6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9')}
RULE_FILES = [(HERE / 'vendor' / 'quad3d' / 'rules' / 'rules.npz',
               '099a79976d03e993dc3fbdcc153a57dae0a2dbc5f917d7ca3674bef28daf859b'),
              (HERE / 'rules3d' / 'rules_extra.npz', '8deacf019839b8cf42993d5d8926126541db8152b811f411eceeb69940bb38cf')]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_info():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                                       text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return []


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    a = jax.block_until_ready(_BURN(jnp.ones((512, 512), jnp.float64) * .01))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def check_env(cfg):
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)


def load_rules():
    rules = {}
    for p, want in RULE_FILES:
        assert sha_file(p) == want, p
        z = np.load(p)
        for k in z.files:
            if k.endswith('_X'):
                nm = k[:-2]
                X, w = np.asarray(z[k]), np.asarray(z[nm + '_w'])
                assert nm not in rules or (np.array_equal(rules[nm][0], X) and np.array_equal(rules[nm][1], w)), nm
                rules[nm] = (X, w)
    return rules


def load_bank(name):
    p, want = BANKS[name]
    assert sha_file(p) == want, (name, sha_file(p))
    b = pickle.loads(p.read_bytes())
    return dict(name=name, params=jax.tree_util.tree_map(jnp.asarray, b['params']), T=np.asarray(b['rotation']),
                spread=b['coefficient_rms_spread'], sha256=want)


def load_cohort(cfg, rep):
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    assert cfg['cohort_seed'] == 923801, 'only the validation cohort 923801 is used in this lane'
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    rr = Path(cfg['refined_ref'])
    dn = json.loads(rr.with_suffix('.done').read_text())
    assert dn['sha256'] == sha_file(rr) and dn['accepted'] and dn['seed'] == cfg['cohort_seed'], dn
    z = np.load(rr)
    assert int(z['n']) == 513 and abs(float(z['dt']) - .0025) < 1e-15 and int(z['count']) >= cfg['cohort_count']
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    REF = {j: np.asarray(z[f'c{j}']) for j in cases}
    rep['reference'] = dict(path=str(rr), sha256=dn['sha256'], n=513, dt=.0025)
    return tab, cases, REF


class Tables:
    """Lean tables of one bank at mesh n for the R' values asked (A at M_max, sliced per setting)."""

    def __init__(self, n, bank, Rps, log=print):
        self.n, self.bank = n, bank
        self.order, self.lam_all = C.mode_order(n)
        self.M_of = {r: int(C.complete_M(n, 4 * r, self.order, self.lam_all)) for r in Rps}
        Mmax = max(self.M_of.values())
        self.tb = TB.build_tables(n, bank['params'], bank['T'], sorted(set(Rps)), lambda r: Mmax, log=log, tensor=False)
        self.kx = self.tb['kxyz']

    def lean(self, Rp):
        M = self.M_of[Rp]
        tb = self.tb
        d = dict(GTb=TB.rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp], lam=tb['lam'][:M])
        return jax.tree_util.tree_map(lambda x: jax.block_until_ready(jnp.asarray(x)), d)

    def rule_data(self, Rp, X, w):
        M = self.M_of[Rp]
        om = OM.offmesh_tables(self.n, self.bank['params'], self.bank['T'][:, :Rp], X, w, self.kx[:M])
        return dict(self.lean(Rp), B=om['B'], D=om['D'], P=om['P'])


def nl3(c, d):
    return ((d['B'] @ c) * (d['D'] @ c)) @ d['P']


def nlJ3(c, d, Tm):
    Bc, Dc = d['B'] @ c, d['D'] @ c
    return (Bc * Dc) @ d['P'], d['P'].T @ (Dc[:, None] * d['B'] + Bc[:, None] * d['D'])


def tol_of(form, level):
    """C2 tolerances (A1.10): LSPG tight 1e-7 / tighter 1e-8; GAL 1e-13 / 1e-15; prod as deployed."""
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


class Solvers:
    """Compiled queries of one setting (cached per static signature)."""

    def __init__(self, n, Rp, trust, gtol, a):
        self.n, self.Rp, self.trust, self.gtol, self.a = n, Rp, trust, gtol, a
        self.cache = {}
        self.ev = {f: T2.make_evolve(nl3, nlJ3, f, Rp, trust) for f in ('LSPG', 'GAL')}

    def fs(self, scheme, dt, K):
        k = ('fs', scheme, round(dt, 12), K)
        if k not in self.cache:
            self.cache[k] = fs3.make_fs_query(self.n, self.Rp, OM.contract_offmesh, scheme, dt, K, self.gtol, self.trust,
                                              self.a, dt0=DT0)
        return self.cache[k]

    def vendor(self):
        k = ('vendor',)
        if k not in self.cache:
            self.cache[k] = OM.make_fsc_rule(self.n, self.Rp, None, OM.contract_offmesh, dt=DT0, gtol=self.gtol,
                                             trust=self.trust, adaptive_first=self.a)
        return self.cache[k]

    def ad(self, form):
        k = ('ad', form)
        if k not in self.cache:
            evolve = self.ev[form]

            def query(u0, nu, data, sch):
                w0, s0 = TB.project(u0, data)
                W, st = evolve(w0, nu, jnp.linalg.norm(s0), data, sch)
                return dict(fields=TB.decode(W, data), W=W, w0=w0, stats=st)
            self.cache[k] = jax.jit(query)
        return self.cache[k]

    def run(self, spec, u0, nu, prep):
        """spec: ('fs', scheme, dt, K) | ('vendor',) | ('ad', form, scheme, dt, level); prep = prepare(data) (form data
        built once outside any timing, code audit 1 item 6). Returns (output, W (6,R'), w0)."""
        if spec[0] == 'fs':
            o = self.fs(spec[1], spec[2], spec[3])(u0, nu, prep['base'])
            return o, o['W'], o['w0']
        if spec[0] == 'vendor':
            o = self.vendor()(u0, nu, prep['base'], {})
            keep = int(round(.05 / DT0))
            return o, o[1][::keep], o[1][0]
        _, form, scheme, dt, lev = spec
        g, t_ = tol_of(form, lev)
        o = self.ad(form)(u0, nu, prep[form], T2.sched(scheme, dt, g, t_))
        return o, o['W'], o['w0']


def prepare(data):
    """Base rule data plus the LSPG / GAL form data of t2core (hmask; GAL: thin QR of A), built once per rule."""
    return dict(base=data, LSPG=T2.form_data(data, {}, 'LSPG'), GAL=T2.form_data(data, {}, 'GAL'))


def spec_key(spec):
    if spec[0] == 'fs':
        return f'fs|{spec[1]}|{spec[2]:.8g}|K{spec[3]}'
    if spec[0] == 'vendor':
        return 'vendor|BE|0.01|K1'
    return f'ad|{spec[1]}|{spec[2]}|{spec[3]:.8g}|{spec[4]}'


def run_valid(spec, o):
    """Per-run validity (code audit 1 item 2). fs/vendor: finite coefficients, no reason-3 step, no damping saturation
    (fs only; the vendor does not report it). ad: every step verified (C2 A1.5). Pooled non-stationarity is separate."""
    if spec[0] == 'ad':
        return bool(int(np.asarray(o['stats']['nfail'])) == 0 and np.isfinite(np.asarray(o['W'])).all())
    rs = np.asarray(o['reason'] if spec[0] == 'fs' else o[3])
    sat = int(np.asarray(o['sat']).sum()) if spec[0] == 'fs' else 0
    W = np.asarray(o['W'] if spec[0] == 'fs' else o[1])
    return bool(np.isfinite(W).all() and int((rs == 3).sum()) == 0 and sat == 0)


def nonstationary(spec, o):
    """(non-stationary steps, steps): reasons 0 (budget / not stationary after the sweeps) and 2 (tiny step), C1 A7."""
    if spec[0] == 'ad':
        st = o['stats']
        k = int(st['steps'])
        r = np.asarray(st['step_reason'])[:k]
        return int(np.sum(np.asarray(st['nfail']))), k
    rs = np.asarray(o['reason'] if spec[0] == 'fs' else o[3])
    return int(((rs == 0) | (rs == 2)).sum()), int(len(rs))


def pooled_eligible(records, frac=0.01):
    """Pooled eligibility over a cohort: every run valid and non-stationary steps <= frac of all pooled steps."""
    num = sum(r[1] for r in records)
    den = sum(r[2] for r in records)
    return dict(all_valid=all(r[0] for r in records), nonstationary=num, steps=den,
                eligible=bool(all(r[0] for r in records) and den > 0 and num <= frac * den))


def step_record(spec, o):
    """Full per-step evidence (A1.2; code audit 1 item 9)."""
    if spec[0] == 'fs':
        d = {k: np.asarray(o[k]).tolist() for k in ('reason', 'it', 'rej', 'pred', 'sat', 'gn', 'rn', 'lam')}
        return d
    if spec[0] == 'vendor':
        return dict(reason=np.asarray(o[3]).tolist(), it=np.asarray(o[2]).tolist(), gn=np.asarray(o[4]).tolist(),
                    rn=np.asarray(o[5]).tolist(), rej=np.asarray(o[6]).tolist())
    st = o['stats']
    k = int(st['steps'])
    return dict(nfail=int(st['nfail']), it_sum=int(st['it_sum']), exits=np.asarray(st['exits']).tolist(),
                worst_ratio=float(st['worst_ratio']), steps=k,
                step_gn=np.asarray(st['step_gn'])[:k].tolist(), step_tolratio=np.asarray(st['step_tolratio'])[:k].tolist(),
                step_reason=np.asarray(st['step_reason'])[:k].tolist(), step_it=np.asarray(st['step_it'])[:k].tolist())


def peak_bytes():
    try:
        return int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
    except Exception:  # noqa: BLE001
        return -1
