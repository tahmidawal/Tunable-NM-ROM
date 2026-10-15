"""jcp-speed3d job context shared by every s3run mode: config, cohort inputs, banks/tables/settings (C_old always loaded),
query and certification caches, single-query rollouts with certification, lattice distances, the A-B-A timing panel
(Step 2 rules, A1.5), E0 / first-order scoring of lattice fields (FOMs, in job), result.json / npz writers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import c3common as C3
import qn3
import s3common as S
import timing as TM

HERE = Path(__file__).resolve().parent
DT0 = S.DT0
COLD = dict(name='M2_512', bank='M2', Rp=512, a=3, rule='gl32')
COLD_SPEC = ('fs', 'BE', DT0, 1)
E0_STORE_VERSION = '3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
E0_STATUS_SHA = 'af7d824e94b42afb83753d6b11a0c4d71407cff0923ebee9c245eb8d51f81005'
E0_COHORT = {923801: 'val64', 923901: 'test32'}


class Ctx:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if not cfg.get('allow_cpu_smoke'):
            assert jax.default_backend() == 'gpu', jax.default_backend()
        assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
        print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
        self.begin = time.perf_counter()
        self.rep = dict(config=cfg, mode=cfg['mode'], commit=os.environ.get('SOURCE_COMMIT'),
                        job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                        gpu=jax.devices()[0].device_kind, nvidia_smi=C3.gpu_info(), jax_version=jax.__version__,
                        rows=[], timing={}, settings={}, gates={}, complete=False)
        seed, count = int(cfg['cohort_seed']), int(cfg['cohort_count'])
        assert seed in (923801, 923901), seed
        if seed == 923901:
            assert cfg['mode'] == 'rep' and cfg.get('frozen_sha256'), 'cohort 923901 is replication-only (DESIGN A1.11)'
        self.seed = seed
        self.tab = C3.C.table(seed, count)
        self.cases = list(cfg.get('case_subset') or range(count))
        self.rep['cohort'] = dict(seed=seed, count=count, table_sha256=self.tab['sha256'], cases=self.cases)
        self.U0 = {j: jax.block_until_ready(jnp.asarray(C3.C.initial_interior(65, self.tab, j))[None]) for j in self.cases}
        self.NU = {j: jnp.asarray([float(self.tab['nu'][j])]) for j in self.cases}
        self.N0 = {j: float(jnp.linalg.norm(self.U0[j])) for j in self.cases}
        sets = list(cfg.get('settings', []))
        if not any(s['name'] == COLD['name'] for s in sets):
            sets = sets + [dict(COLD, reduced=[], timing_only=True)]
        self.sets = sets
        names = {r for s in sets for r in [s['rule']] + s.get('reduced', []) + s.get('extra', [])}
        names |= set(cfg.get('extra_rules', [])) | {COLD['rule']}
        self.rules = S.load_rules(sorted(names))
        self.banks, self.tables = {}, {}
        for bname in sorted({s['bank'] for s in sets}):
            self.banks[bname] = C3.load_bank(bname)
            self.tables[bname] = C3.Tables(65, self.banks[bname], sorted({int(s['Rp']) for s in sets if s['bank'] == bname}),
                                           log=lambda s_: print(s_, flush=True))
        self.ST = {s['name']: S.Setting(s, self.banks, self.tables, self.rules, cfg.get('trust_fraction', .05)) for s in sets}
        self.Gl = {k: jnp.concatenate([g for g in v.lean['GTb']], 0) for k, v in self.ST.items()}   # (R', 63^3)
        self.queries, self.certs = {}, {}
        self.Wstore, self.Cstore, self.acc_sha = {}, {}, {}
        self.UX = {}
        print('SETUP', self.el(), flush=True)

    # ---------------------------------------------------------------------------------------------- utilities
    def el(self):
        return round(time.perf_counter() - self.begin, 1)

    def save(self):
        C3.C.dump(self.out / 'result.json', C3.C.clean(self.rep))

    def dump(self):
        np.savez_compressed(self.out / f"W_{self.cfg['mode']}.npz", **self.Wstore,
                            **{f'cases|{k}': v for k, v in self.Cstore.items()})

    def q_of(self, sname, sp, path='mesh', n=65):
        k = (sname, path, n, sp)
        if k not in self.queries:
            self.queries[k] = S.make_query(sp, self.ST[sname], path=path, n=n)
        return self.queries[k]

    def cert_of(self, sp):
        k = (sp[1], round(sp[2], 12))
        if k not in self.certs:
            self.certs[k] = qn3.make_certify(sp[1], sp[2], S.nsteps(sp[2])[0])
        return self.certs[k]

    def u0x(self, rule, j):
        """u0 at the rule points (mesh-free input), prepared outside every timer."""
        k = (rule, j)
        if k not in self.UX:
            X, _ = self.rules[rule]
            self.UX[k] = jax.block_until_ready(jnp.asarray(C3.C.b3.blob_ic_3d(65, self.tab, j, coords=X))[None])
        return self.UX[k]

    def inp(self, path, rule, j):
        return self.U0[j] if path == 'mesh' else self.u0x(rule, j)

    def dist_cases(self, sname, A_, B_, ids):
        """A2.3 lattice distance per case and output time: (cases, 6)."""
        G = self.Gl[sname]
        out_ = []
        for i, j in enumerate(ids):
            F = (jnp.asarray(A_[i]) - jnp.asarray(B_[i])) @ G
            out_.append(np.asarray(jnp.linalg.norm(F, axis=1)) / self.N0[j])
        return np.array(out_)

    # ----------------------------------------------------------------------------------------------- rollouts
    def rollouts(self, sname, rule, specs, ids, keep_wall=(), path='mesh', data=None, n=65, tag=None, certify=True, in_rule=None):
        """Single queries of every spec on case ids; certification of every non-ADT run; rows appended.
        Returns {key: dict(W (c, 6, R'), valid, nonst, steps)}. tag: storage prefix (default sname|rule)."""
        st = self.ST[sname]
        d = st.data(rule) if data is None else data
        tag = tag or f'{sname}|{rule}' + ('' if path == 'mesh' else f'|{path}') + ('' if n == 65 else f'|n{n}')
        res = {}
        for sp in specs:
            key = S.spec_key(sp)
            f = self.q_of(sname, sp, path, n)
            arr = np.full((len(ids), 6, st.Rp), np.nan)
            walls, vals, nst, nsteps_ = [], [], [], []
            t_s = time.perf_counter()
            for ci, j in enumerate(ids):
                t1 = time.perf_counter()
                o = f(self.inp(path, in_rule or rule, j), self.NU[j], d)
                jax.block_until_ready(o)
                secs = time.perf_counter() - t1
                W = np.asarray(o['W'][:, 0])
                arr[ci] = W
                v = S.run_valid(sp, o, 0)
                row = dict(tag=tag, setting=sname, rule=rule, path=path, n=n, spec=key, case=j, valid=v,
                           seconds_first=secs, steps=S.step_record(sp, o, 0))
                if sp[0] != 'ad' and certify:
                    tc_ = time.perf_counter()
                    g = np.asarray(self.cert_of(sp)(o['W_all'][:, 0], self.NU[j][0], d))
                    ns = int(np.sum(~(g <= 1e-3)))
                    row.update(gam_cert=g.tolist(), nonstationary=ns, steps_n=int(len(g)), cert_seconds=time.perf_counter() - tc_)
                    nst.append(ns)
                    nsteps_.append(len(g))
                    if key in keep_wall:
                        walls.append(np.asarray(o['W_all'][:, 0]))
                vals.append(v)
                self.rep['rows'].append(row)
                self.acc_sha[(tag, key, j)] = C3.sha(W)
                del o
            self.Wstore[f'{tag}|{key}'] = arr
            self.Cstore[f'{tag}|{key}'] = np.array(ids)
            if walls:
                self.Wstore[f'wall|{tag}|{key}'] = np.stack(walls)
            res[key] = dict(W=arr, valid=vals, nonst=nst, steps=nsteps_)
            print('RUN', tag, key, f'{time.perf_counter() - t_s:.1f}s', 'valid', all(vals), 'nonst', sum(nst), '/',
                  sum(nsteps_), self.el(), flush=True)
        return res

    # ------------------------------------------------------------------------------------------------- timing
    def time_panel(self, label, A, Bs, seed_, ncases, reps):
        """A-B-A panel (Step 2 rules). A subject is a dict: name, run(ci) -> output array (hashed), expect(ci) -> the
        accuracy-phase hash or None. Warm JIT: every subject once per panel case before timing."""
        burn = self.cfg['timing']['burn']

        def one(x, ci):
            C3.burn(burn)
            t1 = time.perf_counter()
            o = x['run'](ci)
            o = jax.block_until_ready(o)
            secs = time.perf_counter() - t1
            return secs, C3.sha(np.asarray(o)), x['expect'](ci)
        for x in [A] + Bs:
            for ci in range(ncases):
                one(x, ci)
        rng = np.random.default_rng(seed_)
        inv = []
        for rr in range(reps):
            order_ = [(b, ci) for b in range(len(Bs)) for ci in range(ncases)]
            rng.shuffle(order_)
            for b, ci in order_:
                ta1, h1, e1 = one(A, ci)
                tb, hb, eb = one(Bs[b], ci)
                ta2, h2, _ = one(A, ci)
                inv.append(dict(B=Bs[b]['name'], case=self.cases[ci], rep=rr, tA1=ta1, tB=tb, tA2=ta2,
                                ratio=tb / (.5 * (ta1 + ta2)), drift=ta2 / ta1,
                                hashes_ok=bool(e1 is not None and eb is not None and h1 == e1 and h2 == e1 and hb == eb)))
        self.rep['timing'][label] = dict(A=A['name'], subjects=[x['name'] for x in Bs], invocations=inv,
                                         cases=self.cases[:ncases], reps=reps, burn=burn, seed=seed_, gpu=C3.gpu_info(),
                                         **TM.panel_validity(inv))
        self.save()
        print('TIMING', label, len(inv), 'valid', self.rep['timing'][label]['valid'], self.el(), flush=True)

    def rom_subject(self, sname, rule, sp, path='mesh', data=None, n=65, tag=None, name=None, in_rule=None):
        """Timing subject of one ROM query (output hashed: the coefficient outputs W, as Step 2)."""
        st = self.ST[sname]
        d = st.data(rule) if data is None else data
        tag = tag or f'{sname}|{rule}' + ('' if path == 'mesh' else f'|{path}') + ('' if n == 65 else f'|n{n}')
        key = S.spec_key(sp)
        f = self.q_of(sname, sp, path, n)

        def run(ci):
            j = self.cases[ci]
            return f(self.inp(path, in_rule or rule, j), self.NU[j], d)['W'][:, 0]
        return dict(name=name or f'{tag}|{key}', run=run, expect=lambda ci: self.acc_sha.get((tag, key, self.cases[ci])))

    def t1_subject(self):
        """T1 control: C_old dispatched twice, both synchronized; hash of the second, checked equal to the first."""
        x = self.rom_subject(COLD['name'], COLD['rule'], COLD_SPEC)

        def run(ci):
            a = jax.block_until_ready(x['run'](ci))
            b = jax.block_until_ready(x['run'](ci))
            return b if C3.sha(np.asarray(a)) == C3.sha(np.asarray(b)) else b * np.nan
        return dict(name='T1|C_old twice', run=run, expect=x['expect'])

    # ------------------------------------------------------------------------------------------------ scoring
    def e0(self):
        if not hasattr(self, '_RS'):
            import importlib
            import sys
            d = str(HERE / 'e0loader')
            if d not in sys.path:
                sys.path.insert(0, d)
            RS = importlib.import_module('refstore')
            assert Path(RS.__file__).resolve().parent == Path(d).resolve()
            assert RS.store_version() == E0_STORE_VERSION, 'E0 store version differs from the pin'
            import hashlib
            st = (Path(os.environ.get('JCPREF_META', HERE / 'e0loader' / 'store')) / 'STATUS.json').read_bytes()
            assert hashlib.sha256(st).hexdigest() == E0_STATUS_SHA, 'E0 STATUS.json differs from the pin'
            self._RS = RS
            self._Rc = {}
        return self._RS

    def ref_e0(self, j):
        if j not in getattr(self, '_Rc', {}):
            RS = self.e0()
            coh = E0_COHORT[self.seed]
            R, meta = RS.load(3, coh, int(j), 'ST', field='finest')
            u0 = np.asarray(self.U0[j][0])
            assert abs(meta['judge']['n0'] / self.N0[j] - 1) <= 1e-12, ('E0 normaliser differs', j)
            assert float(np.abs(np.asarray(R[0]) - u0).max()) <= 1e-12 * float(np.abs(u0).max()), ('E0 t0 differs', j)
            st = RS.reference_status(3, coh, int(j), 'ST')
            self._Rc[j] = (np.asarray(R), dict(eps_ref=st['eps_ref'], eligible=bool(st['eligible'])))
        return self._Rc[j]

    def score_fields(self, F, j):
        """In-job E0 score of lattice fields F (6, 63^3): per-time errors relative to ||R(t0)||, worst over t1..t5."""
        R, st = self.ref_e0(j)
        F = np.asarray(F)
        e = [float(np.linalg.norm(F[k] - R[k]) / np.linalg.norm(R[0])) for k in range(6)]
        return dict(e_t=e, e=max(e[1:]), eps_ref=st['eps_ref'], eligible=st['eligible'])
