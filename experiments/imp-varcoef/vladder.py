"""imp-varcoef ladder job (DESIGN §7, A1.10, A1.12, A1.17, A2.10, A2.21, A2.22, A3.10, A3.23, A4.10).

    python vladder.py --config configs/<job>.json --out <dir>

config: {"contrasts": [100, 10], "keys": [...], "refdir": "<dir with ref_C*_*.npz>", "eligibility": "eligibility.json",
         "taus": [0.01, 0.03, 0.1], "screen": {"mesh": 4096, "reps": 3, "burn": 0.1}, "ntop": 3,
         "extensions": true, "qualify": true}
Phases per contrast: A dev ladder -> B extensions (cheap edge, small step, fixed-case temporal check) on dev -> C val for
configurations surviving lossless pruning -> D screening cost (full-field 4096^2 contract) -> E qualification of timing
candidates (tight-solve distance, tight/tighter, spectral Q2 analogue), replacing failures by the next-ranked.
Writes <dir>/result.json (every run), <dir>/candidates.json (timing input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcoef as VC     # noqa: E402
import vspec as VS     # noqa: E402
import vkeys as VK     # noqa: E402

BAR_TIGHT = 2.5e-4
BAR_TT = 2.5e-5
BAR_RHO = 0.116
FAMILIES = ('PSI', 'ETD', 'FDMG', 'CCV')
REQUIRED = ('PSI', 'ETD', 'FDMG')
PRUNE = 0.10


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


_BURN = jax.jit(lambda a: jnp.tanh(a @ a) * .5)


def burn(seconds):
    a = jnp.ones((512, 512), jnp.float64) * .01
    jax.block_until_ready(_BURN(a))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def lat_dist(F, G, n0):
    return float(max(np.linalg.norm((np.asarray(F[k]) - np.asarray(G[k])).ravel()) for k in range(1, 6)) / n0)


def smoke_eligibility(path):
    d = json.loads(Path(path).read_text())
    out = dict(contrasts={}, smoke=True)
    for k, v in d['cases'].items():
        C, cid = k.split('|', 1)
        C = C[1:]
        e = out['contrasts'].setdefault(C, dict(eligible=[], slots=[]))
        if v['judge']['outcome'] == 'accepted' and v['eps_ref'] is not None and v['eps_ref'] <= 1e-3:
            e['eligible'].append(cid)
    for C, e in out['contrasts'].items():
        allc = sorted({k.split('|', 1)[1] for k in d['cases'] if k.startswith(f'C{C}|')})
        src = e['eligible'] or allc
        e['slots'] = [dict(slot=s, anchor=['dev6', s], fom=src[s % len(src)]) for s in range(6)]
    return out


class Ladder:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, out
        if cfg.get('eligibility_from'):                           # smoke only: mask built in-job from the smoke refs
            self.el = smoke_eligibility(cfg['eligibility_from'])
            cfg = dict(cfg, eligibility=cfg['eligibility_from'])
            self.cfg = cfg
        else:
            raw = Path(cfg['eligibility']).read_bytes()
            fro = Path(cfg['eligibility']).with_suffix('.FROZEN').read_text().split()[0]
            assert hashlib.sha256(raw).hexdigest() == fro == cfg['eligibility_sha256'], 'eligibility mask not the frozen one'
            self.el = json.loads(raw)
        self.cases = {s: VC.cohort(s) for s in ('dev', 'val')}
        self.progs = {}
        self.runs = {}            # (key, C, case_id) -> record
        self.fields = {}          # (key, C, case_id) -> lattice fields (kept only for qualification comparisons)
        self.refs = {}
        self.compile = {}
        self.stage_arrays = {}
        self.rep = dict(config=cfg, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                        jax_version=jax.__version__, eligibility_sha256=hashlib.sha256(
                            Path(cfg['eligibility']).read_bytes()).hexdigest(), phases={}, extensions={},
                        temporal={}, screening={}, qualification={}, candidates={}, controls={})
        self.T0 = time.perf_counter()

    # -------------------------------------------------------------------------------------------- basics --------
    def ref(self, C, cid):
        k = (C, cid)
        if k not in self.refs:
            split, j = cid.split('|')
            dirs = self.cfg['refdir'] if isinstance(self.cfg['refdir'], list) else [self.cfg['refdir']]
            fs = [Path(d_) / f'ref_C{C:g}_{split}_{j}.npz' for d_ in dirs]
            fs = [f_ for f_ in fs if f_.exists()]
            assert len(fs) >= 1, ('no reference', C, cid)
            z = np.load(fs[-1])                                   # later directories (fallback reruns) win
            want = self.el['contrasts'][f'{C:g}'].get('cases', {}).get(cid, {}).get('ref_sha256')
            if not self.el.get('smoke'):
                assert want is not None and sha(z['lat']) == want, ('reference differs from the frozen mask', C, cid)
            self.refs[k] = (z['lat'], float(np.linalg.norm(z['t0'].ravel())))
        return self.refs[k]

    def eligible(self, C, cid):
        return cid in self.el['contrasts'][f'{C:g}']['eligible']

    def prog(self, key, out='lattice'):
        k = (key, str(out))
        if k not in self.progs:
            self.progs[k] = VK.program(VK.parse(key), out)
        return self.progs[k]

    def case(self, cid):
        s, j = cid.split('|')
        return self.cases[s][int(j)]

    def run(self, key, C, cid, keep_fields=False):
        rk = (key, C, cid)
        if rk in self.runs and (not keep_fields or rk in self.fields):
            return self.runs[rk]
        sp = VK.parse(key)
        c = self.case(cid)
        x0, tp = VK.prepare(sp, c)
        kp = jnp.asarray(VC.kappa_params(c, C))
        p = self.prog(key)
        t = time.perf_counter()
        F, st = p(x0, kp)
        F = jax.block_until_ready(F)
        secs = time.perf_counter() - t
        if key not in self.compile:
            self.compile[key] = secs
        ok, ev = VK.validity(sp, F, st)
        for a_ in ('it_stage', 'res_stage'):                     # per-stage evidence -> stages npz (A1.12)
            if a_ in ev:
                self.stage_arrays[f'{key}@C{C:g}@{cid}@{a_}'] = np.asarray(ev.pop(a_))
        F = np.asarray(F)
        R, n0 = self.ref(C, cid)
        e = VK.lat_err(F, R, n0) if ok else None
        rec = dict(key=key, C=C, case=cid, valid=ok, e=e, eligible=self.eligible(C, cid), ev=ev, seconds=secs,
                   lattice_sha256=sha(F))
        if sp['f'] in ('psi', 'etd'):
            rec['_coefs'] = np.asarray(jax.device_get(st['coefs']))
        self.runs[rk] = rec
        if keep_fields:
            self.fields[rk] = F
        return rec

    def ids(self, split):
        if 'ids' in self.cfg:                                     # smoke only: a restricted case list
            return list(self.cfg['ids'].get(split, []))
        return [f'{split}|{j}' for j in range(len(self.cases[split]))]

    def stats(self, key, C, ids):
        rr = [self.runs.get((key, C, i)) for i in ids]
        if any(r is None for r in rr):
            return None
        valid = all(r['valid'] for r in rr)
        el = [r['e'] for r in rr if r['eligible'] and r['valid']]
        return dict(valid=valid, n=len(rr), n_eligible=len(el), median=float(np.median(el)) if el else None,
                    worst=float(np.max(el)) if el else None)

    def save(self):
        rows = [{k: v for k, v in r.items() if not k.startswith('_')} for r in self.runs.values()]
        self.rep['runs'] = rows
        if self.stage_arrays:
            np.savez_compressed(self.out / 'stages.npz', **self.stage_arrays)
        self.rep['compile_seconds'] = self.compile
        self.rep['elapsed_seconds'] = time.perf_counter() - self.T0
        (self.out / 'result.json').write_text(json.dumps(self.rep, indent=1, default=float))

    # -------------------------------------------------------------------------------------------- phases --------
    def ladder(self, keys, C, ids, tag):
        for i, k in enumerate(keys):
            for cid in ids:
                try:
                    self.run(k, C, cid)
                except Exception as ex:          # a configuration that cannot run is invalid, recorded
                    self.runs[(k, C, cid)] = dict(key=k, C=C, case=cid, valid=False, e=None,
                                                  eligible=self.eligible(C, cid), ev=dict(error=repr(ex)[:300]))
            if (i + 1) % 10 == 0:
                log(f'{tag} C={C:g} {i + 1}/{len(keys)} elapsed {time.perf_counter() - self.T0:.0f}s')
                self.save()

    def dev_ok(self, k, C, tau):
        s = self.stats(k, C, self.ids('dev'))
        if s is None or not s['valid']:
            return False
        return s['worst'] is None or s['worst'] <= tau      # no eligible dev case -> no pruning (A2.17)

    def rows(self, keys):
        """Row = all knobs except dt (A3.10)."""
        R = {}
        for k in keys:
            sp = VK.parse(k)
            r = dict(sp)
            r.pop('dt')
            R.setdefault(json.dumps(r, sort_keys=True), []).append(k)
        return {r: sorted(v, key=lambda k: -VK.parse(k)['dt']) for r, v in R.items()}

    def extensions(self, keys, C):
        """A2.21 / A3.10 / A4.10 on dev: cheap-edge, small-step and fixed-case temporal extensions."""
        dev = self.ids('dev')
        new = []
        rows = self.rows(keys)
        fams = {}
        for k in keys:
            fams.setdefault(VK.family(k), []).append(k)
        # fixed-case temporal check (A4.10): largest N/L, tightest tol, finest dt vs dt/2 on dev case 0
        temporal = {}
        for f, ks in fams.items():
            res = max(VK.resolution(VK.parse(k)) for k in ks)
            cand = [k for k in ks if VK.resolution(VK.parse(k)) == res]
            tight = lambda sp: sp['tol'] if 'tol' in sp else (int(sp['level']) if 'level' in sp else 0)
            kmin = min(cand, key=lambda k: (VK.parse(k)['dt'], tight(VK.parse(k)), k))
            sp = VK.parse(kmin)
            sp2 = dict(sp, dt=sp['dt'] / 2)
            k2 = VK.key(sp2)
            r1, r2 = self.run(kmin, C, dev[0], True), self.run(k2, C, dev[0], True)
            d = lat_dist(self.fields[(kmin, C, dev[0])], self.fields[(k2, C, dev[0])], self.ref(C, dev[0])[1]) \
                if r1['valid'] and r2['valid'] else None
            temporal[f] = dict(key=kmin, half=k2, distance=d, accepted=bool(d is not None and d <= 1e-3))
        self.rep['temporal'][f'{C:g}'] = temporal
        for r, ks in rows.items():
            f = VK.family(ks[0])
            sts = [self.stats(k, C, dev) for k in ks]
            anyvalid = any(s and s['valid'] for s in sts)
            limited = False
            if len(ks) >= 2 and sts[-1] and sts[-2] and sts[-1]['valid'] and sts[-2]['valid'] \
                    and sts[-1]['worst'] is not None and sts[-2]['worst'] is not None:
                limited = sts[-1]['worst'] < 0.8 * sts[-2]['worst']
            if (not anyvalid) or limited or not temporal.get(f, {}).get('accepted', True):
                sp = VK.parse(ks[-1])
                for div in (2, 4):
                    new.append(VK.key(dict(sp, dt=sp['dt'] / div)))
        # cheap edge (A1.10, restated for code audit 1 item 8): every dev-valid configuration matching a target that sits on
        # a cheap edge of a knob is extended one step beyond that edge (a superset of the provisional-winner rule; no
        # latency is used to choose)
        edge = {'N': 12, 'L': 32}
        for f, ks in fams.items():
            allsp = [VK.parse(k) for k in ks]
            rk = 'N' if 'N' in allsp[0] else 'L'
            for k in ks:
                if not any(self.dev_ok(k, C, tau) and self.stats(k, C, dev)['worst'] is not None for tau in self.cfg['taus']):
                    continue
                sp = VK.parse(k)
                if sp[rk] == min(x[rk] for x in allsp):
                    new.append(VK.key(dict(sp, **{rk: edge[rk]})))
                if sp['dt'] == max(x['dt'] for x in allsp):
                    new.append(VK.key(dict(sp, dt=0.05)))
                if 'tol' in sp and sp['tol'] == max(x['tol'] for x in allsp if 'tol' in x):
                    new.append(VK.key(dict(sp, tol=1e-2)))
                if 'level' in sp and sp['level'] == max((x['level'] for x in allsp if 'level' in x), key=int):
                    new.append(VK.key(dict(sp, level='4')))
        new = sorted(set(new) - set(keys))
        self.rep['extensions'][f'{C:g}'] = new
        log(f'EXT C={C:g}: {len(new)} extension configurations')
        self.ladder(new, C, dev, 'EXT')
        # A5.10: a failed temporal check is repeated on the two finest (extended) steps of the same row
        for f, t in temporal.items():
            if t['accepted']:
                t['final'] = 'accepted'
                continue
            sp = VK.parse(t['key'])
            ka, kb = VK.key(dict(sp, dt=sp['dt'] / 2)), VK.key(dict(sp, dt=sp['dt'] / 4))
            ra, rb = self.run(ka, C, dev[0], True), self.run(kb, C, dev[0], True)
            d2 = lat_dist(self.fields[(ka, C, dev[0])], self.fields[(kb, C, dev[0])], self.ref(C, dev[0])[1]) \
                if ra['valid'] and rb['valid'] else None
            t.update(after=dict(key=ka, half=kb, distance=d2), final='accepted' if (d2 is not None and d2 <= 1e-3) else
                     'unresolved')
        return new

    def tkappa(self, C, dev):
        """Scientific diagnostic T-kappa (A1.19): the constant-coefficient solver (kappa = nu; ETDRK4 N = 48,
        dt = 0.0125, i.e. jcp-fom2d PS) scored against this contrast's references."""
        k = 'etd|N48|0.0125|nu'
        rows = []
        for cid in dev:
            c = self.case(cid)
            sp = VK.parse(k)
            x0, _ = VK.prepare(sp, c)
            F, st = self.prog(k)(x0, jnp.asarray(VC.kappa_params(c, 1.0)))
            F = np.asarray(jax.block_until_ready(F))
            R, n0 = self.ref(C, cid)
            rows.append(dict(case=cid, e=VK.lat_err(F, R, n0), eligible=self.eligible(C, cid)))
        el = [r['e'] for r in rows if r['eligible']]
        self.rep['controls'][f'T-kappa@C{C:g}'] = dict(kind='scientific diagnostic', key=k, rows=rows,
                                                      worst=float(max(el)) if el else None,
                                                      median=float(np.median(el)) if el else None)

    def screen(self, keys, C):
        sc = self.cfg['screen']
        slots = self.el['contrasts'][f'{C:g}']['slots'][:2]
        for k in keys:
            sp = VK.parse(k)
            p = self.prog(k, ('mesh', int(sc['mesh'])))
            ts = []
            for s in slots:
                c = self.case(s['fom'])
                x0, _ = VK.prepare(sp, c)
                kp = jnp.asarray(VC.kappa_params(c, C))
                jax.block_until_ready(p(x0, kp)[0])
                for _ in range(int(sc['reps'])):
                    burn(float(sc['burn']))
                    t = time.perf_counter()
                    jax.block_until_ready(p(x0, kp)[0])
                    ts.append(time.perf_counter() - t)
            self.rep['screening'][f'{k}@C{C:g}'] = dict(times=ts, cost=float(np.median(ts)))
            # A7.1: compiled programs are shared via vspec.cached and kept across contrasts

    def rho(self, key, key3, C, cid):
        """A3.23 own-state operator discrepancy over the six reached states of `key`'s trajectory."""
        sp = VK.parse(key)
        c = self.case(cid)
        kp = jnp.asarray(VC.kappa_params(c, C))
        N = sp['N']
        Q2 = sp.get('Q') or 2 * N
        D2, _ = VS.make_D(N, Q2, kp)
        D3, _ = VS.make_D(N, 3 * N, kp)
        out = []
        for b in self.runs[(key, C, cid)]['_coefs']:
            b = jnp.asarray(b)
            a, d = float(jnp.linalg.norm(D2(b) - D3(b))), float(jnp.linalg.norm(D3(b)))
            out.append(0.0 if (a == 0 and d == 0) else (float('inf') if d == 0 else a / d))
        return max(out)

    def qualify(self, key, C, pooled):
        sp = VK.parse(key)
        f = sp['f']
        q = dict(key=key, C=C)
        for cid in pooled:
            self.run(key, C, cid, True)
        if f in ('psi', 'fdmg', 'ccv'):
            tk = VK.key(dict(sp, tol=1e-10)) if f != 'ccv' else VK.key(dict(sp, level='E'))     # CCv: E0 tight
            tt = VK.key(dict(sp, tol=1e-12)) if f != 'ccv' else VK.key(dict(sp, level='F'))     # CCv: tighter anchor
            dist = []
            for cid in pooled:
                r = self.run(tk, C, cid, True)
                n0 = self.ref(C, cid)[1]
                dist.append(lat_dist(self.fields[(key, C, cid)], self.fields[(tk, C, cid)], n0) if r['valid'] else float('inf'))
            q['tight_key'], q['tight_dist_max'] = tk, float(max(dist))
            q['tight_ok'] = bool(max(dist) <= BAR_TIGHT)
            tt_d = []
            for cid in self.ids('dev')[:2]:
                r1, r2 = self.run(tk, C, cid, True), self.run(tt, C, cid, True)
                tt_d.append(lat_dist(self.fields[(tk, C, cid)], self.fields[(tt, C, cid)], self.ref(C, cid)[1])
                            if r1['valid'] and r2['valid'] else float('inf'))
            q['tighter_key'], q['tt_max'] = tt, float(max(tt_d))
            q['tt_ok'] = bool(max(tt_d) <= BAR_TT)
        else:
            q['tight_ok'] = q['tt_ok'] = True
            q['note'] = 'ETD has no solver: Q1 = run validity only (A1.12)'
        if f in ('psi', 'etd'):
            k3 = VK.key(dict(sp, Q=3 * sp['N']))
            dd, rh = [], []
            for cid in pooled:
                r = self.run(k3, C, cid, True)
                n0 = self.ref(C, cid)[1]
                dd.append(lat_dist(self.fields[(key, C, cid)], self.fields[(k3, C, cid)], n0) if r['valid'] else float('inf'))
                rh.append(self.rho(key, k3, C, cid))
            q['q3_key'], q['q2_dist_max'], q['rho_max'] = k3, float(max(dd)), float(max(rh))
            q['q2_ok'] = bool(max(dd) <= BAR_TIGHT and max(rh) <= BAR_RHO)
        else:
            q['q2_ok'] = True
        q['valid_all'] = all(self.runs[(key, C, cid)]['valid'] for cid in pooled)
        q['qualified'] = bool(q['valid_all'] and q['tight_ok'] and q['tt_ok'] and q['q2_ok'])
        for kk in [x for x in (q.get('tight_key'), q.get('tighter_key'), q.get('q3_key')) if x]:   # release (memory)
            for cid in pooled:
                self.fields.pop((kk, C, cid), None)
        return q

    def main(self):
        keys0 = list(self.cfg['keys'])
        cands_out = {}
        for C in self.cfg['contrasts']:
            C = float(C)
            dev, val = self.ids('dev'), self.ids('val')
            log(f'=== C={C:g}: dev ladder, {len(keys0)} configurations')
            self.ladder(keys0, C, dev, 'DEV')
            self.tkappa(C, dev)
            self.save()
            ext = self.extensions(keys0, C) if self.cfg.get('extensions', True) else []
            keys = keys0 + ext
            self.save()
            if self.cfg.get('dev_only', {}).get(f'{C:g}'):       # A7.4: deferred contrast -- dev ladder only, no selection
                self.rep['phases'][f'{C:g}'] = dict(n_keys=len(keys), n_val=0, mode='dev-only (deferred: val, screening, '
                                                                                 'qualification, timing)')
                self.save()
                continue
            survive = [k for k in keys if self.dev_ok(k, C, PRUNE)]
            log(f'VAL C={C:g}: {len(survive)} of {len(keys)} survive pruning')
            self.rep['phases'][f'{C:g}'] = dict(n_keys=len(keys), n_val=len(survive))
            self.ladder(survive, C, val, 'VAL')
            self.save()
            pooled = dev + val
            match = {}
            for k in survive:
                s = self.stats(k, C, pooled)
                if s and s['valid'] and s['worst'] is not None:
                    match[k] = s
            self.rep['pooled'] = self.rep.get('pooled', {})
            self.rep['pooled'][f'{C:g}'] = match
            scr = [k for k, s in match.items() if s['worst'] <= max(self.cfg['taus'])]
            log(f'SCREEN C={C:g}: {len(scr)} configurations')
            self.screen(scr, C)
            self.save()
            cost = lambda k: self.rep['screening'][f'{k}@C{C:g}']['cost']
            groups = {}
            for f in FAMILIES:
                for tau in self.cfg['taus']:
                    ranked = sorted([k for k in scr if VK.family(k) == f and match[k]['worst'] <= tau], key=lambda k: (cost(k), k))
                    chosen, tried = [], []
                    for k in ranked:
                        if len(chosen) >= int(self.cfg.get('ntop', 3)) or len(tried) >= int(self.cfg.get('max_tries', 8)):
                            break                              # A7.2: at most 8 qualification attempts per group
                        qk = f'{k}@C{C:g}'
                        if qk not in self.rep['qualification']:
                            self.rep['qualification'][qk] = self.qualify(k, C, pooled) if self.cfg.get('qualify', True) \
                                else dict(qualified=True, note='qualification disabled (smoke)')
                            self.save()
                        tried.append(k)
                        if self.rep['qualification'][qk]['qualified']:
                            chosen.append(k)
                    groups[f'{f}|{tau:g}|C{C:g}'] = dict(family=f, tau=tau, C=C, ranked=ranked[:20], tried=tried,
                                                         candidates=chosen,
                                                         status='ok' if chosen else ('no qualified' if ranked else 'no match'))
            self.rep['candidates'][f'{C:g}'] = groups
            # ladder errors per timing slot for the timing job's G-FR identity gate
            for g in groups.values():
                for k in g['candidates']:
                    tk = f'{k}@C{C:g}'
                    cands_out[tk] = dict(key=k, C=C, family=g['family'], screening=cost(k),
                                         pooled=match[k],
                                         slot_e={s['fom']: self.runs[(k, C, s['fom'])]['e']
                                                 for s in self.el['contrasts'][f'{C:g}']['slots']})
            # free kept fields between contrasts
            self.fields = {k: v for k, v in self.fields.items() if k[1] != C}
            self.save()
        self.rep['complete'] = True
        self.save()
        (self.out / 'candidates.json').write_text(json.dumps(dict(candidates=cands_out, groups=self.rep['candidates']),
                                                             indent=1, default=float))
        (self.out / 'COMPLETE').write_text('complete\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Ladder(cfg, out).main()


if __name__ == '__main__':
    main()
