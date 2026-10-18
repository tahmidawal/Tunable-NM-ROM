"""jcp-dims 3D analysis (DESIGN 2.3, 3, 6.2, 6.4, 6.5, A1.2, A1.5, A2.1-A2.3, A3.2, A6.2; CPU only). Inputs: the collected
d3run archives (q3r<k>, m3a, m3b, k3, b3f, b3m, t3s, t3c) and their offline E0 scores (score3.py -> checks/scores-<att>.json).

  * Expected ladders come from configs/MANIFEST-cells3d.json (a missing job / rung is 'missing', never skipped).
  * Ladder (fs / qn) configuration: Q1 = speed3d eligibility on the REGISTERED population (all 64 cases; anchors
    tighter on cases 0-7) with the in-job ADT distance recomputed offline (score3 lattice rows) and non-finite values
    rejected; adaptive configuration (k3): validity, pooled certificate <= 1 %, G-cert, tight / tighter (A2.1 square rules
    in-job). Ladder Q2 per A1.5 (top two rungs: ladder top; interior: successor distance recomputed offline from the
    archived coefficients, rho from the job, check rung on cases 0-7); an independent NumPy rho (vendored npcore bank)
    on case 0 for configurations with m * M <= 1e8 (A6.2), which must agree.
  * Rounds members (q3r<k>): qualify's Q2-3D predicate; Q1 = speed3d eligibility (cited).
  * m* per ladder, width m* (intersection over the three registered specs), family m* (five draws), point efficiency,
    the cross-family gate (A3.2), the bad-vector control (fixture admitted + in-job flag + agreement with the
    independent establishment), B6-3D transitions with control mapping (A2.1), A3-3D panels.
  * --after-screen: timing-job qualification from the t3s certification blocks, screening costs, H2q selection
    (dstats.h2q), the qualified front (valid screen only), C^q_w, open candidates; after t3c: confirmation verdicts.

    JAX_PLATFORMS=cpu python analyze3d.py [--after-screen]
"""
from __future__ import annotations

import offline  # noqa: F401  (CPU, highest, x64)

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import dstats as DS

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
CFG = HERE / 'configs'
SPEED3D = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d')
QUALIFY = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify')
BAR, BAR_TT, RHO = 2.5e-4, 2.5e-5, .116
COMPARATORS = {'C_old': 'M2_512|gl32|fs|BE|0.01|K1', 'C_W768': 'W768|lat32768|fs|BE|0.01|K1',
               'C_W1024': 'W1024|lat65536|fs|BE|0.01|K1'}
ALL64, CHK = list(range(64)), list(range(8))
_DEP = None


def dep_pins():
    """Hash of every numerical dependency of the offline checks (checker code, banks, rules, fixture), bound into caches."""
    global _DEP
    if _DEP is None:
        import hashlib
        h = hashlib.sha256()
        for f in ('analyze3d.py', 'rules3d_names.py', 'vendor/quad3d/npcore.py', 'vendor/quad3d/rules.py', 'inputs/model_W1024/bank.pkl',
                  'vendor/quad3d/inputs/model_M2/bank.pkl', 'vendor/quad3d/rules/rules.npz', 'rules3d/rules_extra.npz',
                  'rules3d/rules_dims.npz', 'inputs/fixture3d.npz', 'score3.py'):
            h.update((HERE / f).read_bytes())
        _DEP = h.hexdigest()
    return _DEP


def out_dir(att):
    import dscore as DSC
    return DSC.out_dir(att, 'd3run', att)


def m3(rule):
    f, n = re.match(r'^([a-z]+)(\d+)', rule).groups()
    n = int(n)
    if f == 'gl':
        return n ** 3
    if f in ('smolyak', 'smol'):
        return {8: 2559, 9: 6271, 10: 15103, 11: 35839, 12: 83967, 13: 194559}[n]
    return n


def base(setting):
    return re.match(r'^(M2_512|W\d+)', setting).group(1)


def glat(setting):
    import score3
    return score3.glat(base(setting))


_N0 = {}


def n0s(cases):
    import c3common as C3
    if not _N0:
        tab = C3.C.table(923801, 64)
        _N0.update({j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in ALL64})
    return np.array([_N0[j] for j in cases])


def odist(W, Wn, setting, cases):
    D = (np.asarray(W) - np.asarray(Wn)) @ glat(setting)
    return (np.linalg.norm(D, axis=-1) / n0s(cases)[:, None])[:, 1:].max(1)


def fin_max(x):
    a = np.asarray(x, float)
    return float(a.max()) if a.size and np.isfinite(a).all() else np.inf


# ------------------------------------------------------------------------------------------------ evidence
def collect():
    C, jobs = {}, {}
    for att in ('m3a', 'm3b', 'k3', 'b3f', 'b3m'):
        d = out_dir(att)
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        if not res.get('complete') or res.get('memory_rejected'):
            continue
        jobs[att] = res
        p = HERE / 'checks' / f'scores-{att}.json'
        sc = json.loads(p.read_text())['scores'] if p.exists() else {}
        W = np.load(d / f"W_{res['mode']}.npz")
        for blk, rb in zip(res['config']['blocks'], res['blocks']):
            if blk['kind'] == 'ladder':
                for k, e in rb['result'].items():
                    sp = e['spec'].split('|')
                    adt = f"{e['setting']}|{e['rule']}|ad|LSPG|{sp[1]}|{float(sp[2]):.8g}|tight"
                    C[k] = dict(att=att, kind='ladder', entry=e, setting=e['setting'], spec=e['spec'], rule=e['rule'], M=e['M'],
                                r=e['Rp'], cases=e['cases'], e0=sc.get(k), W=W[k], W_adt=W[adt] if adt in W.files else None,
                                expected_cases=blk.get('cases') or ALL64, valid_cases=dict(zip(e['cases'], e['valid'])))
            elif blk['kind'] == 'adaptive':
                for k, e in rb['result'].items():
                    z = np.load(d / f"AD_{k.replace('|', '_')}.npz")
                    cs = [int(x) for x in z['case_ids']]
                    C[k] = dict(att=att, kind='adaptive', entry=dict(e, rho=z['rho'].tolist()),
                                setting=k.split('|')[0], spec='ad|' + k.split('|')[3], rule=k.split('|')[1], M=int(z['M']),
                                r=int(z['Rp']), cases=cs, e0=sc.get(k + '|prod'), W=np.asarray(z['W']), z=z,
                                expected_cases=blk.get('cases') or ALL64, valid_cases=dict(zip(cs, [bool(x) for x in z['valid']])),
                                expected_dt=float(blk['dt']))
    return C, jobs


def q1_of(c):
    exp = c['expected_cases']
    tcs = [j for j in CHK if j in exp]
    if c['kind'] == 'ladder':
        e = c['entry']
        pop = (e['cases'] == exp and e['tighter_cases'] == tcs and len(e['dist_adt']) == len(exp) and len(e['valid']) == len(exp)
               and len(e['adt_valid']) == len(exp) and len(e['adt_tighter_valid']) == len(tcs) and len(e['adt_tight_tighter']) == len(tcs))
        off = odist(c['W'], c['W_adt'], c['setting'], c['cases']) if c['W_adt'] is not None else None
        agree = off is not None and bool(np.allclose(off, e['dist_adt'], rtol=1e-9, atol=1e-13))
        return dict(population=bool(pop), valid=all(e['valid']), certificate=bool(e['steps'] > 0 and e['nonstationary'] <= .01 * e['steps']),
                    distance=fin_max(e['dist_adt']) <= BAR, anchors_valid=all(e['adt_valid']) and all(e['adt_tighter_valid']),
                    tight_tighter=bool(e['adt_tight_tighter']) and fin_max(e['adt_tight_tighter']) <= BAR_TT, offline_adt_agrees=agree)
    z = c['z']
    S_ = z['cert'].shape[1]
    tids = [int(x) for x in z['tighter_ids']]
    off_t = odist(z['W'], z['W_tight'], c['setting'], c['cases'])
    off_tt = odist(z['W_tight'][[c['cases'].index(j) for j in tids]], z['W_tighter'], c['setting'], tids) if tids else np.array([])
    agree = bool(np.allclose(off_t, z['dist_tight'], rtol=1e-9, atol=1e-13) and np.allclose(off_tt, z['dist_tight_tighter'], rtol=1e-9, atol=1e-13))
    n = len(exp)
    steps_reg = int(round(.25 / float(c['expected_dt'])))           # from the registered block, not from the archive
    return dict(population=bool(c['cases'] == exp and tids == tcs and len(z['valid']) == n and len(z['valid_tighter']) == len(tcs)
                                and len(z['valid_tight']) == n and len(z['gcert']) == n and len(z['noncert']) == n
                                and z['cert'].shape == (n, steps_reg) and len(z['dist_tight']) == n and len(z['dist_tight_tighter']) == len(tcs)),
                offline_tight_agrees=agree, valid=bool(np.all(z['valid'])),
                certificate=bool(np.sum(z['noncert']) <= .01 * len(c['cases']) * S_), gcert=bool(np.all(z['gcert'])),
                distance=fin_max(z['dist_tight']) <= BAR, anchors_valid=bool(np.all(z['valid_tight']) and np.all(z['valid_tighter'])),
                tight_tighter=fin_max(z['dist_tight_tighter']) <= BAR_TT)


_NB = {}


def rho_offline(c, qp):
    """A6.2: independent NumPy rho (vendored npcore bank) on case 0, six outputs, when m * M <= 1e8."""
    if m3(c['rule']) * c['M'] > 1e8 or m3(qp) * c['M'] > 2e8 or not {0, 1} <= set(c['cases']):
        return None
    cp = HERE / 'checks' / 'rho3d-offline.json'
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    import hashlib
    bind = hashlib.sha256(np.ascontiguousarray(np.asarray(c['W'])[[c['cases'].index(0), c['cases'].index(1)]]).tobytes()
                          + json.dumps(np.asarray(c['entry']['rho'], float)[[c['cases'].index(0), c['cases'].index(1)]].tolist()).encode()
                          + dep_pins().encode() + str(c['M']).encode()).hexdigest()
    ck = f"{c['setting']}|{c['rule']}|{qp}|{c['spec']}|{c['att']}|{bind}"
    if ck in cache:
        return cache[ck]
    sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
    import npcore as N3
    import d3common as D3
    import rules3d_names as RN
    pinned = D3.load_rules_dims(None)
    bk = 'M2' if c['setting'].startswith('M2') else 'W1024'
    if bk not in _NB:
        _NB[bk] = N3.load_bank(HERE / 'inputs' / 'model_W1024' if bk == 'W1024' else HERE / 'vendor' / 'quad3d' / 'inputs' / 'model_M2')
    p, T = _NB[bk]
    T = T[:, :c['r']]
    k = N3.modes(65, c['M'])[0][:c['M']]
    Cs = np.concatenate([np.asarray(c['W'][c['cases'].index(j)]) for j in (0, 1)])       # cases 0 and 1, six outputs each
    vals = []
    for rl in (c['rule'], qp):
        X, w = RN.make(rl, pinned)
        acc = np.zeros((12, len(k)))
        for s in range(0, len(X), 1 << 15):
            G, D = N3.bank_np(p, T, X[s:s + (1 << 15)], deriv=True)
            acc += N3.offmesh_adv_np(G, D, X[s:s + (1 << 15)], w[s:s + (1 << 15)], 65, k, Cs)
        vals.append(acc)
    rho = [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(*vals)]
    job = np.concatenate([np.asarray(c['entry']['rho'], float)[c['cases'].index(j)] for j in (0, 1)])
    out = dict(cases=[0, 1], offline=rho, job=job.tolist(), agree=bool(np.allclose(rho, job, rtol=1e-6, atol=1e-12)))
    cache[ck] = out
    cp.write_text(json.dumps(cache) + '\n')
    return out


def fixgate_offline3d(rules):
    """A2.2: component (b) recomputed with the vendored NumPy bank (npcore) at the pinned 3D fixture states (W512,
    M = 2052, target gl64); cached in checks/fixgate3d-offline.json by rule sha256."""
    sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
    import npcore as N3
    import d3common as D3
    import rules3d_names as RN
    cp = HERE / 'checks' / 'fixgate3d-offline.json'
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    pinned = D3.load_rules_dims(None)
    p, T = N3.load_bank(HERE / 'inputs' / 'model_W1024')
    T = T[:, :512]
    k = N3.modes(65, 2048)[0]
    Cs = np.asarray(np.load(HERE / 'inputs' / 'fixture3d.npz')['W'])

    def N(rl):
        X, w = RN.make(rl, pinned)
        acc = np.zeros((6, len(k)))
        for s in range(0, len(X), 1 << 15):
            G, D = N3.bank_np(p, T, X[s:s + (1 << 15)], deriv=True)
            acc += N3.offmesh_adv_np(G, D, X[s:s + (1 << 15)], w[s:s + (1 << 15)], 65, k, Cs)
        return acc, RN.sha_rule(X, w)
    Nt = None
    for rl in rules:
        X, w = RN.make(rl, pinned)
        sh = RN.sha_rule(X, w)
        if rl in cache and cache[rl]['sha256'] == sh and cache[rl].get('deps') == dep_pins():
            continue
        if Nt is None:
            Nt = N('gl64')[0]
        a, _ = N(rl)
        cache[rl] = dict(sha256=sh, deps=dep_pins(), rho=[float(np.linalg.norm(x - y) / np.linalg.norm(y)) if np.isfinite(x).all() else float('inf')
                                         for x, y in zip(a, Nt)])
        cp.write_text(json.dumps(cache, indent=1) + '\n')
    return cache


def ladder_q2(C, keys, i):
    if i >= len(keys) - 2:
        ro = rho_offline(C[keys[i]], keys[i + 1].split('|')[1]) if (i == len(keys) - 2 and keys[i + 1] in C) else None
        return dict(label='unverified (ladder top)', rho_offline=ro,
                    audit_flag=(ro is not None and not ro['agree']))
    k, kn, knn = keys[i], keys[i + 1], keys[i + 2]
    if kn not in C or C[k]['entry'].get('dist_next') is None:
        return dict(label='unverified (successor invalid)')
    c, cn = C[k], C[kn]
    e, en = c['entry'], cn['entry']
    if cn['cases'] != c['cases'] or cn['M'] != c['M'] or cn['spec'] != c['spec'] or cn['setting'] != c['setting'] \
            or mode_hash(cn) != mode_hash(c):
        return dict(label='blocked (successor identity)')
    if len(cn['valid_cases']) != len(c['cases']) or bool(all(cn['valid_cases'].values())) != bool(all(e['next_valid'])):
        return dict(label='blocked (successor validity inconsistent)')
    off = odist(c['W'], cn['W'], c['setting'], c['cases'])
    agree = bool(np.allclose(off, e['dist_next'], rtol=1e-9, atol=1e-13))
    rho = np.asarray(e['rho'], float).ravel()
    if len(rho) != 6 * len(c['cases']):
        return dict(label='unverified (successor invalid)', note='rho population incomplete')
    rho_f = rho[~np.isnan(rho)]
    chk = None
    if knn in C and en.get('dist_next') is not None:
        cnn = C[knn]
        if cnn['cases'] != c['cases'] or cnn['M'] != c['M'] or cnn['spec'] != c['spec'] or cnn['setting'] != c['setting'] \
                or mode_hash(cnn) != mode_hash(c):
            return dict(label='blocked (successor identity)')
        idx = [c['cases'].index(j) for j in CHK if j in c['cases']]
        dchk = odist(cn['W'][idx], cnn['W'][idx], c['setting'], [c['cases'][j] for j in idx])
        if not np.allclose(dchk, [en['dist_next'][j] for j in idx], rtol=1e-9, atol=1e-13):
            return dict(label='blocked (offline recomputation disagrees)')
        chk = (bool(all(cnn['valid_cases'].get(c['cases'][j], False) for j in idx)), fin_max(dchk))
    rmax = float(rho_f.max()) if rho_f.size else np.nan
    lab = DS.q2_label(True, (bool(all(e['next_valid'])), fin_max(e['dist_next'])), rmax, chk)
    if not agree:
        lab = 'blocked (offline recomputation disagrees)'
    ro = rho_offline(c, kn.split('|')[1])
    if ro is not None and not ro['agree'] and lab == 'pass':
        lab = 'blocked (independent rho disagrees)'
    return dict(label=lab, worst_dist_qp=fin_max(e['dist_next']), rho_max=rmax, check=chk, offline_agrees=agree, rho_offline=ro)


def mode_hash(c):
    return c['entry'].get('mode_hash') if c['kind'] == 'ladder' else str(c['z']['mode_hash'])


def key_of(setting, rule, spec, dt=0.01):
    if spec.startswith('ad|'):
        return f"{setting}|{rule}|ad|{spec.split('|')[1]}|CN|{dt:.8g}"
    return f'{setting}|{rule}|{spec}'


def rounds():
    """Round members: qualify's Q2-3D predicate, with the q / q+ and q+ / q++ distances recomputed offline from the
    archived coefficients (score3 lattice rows) and an independent rho on cases 0-1 where it fits (A6.2)."""
    import c3common as C3
    out = {}
    for k in range(4, 9):
        d = out_dir(f'q3r{k}')
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        W = np.load(d / f"W_{res['mode']}.npz")
        for blk in res['blocks']:
            if blk['kind'] != 'rounds':
                continue
            for key, m in blk['result'].items():
                if key.startswith('control|'):
                    out[f'round{k}|{key}'] = dict(label='control', worst=fin_max(m['dist_qp']), fired=bool(fin_max(m['dist_qp']) > BAR))
                    continue
                pop = (m['cases'] == ALL64 and len(m['dist_qp']) == 64 and len(m['valid']) == 64 and len(m['valid_qp']) == 64
                       and np.shape(m['rho']) == (64, 6) and (not m.get('qpp') or (m.get('qpp_cases') == CHK and len(m['valid_qpp']) == 8
                                                                                   and len(m['dist_qp_qpp']) == 8)))
                if not pop:                                     # A1.5: missing-first precedence
                    out[key] = dict(label='missing (incomplete population)', round=k)
                    continue
                dq = np.asarray(m['dist_qp'], float)
                rho = np.asarray(m['rho'], float).ravel()
                rho = rho[~np.isnan(rho)]
                chk = (all(m['valid_qpp']), fin_max(m['dist_qp_qpp'])) if m.get('qpp') else None
                lab = DS.q2_label(True, (all(m['valid_qp']), fin_max(dq)), float(rho.max()) if rho.size else np.nan, chk)
                st = key.split('|')[0]
                Wq, Wp = W[m['wkeys']['q']], W[m['wkeys']['qp']]
                agree = bool(np.allclose(odist(Wq, Wp, st, ALL64), m['dist_qp'], rtol=1e-9, atol=1e-13))
                if m.get('qpp'):
                    Wpp = W[m['wkeys']['qpp']]
                    agree = agree and bool(np.allclose(odist(Wp[:8], Wpp, st, CHK), m['dist_qp_qpp'], rtol=1e-9, atol=1e-13))
                M_ = int(C3.C.complete_M(65, 4 * int(re.search(r'\d+$', base(st)).group(0))))
                cfake = dict(setting=st, rule=key.split('|')[1], M=M_, r=int(re.search(r'\d+$', base(st)).group(0)),
                             cases=ALL64, W=Wq, entry=dict(rho=m['rho']), spec='|'.join(key.split('|')[2:]), att=f'q3r{k}')
                ro = rho_offline(cfake, m['qp'])
                if not all(m['valid']):
                    lab = 'fail (own runs invalid)'
                elif not pop:
                    lab = 'missing (incomplete population)'            # not a completed test: stays a candidate
                elif lab == 'pass' and not agree:
                    lab = 'blocked (offline recomputation disagrees)'
                elif lab == 'pass' and ro is not None and not ro['agree']:
                    lab = 'blocked (independent rho disagrees)'
                out[key] = dict(label=lab, worst_dist_qp=fin_max(dq), rho_max=float(rho.max()) if rho.size else None, check=chk, round=k)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--after-screen', action='store_true')
    a = ap.parse_args()
    C, jobs = collect()
    man = json.loads((CFG / 'MANIFEST-cells3d.json').read_text())['ladders']
    out = dict(label='E0 R_ST: conditional Richardson estimate (E0 A10; combined A10, A11, retracted A15 carried)',
               configurations={}, ladders=[], width_mstar={}, rounds=rounds(), families={}, B6={}, A3={}, h2q={})
    lab = {}
    for L in man:
        spec = ('ad|' + L['form'].split('|')[0]) if L['kind'] == '3B-ad' else L['spec']
        setting = L['setting']
        keys = [key_of(setting, rl, spec) for rl in L['rungs']]
        labels = []
        for i, k in enumerate(keys):
            if k not in C:
                labels.append('missing')
                continue
            q1 = q1_of(C[k])
            if not q1['population']:
                labels.append('missing')                  # A1.5: an incomplete registered population is missing evidence
                out['configurations'][k] = dict(Q1=q1, note='registered population incomplete')
                continue
            q2 = ladder_q2(C, keys, i)
            qual = bool(all(q1.values()) and q2['label'] == 'pass')
            lab[k] = qual
            e0 = C[k]['e0']
            el = [v for v in e0.values() if v['eligible']] if e0 else []
            out['configurations'][k] = dict(Q1=q1, Q1_pass=all(q1.values()), Q2=q2, qualified=qual, M=C[k]['M'], r=C[k]['r'],
                                            rule=C[k]['rule'], spec=C[k]['spec'], setting=setting,
                                            median=float(np.median([v['e'] for v in el])) if el else None,
                                            worst=float(np.max([v['e'] for v in el])) if el else None)
            labels.append('qualified' if qual else 'not qualified')
        cand = L['candidates']
        ms = DS.mstar(labels[:len(cand)], [m3(r) for r in cand]) if cand else None
        out['ladders'].append(dict(kind=L['kind'], setting=setting, spec=spec, family=L['family'], draw=L['draw'], rungs=L['rungs'],
                                   labels=labels, mstar=ms))
    for st in ('M2_512', 'W256', 'W512', 'W768', 'W1024'):                 # width m*: all three registered specs
        for fam in ('gauss', 'lattice'):
            per = {L['spec']: L for L in out['ladders'] if L['kind'] == '3M' and L['setting'] == st and L['family'] == fam}
            cand = next(iter(per.values()))['rungs'][:-2]
            out['width_mstar'][f'{st}|{fam}'] = dict(per_spec={k: v['mstar'] for k, v in per.items()},
                                                    width=DS.width_mstar({k: v['labels'][:len(cand)] for k, v in per.items()},
                                                                         [m3(r) for r in cand]))

    def e0_of(k):
        if k in C:
            return {int(j): v for j, v in C[k]['e0'].items()} if C[k]['e0'] else None
        for g in ('g1', 'g2', 'g3'):
            p = SPEED3D / 'checks' / f'scores-{g}.json'
            sc = json.loads(p.read_text())['scores'] if p.exists() else {}
            if k in sc:
                return {int(j): v for j, v in sc[k].items()}
        return None

    def arr(kX, kY):
        a_, b_ = e0_of(kX), e0_of(kY)
        ids = sorted(set(a_) & set(b_))
        vx, vy = C[kX]['valid_cases'], C[kY]['valid_cases']
        return (np.array([a_[j]['e'] for j in ids]), np.array([b_[j]['e'] for j in ids]), np.array([a_[j]['eps_ref'] for j in ids]),
                np.array([bool(a_[j]['eligible'] and vx.get(j, False) and vy.get(j, False)) for j in ids]))

    def tr(kX, kY):
        if kX not in C or kY not in C or not C[kX]['e0'] or not C[kY]['e0']:
            return dict(label='inconclusive (missing)', X=kX, Y=kY)
        a_, b_, e_, el = arr(kX, kY)
        l_, p = DS.transition(a_, b_, e_, el, qualified=(lab.get(kX, False), lab.get(kY, False)))
        return dict(label=l_, stats=p, X=kX, Y=kY)
    # ---- B6-3D
    ad = lambda k, f: key_of(f'W512k{k}', 'lat65536', f'ad|{f}')
    fsk = lambda k, rl='lat65536': f'W512|{rl}|fs|CN|0.01|K1' if k == 4 else f'W512k{k}|{rl}|fs|CN|0.01|K1'
    for f in ('LSPG', 'GAL'):
        t_ = {f'{x}->{y}': tr(ad(x, f), ad(y, f)) for x, y in ((1, 2), (2, 3), (3, 4), (4, 8))}
        c_ = tr(ad(1, f), ad(4, f))
        ctl = DS.control_mr(c_['label'], (lab.get(ad(1, f), False), lab.get(ad(4, f), False)))
        labs_raw = [t_['2->3']['label'], t_['3->4']['label']]
        if ctl != 'fired':
            t_ = {x: dict(v, raw_label=v['label'], label=f'blocked (M = r control: {ctl})') for x, v in t_.items()}
        out['B6'][f'W512|{f}-CN adaptive'] = dict(
            transitions=t_, control_1_to_4=c_, control=ctl,
            range_2_4=(DS.range_verdict(labs_raw) if ctl == 'fired' else
                       'test-space sensitivity not demonstrated' if ctl == 'did not fire' else 'blocked (control unavailable)'),
            range_1_8=DS.range_verdict([v['label'] for v in t_.values()]) if ctl == 'fired' else None,
            secondary_at_mstar='identical to primary (the adaptive kappa ladders have the single candidate lat65536)')
    ctl_ad = out['B6']['W512|LSPG-CN adaptive']['control']
    t_ = {f'{x}->{y}': tr(fsk(x), fsk(y)) for x, y in ((2, 3), (3, 4), (4, 8))}
    sec = {}

    def fs_mstar(k):
        s_ = 'W512' if k == 4 else f'W512k{k}'
        ms_ = [L['mstar'] for L in out['ladders'] if L['spec'] == 'fs|CN|0.01|K1' and L['family'] == 'lattice' and L['draw'] == 0
               and L['setting'] == s_ and L['kind'] in ('3M', '3B-fs')]
        return ms_[0]['m'] if ms_ and ms_[0] else None
    for x, y in ((2, 3), (3, 4), (4, 8)):
        mx, my = fs_mstar(x), fs_mstar(y)
        sec[f'{x}->{y}'] = tr(fsk(x, f'lat{mx}'), fsk(y, f'lat{my}')) if (mx and my) else dict(label='inconclusive (no m*)')
    blk = (lambda d_: {k: dict(v, label=v['label'] if ctl_ad == 'fired' else f"blocked (adaptive LSPG control: {ctl_ad}); raw: {v['label']}")
                       for k, v in d_.items()})
    out['B6']['W512|fs-CN (control from the adaptive LSPG form)'] = dict(
        transitions=blk(t_), secondary_at_mstar=blk(sec), control=ctl_ad,
        range_2_4=DS.range_verdict([t_['2->3']['label'], t_['3->4']['label']]) if ctl_ad == 'fired' else 'blocked or not demonstrated')
    for k in (1, 2, 3, 4, 8):
        g, l_ = ad(k, 'GAL'), ad(k, 'LSPG')
        if g in C and l_ in C and C[g]['e0'] and C[l_]['e0']:
            a_, b_, e_, el = arr(g, l_)
            v, p = DS.two_way(a_, b_, e_, el, qualified=(lab.get(g, False), lab.get(l_, False)), names=('GAL', 'LSPG'))
            cg = out['B6']['W512|GAL-CN adaptive']['control']
            if (ctl_ad, cg) != ('fired', 'fired'):
                v = f'blocked (M = r controls LSPG / GAL: {(ctl_ad, cg)}); raw: {v}'
            out['B6'].setdefault('GAL vs LSPG (adaptive CN)', {})[f'kappa{k}'] = dict(label=v, stats=p)
    # ---- A3-3D
    fx = lambda w: 'W512|lat65536|fs|CN|0.01|K1' if w == 'W512' else f'{w}M2052|lat65536|fs|CN|0.01|K1'
    ws = ('W256', 'W512', 'W768', 'W1024')
    t1 = [tr(fx(ws[i]), fx(ws[i + 1])) for i in range(3)]
    t2_ = [tr(f'{ws[i]}|lat65536|fs|CN|0.01|K1', f'{ws[i + 1]}|lat65536|fs|CN|0.01|K1') for i in range(3)]
    out['A3'] = {'fixed (M=2052 frozen list, lat65536)': dict(transitions=t1, range=DS.range_verdict([x['label'] for x in t1])),
                 'fixed m lat65536, nominal M': dict(transitions=t2_, range=DS.range_verdict([x['label'] for x in t2_]))}
    # ---- B3-3D families (W512, kappa 4, fs|CN); five draws for the random families
    fams = {}
    for L in out['ladders']:
        if L['setting'] == 'W512' and L['spec'] == 'fs|CN|0.01|K1' and L['kind'] in ('3M', '3F', '3Fii'):
            fams.setdefault(L['family'], {})[L['draw']] = L
    fres = {}
    for f, dr in fams.items():
        cand = dr[min(dr)]['rungs'][:-2]
        sizes = [m3(r) for r in cand]
        if f in ('gauss', 'smolyak'):
            fres[f] = dict(family=DS.mstar(dr[0]['labels'][:len(cand)], sizes))
        else:
            fres[f] = DS.family_mstar([dr[s_]['labels'][:len(cand)] if s_ in dr else ['missing'] * len(cand) for s_ in range(5)], sizes)
    fg = {}
    for att in ('b3f', 'b3m'):
        if att in jobs:
            for b in jobs[att]['blocks']:
                if b['kind'] == 'fixgate':
                    fg.update(b['result']['rules'])
    off3 = fixgate_offline3d(sorted(r_ for r_ in fg if r_ != 'gl64'))
    est = json.loads((HERE / 'checks' / 'fixture-badlat.json').read_text())
    p3 = next(p for p in est['panels'] if p['panel'].startswith('3d'))
    tgt = 'W512|gl64|fs|CN|0.01|K1'
    gates = {}
    for f, dr in fams.items():
        for s_, L in dr.items():
            fin = L['rungs'][-1]
            kf = key_of('W512', fin, 'fs|CN|0.01|K1')
            avail = kf in C and tgt in C and all(q1_of(C[kf]).values()) and all(q1_of(C[tgt]).values())
            dist = float(odist(C[kf]['W'], C[tgt]['W'], 'W512', C[kf]['cases']).max()) if avail else np.inf
            rb = [0.] * 6 if fin == 'gl64' else fg.get(fin, {}).get('rho_tgt', [])
            agree = fin == 'gl64' or (fin in off3 and len(rb) == 6 and np.allclose(off3[fin]['rho'], rb, rtol=1e-6, atol=1e-12))
            gates[f'{f}|s{s_}'] = dict(rule=fin, gate=DS.cross_family_gate(rb, avail, dist) if agree else 'gate unavailable (offline (b) disagrees)',
                                       worst_dist_to_target=dist)
    bad = fg.get('badlat262144', {})
    agree_est = bool(bad.get('rho_tgt') and np.allclose(bad['rho_tgt'], [s['rho_tgt'] for s in p3['states']], rtol=1e-6))
    ctl_fired = bool(p3['admitted'] and bad.get('flags') is True and agree_est)
    gates['badlat262144 (control)'] = dict(fixture_admitted=p3['admitted'], fired=ctl_fired, agrees_with_establishment=agree_est)
    pe = {}
    nm = sorted(fres)
    for i in range(len(nm)):
        for j in range(i + 1, len(nm)):
            fa, fb = fres[nm[i]]['family'], fres[nm[j]]['family']
            ok = all(v.get('gate') == 'pass' for kk, v in gates.items() if kk.split('|')[0] in (nm[i], nm[j]))
            pe[f'{nm[i]} vs {nm[j]}'] = ('blocked (bad-vector control unavailable or not fired)' if not ctl_fired else
                                         'inconclusive (not reached)' if fa['outcome'].startswith('not reached') or fb['outcome'].startswith('not reached') else
                                         'inconclusive (gate flagged or unavailable)' if not ok else
                                         DS.point_efficiency(fa['m'], fb['m'], names=(nm[i], nm[j])))
    out['families'] = dict(mstar=fres, gates=gates, point_efficiency=pe)
    # ---- the qualified 3D set (deployment-test-space ladders, rounds, qualify passes) and coverage
    qd = json.loads((QUALIFY / 'runs' / 'EXPORT.json').read_text())['threeD']['tested']
    qual = sorted({k for k, v in lab.items() if v and k.split('|')[0] in ('M2_512', 'W256', 'W512', 'W768', 'W1024')}
                  | {k for k, v in out['rounds'].items() if v.get('label') == 'pass'} | {k for k, v in qd.items() if v == 'pass'})
    out['qualified_3d'] = qual
    out['coverage'] = {w: [k for k in qual if k.split('|')[0] == w] for w in ('M2_512', 'W256', 'W512', 'W768', 'W1024')}
    if a.after_screen:
        after_screen(out, C, lab, e0_of, qual)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'rounds3d.json').write_text(json.dumps(dict(labels={k: v['label'] for k, v in out['rounds'].items()
                                                                            if v.get('label') != 'control'}), indent=1) + '\n')
    (HERE / 'checks' / 'analysis-3d.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('3D configs', len(out['configurations']), 'qualified', len(qual))


def cert_of_job(res, d):
    """Timing-job qualification (A1.1, A2.3) from the single-rung certification blocks of a timing job (t3s / t3c):
    every registered case valid, pooled certificate <= 1 %, distance to ADT / tight <= 2.5e-4, anchors valid, tight /
    tighter <= 2.5e-5 on every tighter case (complete inventories)."""
    cert = {}
    for blk, rb in zip(res['config']['blocks'], res['blocks']):
        if blk['kind'] == 'ladder':
            for k, e in rb['result'].items():
                cert[k] = bool(e['cases'] == blk['cases'] and len(e['valid']) == len(blk['cases']) and all(e['valid'])
                               and len(e['adt_valid']) == len(blk['cases']) and e['tighter_cases'] == blk['tighter_cases']
                               and len(e['adt_tighter_valid']) == len(blk['tighter_cases'])
                               and e['nonstationary'] <= .01 * e['steps'] and len(e['dist_adt']) == len(blk['cases'])
                               and fin_max(e['dist_adt']) <= BAR and all(e['adt_valid']) and all(e['adt_tighter_valid'])
                               and len(e['adt_tight_tighter']) == len(blk['tighter_cases']) and fin_max(e['adt_tight_tighter']) <= BAR_TT)
        elif blk['kind'] == 'adaptive':
            for k, e in rb['result'].items():
                z = np.load(d / f"AD_{k.replace('|', '_')}.npz")
                n = len(blk['cases'])
                cert[k] = bool([int(x) for x in z['case_ids']] == blk['cases'] and [int(x) for x in z['tighter_ids']] == blk['tighter_cases']
                               and len(z['valid']) == n and np.all(z['valid']) and len(z['valid_tight']) == n and np.all(z['valid_tight'])
                               and len(z['valid_tighter']) == len(blk['tighter_cases']) and np.all(z['valid_tighter'])
                               and len(z['gcert']) == n and np.all(z['gcert']) and len(z['noncert']) == n
                               and z['cert'].shape == (n, int(round(.25 / float(blk['dt']))))
                               and len(z['dist_tight']) == n and len(z['dist_tight_tighter']) == len(blk['tighter_cases'])
                               and np.sum(z['noncert']) <= .01 * n * z['cert'].shape[1]
                               and fin_max(z['dist_tight']) <= BAR and fin_max(z['dist_tight_tighter']) <= BAR_TT)
    return cert


def after_screen(out, C, lab, e0_of, qual):
    """Timing-job qualification (t3s certification blocks, A1.1), screening costs, H2q, front, C^q_w, open candidates."""
    import qstats as QS
    d = out_dir('t3s')
    res = json.loads((d / 'result.json').read_text())
    pan = res['timing']['screen']
    cert = cert_of_job(res, d)
    ratios = {}
    for x in pan['invocations']:
        ratios.setdefault(x['B'], []).append(x['ratio'])
    CC = {}
    for k in set(qual) | set(COMPARATORS.values()):
        name = k + ('|prod' if k.split('|')[2] == 'ad' else '')
        e = e0_of(k)
        if e is None:
            continue
        el = {j: v for j, v in e.items() if v['eligible']}
        q1_acc = all(q1_of(C[k]).values()) if k in C else True          # rounds / qualify passes: speed3d Q1 (cited)
        cost = 1.0 if k == COMPARATORS['C_old'] else (float(np.median(ratios[name])) if name in ratios else None)
        CC[k] = dict(cost=cost, median=float(np.median([v['e'] for v in el.values()])), worst=float(np.max([v['e'] for v in el.values()])),
                     e={j: v['e'] for j, v in el.items()}, eta={j: v['eps_ref'] for j, v in el.items()},
                     q1=bool(q1_acc and cert.get(k, False)), screen_valid=bool(pan['valid'] and cost is not None))
    qset = {k for k in qual if k in CC and CC[k]['q1']}
    conf = {}
    dc = out_dir('t3c')
    if dc is not None:
        rc = json.loads((dc / 'result.json').read_text())
        ccert = cert_of_job(rc, dc)                         # confirmation-job Q1 of both subjects on cases 0-7 (A1.1)
        for lb, t in rc['timing'].items():
            pair = [t['A']] + sorted({x['B'] for x in t['invocations']})
            nm = lambda x: x[:-5] if x.endswith('|prod') else x
            certified = all(ccert.get(nm(x), False) for x in pair)
            conf[lb.split('|', 1)[1]] = dict(valid=bool(t['valid'] and certified), certified=certified,
                                             median_ratio=float(np.median([x['ratio'] for x in t['invocations']])),
                                             A=t['A'], B=sorted({x['B'] for x in t['invocations']}))
    for cn, ck in COMPARATORS.items():
        sel = DS.h2q(CC, ck, qset)
        pnl = conf.get(cn)
        if pnl and sel.get('winner'):
            wname = sel['winner'] + ('|prod' if sel['winner'].split('|')[2] == 'ad' else '')
            if pnl['A'] != ck or pnl['B'] != [wname]:
                pnl = dict(valid=False, median_ratio=float('nan'))       # confirmation of a different pair: invalid
        r_ = DS.h2q(CC, ck, qset, pnl) if pnl else sel
        r_.update(comparator=ck, comparator_q2=out['configurations'].get(ck, {}).get('Q2', {}).get('label'),
                  comparator_label=('qualified vs qualified comparator' if lab.get(ck) else
                                    'qualified configuration vs the deployed (unqualified) comparator'),
                  scope='full-field 65^3 mesh-path query, H200; not a comparison with the best FOM (K2 already fired)')
        out['h2q'][cn] = r_
    cq = {}
    for w in ('M2_512', 'W768', 'W1024'):                                # C^q_w (descriptive, screening ratio only)
        opts = []
        for fam in ('gauss', 'lattice'):
            m = out['width_mstar'].get(f'{w}|{fam}', {}).get('per_spec', {}).get('fs|BE|0.01|K1')
            if m and m.get('m'):
                rl = f"gl{round(m['m'] ** (1 / 3))}" if fam == 'gauss' else f"lat{m['m']}"
                k = f'{w}|{rl}|fs|BE|0.01|K1'
                if k in CC and CC[k]['cost'] is not None:
                    opts.append((CC[k]['cost'], k))
        if opts:
            ck = min(opts)[1]
            sel = DS.h2q(CC, ck, qset)
            pnl = conf.get(f'Cq|{w}')
            if pnl and sel.get('winner'):
                wname = sel['winner'] + ('|prod' if sel['winner'].split('|')[2] == 'ad' else '')
                if pnl['A'] != ck or pnl['B'] != [wname]:
                    pnl = dict(valid=False, median_ratio=float('nan'))
            res_ = DS.h2q(CC, ck, qset, pnl) if pnl else sel
            if sel.get('winner') and CC[sel['winner']]['cost']:
                res_['S_screen'] = CC[ck]['cost'] / CC[sel['winner']]['cost']
            cq[w] = dict(Cq=ck, result=res_, label='secondary, descriptive: qualified-vs-qualified comparator (DESIGN 6.5)')
    out['h2q_Cq'] = cq
    out['screen'] = {k: dict(cost=v['cost'], median=v['median'], worst=v['worst'], q1=v['q1']) for k, v in CC.items()}
    pts = lambda stat: {k: dict(c=v['cost'], e=v[stat], eps=float(np.median(list(v['eta'].values())))) for k, v in CC.items()
                        if k in qset and v['screen_valid']}
    out['front'] = dict(median=QS.front(pts('median'))[0] if pts('median') else [], worst=QS.front(pts('worst'))[0] if pts('worst') else [],
                        label='qualified subset of the tested candidates (bounded search; never complete)')
    import front3d as F3                                                 # open candidates (qualify A2.1)
    cand = F3.candidates()
    tested = dict(json.loads((QUALIFY / 'runs' / 'EXPORT.json').read_text())['threeD']['tested'])
    tested.update({k: v['label'] for k, v in out['rounds'].items() if v.get('label') != 'control'
                   and not v['label'].startswith(('missing', 'blocked'))})        # missing / blocked evidence is not a test
    passed = [k for k, v in tested.items() if v == 'pass' and k in cand]
    openc = []
    for k in cand:
        if k in tested:
            continue
        dom_both = True                       # open unless dominated on BOTH the median and the worst statistic
        for stat in ('median', 'worst'):
            P = F3.points({x: cand[x] for x in passed + [k]}, stat)
            if not any(QS.dominates(P[x], P[k]) for x in passed):
                dom_both = False
        if not dom_both:
            openc.append(k)
    out['open_candidates'] = dict(count=len(openc), keys=sorted(openc),
                                  note='not dominated (median or worst) by a Q2-passed configuration in the speed3d screening contract')


if __name__ == '__main__':
    main()
