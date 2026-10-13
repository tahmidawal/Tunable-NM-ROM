"""jcp-combined offline scorer (DESIGN 4 "one-switch rescoring", A1.8, A2.10, A3.5, A4.7, A5.4; code audit 2 items 14-16).

Decodes saved output coefficients W (6, R') through the bank evaluated AT THE REFERENCE'S OWN LATTICE COORDINATES (an
independent path from the job's mesh-row decode) and scores them against an authenticated registry entry (refs.json).
2D FOM archives (fom_f257.npz, 257^2 nodes) are scored too, so a new reference rescoring never mixes references.

    python score.py --refs <name> --run <runs/<attempt>/archive/output> [--check] [--out <json>]

Registry kinds (anything else is rejected):
  2d_f257  references on the 257^2 nodes x = i/256 (boundary zero): files per (cohort, case, tag), manifest result.json
           pinned by sha256 and every f257 file checked against the manifest's per-file sha256; derived tags (TX) are
           declared linear combinations; normaliser = ||u0|| of the physical initial field at the nodes.
  3d_l65   one npz on the 63^3 lattice x = k/64 pinned by sha256; normaliser = ||u0|| of the physical initial field.
Every registry entry pins the bank checkpoint sha256 it may score (checked against the run's result.json) and its output
times. --check (G-score): the in-job errors must be reproduced at EVERY output time for EVERY row (|x - y| <= 1e-13 +
1e-10 |y|); full coverage and finite values are required; mandatory negative controls must each break agreement by > 1e-6
(2D: case swap, output-time shift, ST/S swap, coordinate shift of the pinned dev6/0 field at t = 0.05; 3D: case swap,
output-time shift, one-node lattice shift). gate.status in {unchecked, pass, fail}; exit status 1 on fail.
"""
from __future__ import annotations

import offline  # noqa: F401  (A8.1: CPU, highest, x64 - before any JAX import)

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / 'refs.json').read_text())
KINDS = ('2d_f257', '3d_l65')
REQUIRED = {'2d_f257': ('dir', 'tags', 'manifest', 'manifest_sha256', 'bank_sha256', 'rotation_sha256', 'times', 'nodes',
                         'normaliser'),
            '3d_l65': ('file', 'sha256', 'bank_sha256', 'times', 'nodes', 'normaliser', 'cohort_seed', 'cohort_table_sha256')}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def tol_ok(x, y):
    return bool(np.isfinite(x) and np.isfinite(y) and abs(x - y) <= 1e-13 + 1e-10 * abs(y))


CONTRACT = {'2d_f257': dict(times=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
                           nodes='257^2 nodes x=i/256, y=j/256, i,j=0..256, ij order, boundary rows/cols zero',
                           normaliser='||u0||_2 of the physical initial field (engines.initial(256, case descriptors)) at the nodes'),
            '3d_l65': dict(times=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25], nodes='63^3 lattice x=k/64, k=1..63, ij(k) order (x slowest)',
                           normaliser='||u0||_2 of the physical initial field (common.initial_interior(65, ...)) at the lattice')}


def validate_entry(name):
    """Registry contract (code audit 4 item 14): kind supported; required fields present; the coordinate / time /
    normalisation contract equals the one this scorer implements (any change is rejected, not silently ignored);
    reference files authenticated by sha256."""
    ent = REG[name]
    if ent.get('kind') not in KINDS:
        raise ValueError(f'unsupported registry kind {ent.get("kind")!r}')
    miss = [k for k in REQUIRED[ent['kind']] if k not in ent]
    if miss:
        raise ValueError(f'registry entry {name} lacks {miss}')
    for k, v in CONTRACT[ent['kind']].items():
        if ent[k] != v:
            raise ValueError(f'registry {name}: unsupported {k} {ent[k]!r} (this scorer implements {v!r})')
    if ent['kind'] == '2d_f257':
        assert sha_file(Path(ent['dir']) / ent['manifest']) == ent['manifest_sha256'], 'reference manifest sha256'
    else:
        assert sha_file(ent['file']) == ent['sha256'], '3D reference sha256'
    return ent


# ------------------------------------------------------------------------------------------------ 2D
def bank2d(Rp_max, shift=0):
    import t2core as T2
    Q = T2.load_qcore()
    import jax.numpy as jnp
    mdl = Q.Model(ckpt=T2.CKPT)
    x = (np.arange(1, 256) + shift) / 256
    xy = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    G = np.asarray(mdl.base.at(xy, chunk=8192) @ jnp.asarray(mdl.T[:, :Rp_max]))       # (255^2, R')
    return Q, G


def decode2d(G, W):
    U = np.zeros((W.shape[0], 257, 257))
    U[:, 1:-1, 1:-1] = (W @ G[:, :W.shape[1]].T).reshape(-1, 255, 255)
    return U


_MAN = {}


def load_ref2d(ent, coh, c):
    key = ent['dir']
    if key not in _MAN:
        man = json.loads((Path(ent['dir']) / ent['manifest']).read_text())
        _MAN[key] = {(x['cohort'], x['case'], x['ref']): x for x in man['cases']}
    out = {}
    for tag, fn in ent['tags'].items():
        z = np.load(Path(ent['dir']) / fn.format(cohort=coh, case=c))['f257']
        m = _MAN[key][(coh, c, tag)]
        assert m['accepted'] and sha(z) == m['f257_sha256'], ('reference file sha256', coh, c, tag)
        out[tag] = z
    for tag, comb in ent.get('derived', {}).items():
        out[tag] = sum(w * out[t] for t, w in comb.items())
    return out


def score2d(ent, run, check):
    res = json.loads((run / 'result.json').read_text())
    import t2core as T2
    Q_ = T2.load_qcore()
    assert res['checkpoint_sha256'] == ent['bank_sha256'] == sha_file(T2.CKPT), 'bank checkpoint differs from the registry'
    assert sha_file(Q_.ROTATION) == ent['rotation_sha256'], 'rotation differs from the registry'
    man = json.loads((Path(ent['dir']) / ent['manifest']).read_text())
    for coh, v in res['cohorts'].items():           # descriptors authenticated: hash of the ACTUAL array used for n0
        h = sha(np.asarray(v['physical'], dtype=np.float64))
        assert h == v['physical_sha256'] == man['cohort_sha256'][coh], ('case descriptors differ from the reference cohort', coh)
    cohorts = res['cohorts']
    Rps = [res['setup'][s['name']]['Rp'] for s in res['config']['settings']]
    Q, G = bank2d(max(Rps))
    n0c = {}
    out, gate = {}, dict(status='unchecked', compared=0, expected=0, worst=0., nonfinite=0)
    inrow = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    tags = list(ent['tags']) + list(ent.get('derived', {}))
    for s in res['config']['settings']:
        z = np.load(run / f"W_{s['name']}.npz")
        W, keys, cases = z['W'], [str(k) for k in z['runs']], [str(c) for c in z['cases']]
        for ci, ck in enumerate(cases):
            coh, c = ck.split('|')
            c = int(c)
            if (coh, c) not in n0c:
                n0c[(coh, c)] = float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
            n0 = n0c[(coh, c)]
            refs = load_ref2d(ent, coh, c)
            for ri, k in enumerate(keys):
                U = decode2d(G, W[ci, ri])
                pe = {t: [float(np.linalg.norm(U[j] - refs[t][j])) / n0 for j in range(6)] for t in tags}
                out[f"{s['name']}|{k}|{coh}|{c}"] = {t: dict(per_time=v, worst=max(v[1:])) for t, v in pe.items()}
                if check:
                    r = inrow.get((s['name'], k, coh, c))
                    for t in ('ST', 'S', 'TX'):
                        if t not in pe:
                            continue
                        gate['expected'] += 6
                        if r is None:
                            continue
                        for j in range(6):
                            x, y = pe[t][j], (np.nan if r[f'e_{t}_per_time'][j] is None else r[f'e_{t}_per_time'][j])
                            gate['compared'] += 1
                            if not (np.isfinite(x) and np.isfinite(y)):
                                gate['nonfinite'] += 1     # allowed only for an invalid (unverified) in-job rollout
                                ok_nf = not (np.isfinite(x) or np.isfinite(y)) and not r.get('verified', True)
                                gate['failures'] = gate.get('failures', 0) + (0 if ok_nf else 1)
                                continue
                            gate['worst'] = max(gate['worst'], abs(x - y))
                            gate['failures'] = gate.get('failures', 0) + (0 if tol_ok(x, y) else 1)
    if check:                                        # coverage: configured inventory == result rows == archive entries
        import c2grid as C2G
        cfg_ = res['config']
        inv_runs = [C2G.key(r) for r in C2G.run_list(cfg_['grid'])]
        cases_ = [(coh, str(c)) for coh in cfg_['cohorts'] for c in
                  (cfg_.get('case_subset', {}).get(coh) if cfg_.get('case_subset', {}).get(coh) is not None
                   else range(res['cohorts'][coh]['n']))]
        expected = {(s_['name'], k, coh, c) for s_ in cfg_['settings'] for k in inv_runs for coh, c in cases_}
        arch = set()
        for s in res['config']['settings']:
            z = np.load(run / f"W_{s['name']}.npz")
            arch |= {(s['name'], str(k), *str(c).split('|')) for k in z['runs'] for c in z['cases']}
        rowk = {(r['setting'], r['run'], r['cohort'], str(r['case'])) for r in res['rows']}
        gate['rows_without_archive'] = len(rowk - arch)
        gate['archive_without_rows'] = len(arch - rowk)
        gate['inventory_mismatch'] = len(expected ^ rowk)
        fexp = {(f'fom|{sc}|{f:g}', coh, c) for sc in cfg_['fom']['schemes'] for f in cfg_['fom']['dt_factors']
                if res['fom']['calibration'].get(f'{sc}|{f:g}', {}).get('status') in ('calibrated', 'fallback_tight')
                for coh, c in cases_}
        frk = {(r['run'], r['cohort'], str(r['case'])) for r in res.get('fom', {}).get('rows', [])}
        gate['fom_inventory_mismatch'] = len(fexp ^ frk)
    # FOM archive (same reference, same nodes); required whenever the result has FOM rows
    fom = {}
    fp = run / 'fom_f257.npz'
    frow = {(r['run'], r['cohort'], r['case']): r for r in res.get('fom', {}).get('rows', [])}
    if frow and not fp.exists():
        raise SystemExit('FOM rows without fom_f257.npz')
    if fp.exists():
        Fz = np.load(fp)
        assert np.allclose(Fz['__coords_x'], np.arange(257) / 256) and np.allclose(Fz['__times'], np.arange(6) * .05)
        fkeys = {tuple(k.replace('__', '|').rsplit('|', 2)) for k in Fz.files if not k.startswith('__')}
        gate['fom_archive_mismatch'] = len(fkeys ^ {(r['run'], r['cohort'], str(r['case'])) for r in res.get('fom', {}).get('rows', [])})
        for k in Fz.files:
            if k.startswith('__'):
                continue
            kk, coh, c = k.replace('__', '|').rsplit('|', 2)
            c = int(c)
            refs = load_ref2d(ent, coh, c)
            n0 = n0c.get((coh, c)) or float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
            pe = {t: [float(np.linalg.norm(Fz[k][j] - refs[t][j])) / n0 for j in range(6)] for t in tags}
            fom[f'{kk}|{coh}|{c}'] = {t: dict(per_time=v, worst=max(v[1:])) for t, v in pe.items()}
            if check:
                r = frow.get((kk, coh, c))
                for t in ('ST', 'S', 'TX'):
                    gate['expected'] += 6
                    if r is None:
                        continue
                    for j in range(6):
                        x, y = pe[t][j], (np.nan if r[f'e_{t}_per_time'][j] is None else r[f'e_{t}_per_time'][j])
                        gate['compared'] += 1
                        gate['worst'] = max(gate['worst'], abs(x - y)) if np.isfinite(x) and np.isfinite(y) else gate['worst']
                        gate['failures'] = gate.get('failures', 0) + (0 if tol_ok(x, y) else 1)
    ctrl = {}
    if check:
        s0 = res['config']['settings'][0]['name']
        z = np.load(run / f'W_{s0}.npz')
        W, keys, cases = z['W'], [str(k) for k in z['runs']], [str(c) for c in z['cases']]
        k0 = keys.index('LSPG|BE|0.005|prod')
        ci0 = cases.index('dev6|0')
        r = inrow[(s0, keys[k0], 'dev6', 0)]
        if r.get('e_ST') is None:
            raise SystemExit('G-score controls: the pinned dev6/0 baseline row is invalid (non-finite)')
        n0 = n0c[('dev6', 0)]
        refs = load_ref2d(ent, 'dev6', 0)
        U = decode2d(G, W[ci0, k0])
        e = lambda U_, R_: max(float(np.linalg.norm(U_[j] - R_[j])) / n0 for j in range(1, 6))
        other = next((i for i in range(len(cases)) if i != ci0), None)
        ctrl['case_swap'] = abs(e(decode2d(G, W[other, k0]), refs['ST']) - r['e_ST']) if other is not None else None
        ctrl['time_shift'] = abs(e(np.roll(U, 1, 0), refs['ST']) - r['e_ST'])
        ctrl['ST_S_swap'] = abs(e(U, refs['S']) - r['e_ST'])
        _, Gs = bank2d(W.shape[-1], shift=1)
        ctrl['coordinate_shift_field_t005'] = float(np.linalg.norm(decode2d(Gs, W[ci0, k0])[1] - U[1])) / n0
        ctrl['all_break'] = bool(other is not None and all(v is not None and v > 1e-6 for v in ctrl.values()))
        ok = (gate['compared'] == gate['expected'] and gate.get('failures', 0) == 0 and ctrl['all_break'] and
              gate['rows_without_archive'] == 0 and gate['archive_without_rows'] == 0 and gate['inventory_mismatch'] == 0
              and gate['fom_inventory_mismatch'] == 0 and gate.get('fom_archive_mismatch', 0) == 0)
        gate['status'] = 'pass' if ok else 'fail'
    return out, gate, ctrl, fom


# ------------------------------------------------------------------------------------------------ 3D
def score3d(ent, run, check):
    import c3common as C3
    res = json.loads((run / 'result.json').read_text())
    z = np.load(ent['file'])
    cfg = res['config']
    tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
    assert cfg['cohort_seed'] == ent['cohort_seed'] and tab['sha256'] == ent['cohort_table_sha256'], 'cohort table differs'
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    i65 = C3.lattice65_index(65)
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65])) for j in cases}
    Wz = np.load(run / f"W_{res['mode']}.npz")
    R = {j: np.asarray(z[f'c{j}']) for j in cases}
    out, gate = {}, dict(status='unchecked', compared=0, expected=0, worst=0., nonfinite=0, failures=0)
    inrow = {(r['setting'], r['rule'], r['spec'], r['case']): r for r in res['rows']}
    G, banks = {}, {}
    for s in cfg['settings']:
        bk = C3.load_bank(s['bank'])
        assert bk['sha256'] == ent['bank_sha256'].get(s['bank']), 'bank not pinned by the registry entry'
        banks[s['name']] = bk
        G[s['name']] = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
    for key in Wz.files:
        if key.startswith(('w0|', 'cases|')):
            continue
        sname, rule, spec = key.split('|', 2)
        A = Wz[key]
        ids = [int(x) for x in Wz[f'cases|{key}']] if f'cases|{key}' in Wz.files else list(cases)
        assert len(ids) == A.shape[0], ('case ids do not match the trajectory array', key)
        for ci, j in enumerate(ids):
            F = A[ci] @ G[sname]
            pe = [float(np.linalg.norm(F[k] - R[j][k])) / n0[j] for k in range(6)]
            out[f'{key}|{j}'] = dict(ref=dict(per_time=pe, worst=max(pe[1:])))
            if check:
                gate['expected'] += 6
                r = inrow.get((sname, rule, spec, j))
                if r is None:
                    continue
                for k in range(6):
                    x, y = pe[k], (np.nan if r['e_ref_per_time'][k] is None else r['e_ref_per_time'][k])
                    gate['compared'] += 1
                    if not (np.isfinite(x) and np.isfinite(y)):
                        gate['nonfinite'] += 1           # allowed only for an invalid in-job rollout
                        gate['failures'] += 0 if (not (np.isfinite(x) or np.isfinite(y)) and not r.get('valid', True)) else 1
                        continue
                    gate['worst'] = max(gate['worst'], abs(x - y))
                    gate['failures'] += 0 if tol_ok(x, y) else 1
    ctrl = {}
    if check:
        key = next(k for k in Wz.files if k.endswith('|fs|BE|0.01|K1') or k.endswith('vendor|BE|0.01|K1'))
        sname = key.split('|')[0]
        A = Wz[key]
        r0 = inrow[(sname, key.split('|')[1], key.split('|', 2)[2], cases[0])]
        if r0.get('e_ref') is None:
            raise SystemExit('G-score controls: the baseline row of the first case is invalid (non-finite)')
        e = lambda F_, j: max(float(np.linalg.norm(F_[k] - R[j][k])) / n0[j] for k in range(1, 6))
        F0 = A[0] @ G[sname]
        ctrl['case_swap'] = abs(e(A[1] @ G[sname], cases[0]) - r0['e_ref']) if len(cases) > 1 else None
        ctrl['time_shift'] = abs(e(np.roll(F0, 1, 0), cases[0]) - r0['e_ref'])
        bk = banks[sname]
        Rp = A.shape[-1]
        Gs = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :Rp], X65 + 1. / 64))
        ctrl['lattice_shift_field_t005'] = float(np.linalg.norm((A[0][1] @ Gs) - F0[1])) / n0[cases[0]]
        ctrl['all_break'] = bool(all(v is not None and v > 1e-6 for v in ctrl.values()))
        arch = set()
        for key in Wz.files:
            if key.startswith(('w0|', 'cases|')):
                continue
            ids = [int(x) for x in Wz[f'cases|{key}']] if f'cases|{key}' in Wz.files else list(cases)
            arch |= {(*key.split('|', 2), j) for j in ids}
        rowk = {(r['setting'], r['rule'], r['spec'], r['case']) for r in res['rows']}
        gate.update(rows_without_archive=len(rowk - arch), archive_without_rows=len(arch - rowk))
        import c3run as C3R                       # required inventory from the driver plan (codex-code-6 item 13)
        inv = C3R.inventory(cfg)
        exp = set()
        for s_ in cfg['settings']:
            for item in inv['settings'][s_['name']]['runs']:
                rule, spk = (item if isinstance(item, (list, tuple)) else (s_.get('rule'), item))
                if 'conditional' in rule:
                    rule = rule.replace(' (conditional)', '')
                    if not any(k_.startswith(f"{s_['name']}|{rule}|") for k_ in Wz.files):
                        continue                  # the conditional successor was not triggered
                exp |= {(s_['name'], rule, spk, j) for j in (cases if res['mode'] != 'bud' else
                                                             [int(x) for x in Wz.get(f"cases|{s_['name']}|{rule}|{spk}", [])])}
        gate['inventory_mismatch'] = len(exp - rowk)
        ok = (gate['compared'] == gate['expected'] and gate['failures'] == 0 and ctrl['all_break'] and
              gate['rows_without_archive'] == 0 and gate['archive_without_rows'] == 0 and gate['inventory_mismatch'] == 0)
        gate['status'] = 'pass' if ok else 'fail'
    return out, gate, ctrl, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refs', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    ent = validate_entry(a.refs)
    run = Path(a.run)
    fn = score2d if ent['kind'] == '2d_f257' else score3d
    out, gate, ctrl, fom = fn(ent, run, a.check)
    res = json.loads((run / 'result.json').read_text())
    rep = dict(refs=a.refs, registry_entry=ent, run=str(run), result_sha256=sha_file(run / 'result.json'),
               smoke=bool(res['config'].get('smoke')), gate=gate, controls=ctrl, scores=out, fom_scores=fom)
    dest = Path(a.out or HERE / 'checks' / f"scores-{run.parents[1].name}-{a.refs}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep) + '\n')
    print(dest, 'gate', gate, 'controls', ctrl)
    if gate['status'] == 'fail':
        sys.exit(1)


if __name__ == '__main__':
    main()
