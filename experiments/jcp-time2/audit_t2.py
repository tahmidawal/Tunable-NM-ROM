"""Independent audit of one pulled jcp-time2 job (DESIGN A3.6 + gates G0, G1a, G1b, G4). NumPy only (no JAX, not the
driver's code path): the bank is evaluated by vendor/quad2d/npbank.py at the 257^2 shared nodes, rotated by T[:, :R'],
and every saved coefficient trajectory is decoded and re-scored against the stored references.

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/audit_t2.py runs/<attempt>/archive [--max-cases N]

Writes checks/audit-<attempt>.json. Acceptance: the unmutated output passes every check; rejection tests (case swap,
time shift, ST/S swap, reference-file corruption, 1 % metric perturbation) must each be rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'vendor/quad2d'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))
import npbank  # noqa: E402

REFSRC = ROOT.parent / '2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output'
LANE_DV = ROOT.parent / '2026-10-01-quadrature-study/experiments/quadrature-study/runs'
CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
ROT = HERE / 'vendor/quad2d/inputs/rotation_R512.npz'
RTOL, ATOL = 1e-9, 1e-13


def params_draw(seed, count):        # engines.params_draw, re-typed (no import of the driver's modules)
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count), r.uniform(.05, .20, count),
                     r.uniform(.5, 2., count), np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def cohort(name):
    if name == 'dev6':
        return np.concatenate((params_draw(7090702, 4), params_draw(911702, 2)))
    if name == 'val32':
        return params_draw(20260927, 32)
    raise ValueError(name)


def initial257(ph):
    x = np.arange(257) / 256
    X, Y = np.meshgrid(x, x, indexing='ij')
    cx, cy, w, a, _ = ph
    u = a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w))
    u[[0, -1], :] = 0.
    u[:, [0, -1]] = 0.
    return u


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Decoder:
    def __init__(self):
        P = npbank.load(CKPT)
        x = np.arange(1, 256) / 256
        X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
        self.G = np.concatenate([npbank.features_grad(P, X[s:s + 8192])[0] for s in range(0, len(X), 8192)])
        self.T = np.load(ROT)['T']

    def fields(self, W):
        """W (..., 6, R') -> (..., 6, 257, 257), zero boundary."""
        Rp = W.shape[-1]
        GT = self.G @ self.T[:, :Rp]
        U = (W @ GT.T).reshape(W.shape[:-1] + (255, 255))
        out = np.zeros(W.shape[:-1] + (257, 257))
        out[..., 1:-1, 1:-1] = U
        return out


def errs(F, R, n0r):
    pe = np.linalg.norm((F - R).reshape(6, -1), axis=1) / n0r
    return pe, float(pe[1:].max())


def close(a, b):
    return abs(a - b) <= ATOL + RTOL * abs(b)


def check_metrics(res, Wz, setting, dec, refs, max_cases, swapST=False, timeshift=False):
    """Recompute e_ST/e_S/e_TX, d_half, anchor_257, prod_vs_tight, s_h for every run of the first max_cases cases;
    returns list of mismatches."""
    rows = {(r['cohort'], r['case'], r['run']): r for r in res['rows'] if r['setting'] == setting}
    runs = list(Wz['runs'])
    cases = [tuple(x.split('|')) for x in Wz['cases']]
    bad = []
    for ci, (coh, c) in enumerate(cases[:max_cases]):
        c = int(c)
        W = Wz['W'][ci]
        if timeshift:
            W = np.roll(W, 1, axis=1)
        if not np.allclose(W[:, 0], Wz['w0'][ci][None], rtol=0, atol=1e-12 * np.abs(Wz['w0'][ci]).max()):
            bad.append((coh, c, 'row0_is_not_w0'))
        F = dec.fields(W)
        ph = cohort(coh)[c]
        n0r = np.linalg.norm(initial257(ph))
        rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
        if swapST:
            rST, rS = rS, rST
        rTX = (16 * rST - rS) / 15
        Fk = {k: F[i] for i, k in enumerate(runs)}
        dist = lambda a, b: float((np.linalg.norm((Fk[a] - Fk[b]).reshape(6, -1), axis=1) / n0r)[1:].max())
        for i, k in enumerate(runs):
            row = rows[(coh, c, k)]
            for tag, R_ in (('ST', rST), ('S', rS), ('TX', rTX)):
                _, e = errs(F[i], R_, n0r)
                if not close(e, row[f'e_{tag}']):
                    bad.append((coh, c, k, tag, e, row[f'e_{tag}']))
            role, form, sc, dt, lev = k.split('|')
            dt = float(dt)
            anc = [x for x in runs if x.startswith('main|GAL|BDF2|') and abs(float(x.split('|')[3]) - .005 / 16) < 1e-12
                   and x.endswith('|tight')]
            if 'anchor_257' in row and not close(dist(k, anc[0]), row['anchor_257']):
                bad.append((coh, c, k, 'anchor_257'))
            half = f'{role}|{form}|{sc}|{dt / 2:.8g}|{lev}'
            if 'd_half' in row and not close(dist(k, half), row['d_half']):
                bad.append((coh, c, k, 'd_half'))
            if 'prod_vs_tight' in row and not close(dist(k, f'{role}|{form}|{sc}|{dt:.8g}|tight'), row['prod_vs_tight']):
                bad.append((coh, c, k, 'prod_vs_tight'))
            if 's_h' in row and not close(dist(k, f'{role}|{form}|{sc}|{dt:.8g}|tighter'), row['s_h']):
                bad.append((coh, c, k, 's_h'))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--max-cases', type=int, default=10 ** 6)
    a = ap.parse_args()
    arc = Path(a.archive)
    res = json.loads((arc / 'output/result.json').read_text())
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out')) if (arc / 'logs').exists() else ''
    att = res['config']['attempt']
    out = dict(attempt=att, job_id=res['job_id'], commit=res['commit'], gpu=res['gpu'], checks={}, info={})
    ck = out['checks']
    ck['complete'] = bool(res.get('complete'))
    ck['backend_gpu'] = res['backend'] == 'gpu' and 'jax_backend=gpu' in logs
    # cohort pins
    for coh, d in res['cohorts'].items():
        ck[f'cohort_values_{coh}'] = bool(np.abs(np.array(d['physical']) - cohort(coh)).max() <= 1e-15)
    man = json.loads((REFSRC / 'result.json').read_text())
    ent = {(x['cohort'], x['case'], x['ref']): x['f257_sha256'] for x in man['cases']}
    for coh, d in res['cohorts'].items():
        ck[f'cohort_hash_matches_reference_manifest_{coh}'] = d['physical_sha256'] == man['cohort_sha256'][coh]
    # reference pins
    refs, refbad = {}, []
    for kk, h in res['references'].items():
        tag, coh, c = kk.split('|')
        c = int(c)
        z = np.load(REFSRC / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
        if not (h == ent[(coh, c, tag)] == sha(z)):
            refbad.append(kk)
        refs[(coh, c, tag)] = z
    ck['references_pinned'] = not refbad
    dec = Decoder()
    rej = {}
    for s in res['config']['settings']:
        Wz = dict(np.load(arc / f'output/W_{s}.npz'))
        bad = check_metrics(res, Wz, s, dec, refs, a.max_cases)
        ck[f'metrics_reconstructed_{s}'] = not bad
        out['info'][f'metric_mismatches_{s}'] = [list(map(str, b)) for b in bad[:20]]
        # rejection tests on the first two cases
        if len(Wz['W']) >= 2:
            Wsw = dict(Wz)
            Wsw['W'] = Wz['W'].copy()
            Wsw['W'][[0, 1]] = Wz['W'][[1, 0]]
            Wsw['w0'] = Wz['w0'].copy()
            Wsw['w0'][[0, 1]] = Wz['w0'][[1, 0]]
            rej[f'{s}_case_swap'] = bool(check_metrics(res, Wsw, s, dec, refs, 2))
        else:
            rej[f'{s}_case_swap'] = None          # needs two cases (not applicable)
        rej[f'{s}_time_shift'] = bool(check_metrics(res, Wz, s, dec, refs, 1, timeshift=True))
        rej[f'{s}_ST_S_swap'] = bool(check_metrics(res, Wz, s, dec, refs, 1, swapST=True))
        res_p = json.loads(json.dumps(res))
        r0 = next(r for r in res_p['rows'] if r['setting'] == s)
        r0['e_ST'] *= 1.01
        rej[f'{s}_metric_perturbed'] = bool(check_metrics(res_p, Wz, s, dec, refs, 1))
        # G1a / G1b / G4 / verification summaries
        g = [x for x in res['g1a'] if x['setting'] == s]
        ck[f'G1a_{s}'] = bool(g) and all(x['W_rel'] <= 1e-8 and x['w0_rel'] <= 1e-12 and x['field_rel'] <= 1e-8 for x in g)
        out['info'][f'G1a_{s}'] = dict(W_rel_max=max(x['W_rel'] for x in g), w0_rel_max=max(x['w0_rel'] for x in g),
                                       field_rel_max=max(x['field_rel'] for x in g),
                                       it_diff=[x['it_vendor'] - x['it_generic'] for x in g])
        lane = LANE_DV / f"dv{res['mesh']}/archive/output/result.json"
        if lane.exists():
            L_ = json.loads(lane.read_text())
            arm = res['config']['rules'][s]['main']['rule']
            lr = {(r['cohort'], r['case']): r for r in L_['rows'] if r['setting'] == s and r['arm'] == arm}
            dd = [max(abs(v['e_ST'] - lr[(v['cohort'], v['case'])]['ref_ST_evolved']),
                      abs(v['e_S'] - lr[(v['cohort'], v['case'])]['ref_S_evolved']))
                  for v in res['vendor_rows'] if v['setting'] == s and (v['cohort'], v['case']) in lr]
            out['info'][f'G1b_{s}'] = dict(cases=len(dd), max_abs_diff=max(dd) if dd else None, tol=1e-6,
                                           passed=bool(dd) and max(dd) <= 1e-6)
        tm = res['timing'].get(s)
        if tm:
            inv = tm['invocations']
            ck[f'G4_timed_outputs_match_{s}'] = all(i['A_matches_accuracy'] and i.get('B_matches_accuracy', True) for i in inv)
            ck[f'G4_no_recompile_{s}'] = tm['cache_before'] == tm['cache_after']
        rows = [r for r in res['rows'] if r['setting'] == s]
        out['info'][f'unverified_{s}'] = sorted({r['run'] for r in rows if not r['verified']})
        out['info'][f'nonfinite_{s}'] = sorted({r['run'] for r in rows if not r['finite']})
    # reference-file corruption must be rejected by the reference pin
    k0 = next(iter(res['references']))
    tag, coh, c = k0.split('|')
    z = np.load(REFSRC / f'ref_{tag}_{coh}_{int(c):03d}.npz')['f257'].copy()
    z[3, 100, 100] += 1e-9
    rej['reference_corrupted'] = sha(z) != ent[(coh, int(c), tag)]
    out['rejections'] = rej
    ck['all_rejections_fire'] = all(v for v in rej.values() if v is not None)
    out['all_pass'] = bool(all(ck.values()))
    (HERE / 'checks' / f'audit-{att}.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    for k, v in ck.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print(json.dumps(out['info'], default=str)[:3000])
    print('AUDIT ALL PASS' if out['all_pass'] else 'AUDIT FAILED')


if __name__ == '__main__':
    main()
