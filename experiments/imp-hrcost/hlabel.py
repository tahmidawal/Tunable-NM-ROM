"""imp-hrcost qualification labels (DESIGN 3, A1.3, A2.1, A2.3, A2.4), pure NumPy: used in-job by the smoke controls
and offline by hscore.py, so a control exercises the same label code as the verdicts."""
from __future__ import annotations

import numpy as np

BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, ETA_BAR, NONCONV_FRAC = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-8, .01
GCERT_REL, GCERT_ABS = 1e-6, 1e-13


def rho(a, b):
    """||a - b|| / ||b||; both zero -> nan (trivial, excluded and counted); b = 0 != a -> inf; non-finite -> inf."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('inf')
    nb = float(np.linalg.norm(b))
    if nb == 0.:
        return float('nan') if float(np.linalg.norm(a)) == 0. else float('inf')
    return float(np.linalg.norm(a - b) / nb)


def certified(eta):
    e = np.asarray(eta, float)
    return np.isfinite(e) & (e <= ETA_BAR)


def gcert_ok(eta_cert, eta_rec):
    a, b = np.asarray(eta_cert, float), np.asarray(eta_rec, float)
    return bool(np.all(np.isfinite(a) & np.isfinite(b)) and np.all(np.abs(a - b) <= GCERT_REL * np.abs(b) + GCERT_ABS))


def q1_label(cases, dev):
    """cases: {cid: dict(valid, eta (S,), eta_rec (S,), tight_valid, dist_tight, [tighter_valid, dist_tt])};
    dev: case ids that need the tight/tighter check. Returns dict(label, reasons, demonstrated, nonconv_frac, ...).
    demonstrated = the failure rests on a bar exceedance or an invalid production run (evidence of failure), not only on
    invalid / missing anchors (unresolved evidence; DESIGN A2.3)."""
    reasons, demo = [], False
    if not cases:
        return dict(label='fail', reasons=['empty population'], demonstrated=False, nonconv_frac=float('nan'), steps=0)
    steps = sum(len(v['eta']) for v in cases.values())
    bad = sum(int((~certified(v['eta'])).sum()) for v in cases.values())
    frac = bad / max(steps, 1)
    if not all(v['valid'] for v in cases.values()):
        reasons.append('invalid run')
        demo = True
    if frac > NONCONV_FRAC:
        reasons.append(f'non-converged steps {frac:.4f} > 1%')
        demo = True
    for cid, v in cases.items():
        if not v['tight_valid']:
            reasons.append(f'{cid}: tight anchor invalid')
        elif not (np.isfinite(v['dist_tight']) and v['dist_tight'] <= BAR_Q1):
            reasons.append(f'{cid}: distance to tight {v["dist_tight"]:.3e}')
            demo = True
        if not gcert_ok(v['eta'], v['eta_rec']):
            reasons.append(f'{cid}: G-cert disagreement')
    for cid in dev:
        v = cases.get(cid)
        if v is None or 'tighter_valid' not in v:
            reasons.append(f'{cid}: tighter anchor missing')
        elif not v['tighter_valid']:
            reasons.append(f'{cid}: tighter anchor invalid')
        elif not (np.isfinite(v['dist_tt']) and v['dist_tt'] <= BAR_TT):
            reasons.append(f'{cid}: tight/tighter {v["dist_tt"]:.3e}')
            demo = True
    return dict(label='pass' if not reasons else 'fail', reasons=reasons, demonstrated=bool(reasons and demo),
                nonconv_frac=frac, steps=steps)


def q2_label(own_valid, succ_valid, dist_succ, rho_states, chk_succ_valid, chk_valid, dist_chk, bar_rho=BAR_RHO):
    """Binding Q2 (A2.1, A2.3). own/succ arrays over the 16-case population; rho_states (16, 6) tested-vector rho at the
    arm's own output states; chk_* over dev6 (q+ vs q++). Labels: pass / fail / unverified (...)."""
    out = dict(rollout_max=float('nan'), rho_max=float('nan'), chk_max=float('nan'), trivial=0)
    if not all(own_valid):
        out['label'] = 'unverified (own run invalid)'
        return out
    if not all(succ_valid):
        out['label'] = 'unverified (successor invalid)'
        return out
    if not (all(chk_succ_valid) and all(chk_valid)):
        out['label'] = 'unverified (check invalid)'
        return out
    d = np.asarray(dist_succ, float)
    out['rollout_max'] = float(np.inf if not np.isfinite(d).all() else d.max())
    r = np.asarray(rho_states, float)
    out['trivial'] = int(np.isnan(r).sum())
    if np.isnan(r).all():
        out['label'] = 'unverified (all rho trivial)'
        return out
    rr = r[~np.isnan(r)]
    out['rho_max'] = float(rr.max())
    fail_roll = not (out['rollout_max'] <= BAR_Q2)
    fail_rho = not (out['rho_max'] <= bar_rho)
    out['fail_rollout'], out['fail_rho'] = bool(fail_roll), bool(fail_rho)
    if fail_roll or fail_rho:
        out['label'] = 'fail'
        return out
    c = np.asarray(dist_chk, float)
    out['chk_max'] = float(np.inf if not np.isfinite(c).all() else c.max())
    out['label'] = 'pass' if out['chk_max'] <= BAR_Q2 else 'unverified (finer rule not converged)'
    return out


def fdist(fa, fb, n0):
    """Scoring-lattice relative distance: max over the five evolved outputs of ||fa_k - fb_k|| / n0 (NaN-safe)."""
    d = [float(np.linalg.norm(np.asarray(fa[k]) - np.asarray(fb[k]))) / n0 for k in range(1, 6)]
    return float('inf') if not np.isfinite(d).all() else max(d)
