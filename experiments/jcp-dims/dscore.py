"""jcp-dims offline 2D scoring and qualification (DESIGN 3, A1.5, A2.1, A3.3; CPU only, no JAX). Reuses qualify's
qscore.qualify (offline recomputation of every Q1 distance with the independent NumPy decode, per-case validity, the
dev-case tight/tighter check, G-cert) with two lane-specific certificate rules:

  * square LSPG (M = r, A2.1): certified iff eta_w = ||D R|| / scale <= 1e-7 (the stored 'cert' is eta_w); G-cert by
    exact inversion r_rec = tolratio * (tol_prod * scale + 1e-300) (A3.3);
  * GAL: qualify's eta classification, G-cert additionally by the A3.3 inversion.

Q2 is computed from the ladder (DESIGN 3): the successor run is the next rung's production run of the same cell and
scheme (rollout bar on every case, ρ from the job on every case), the check run is rung + 2 (dev6), labelled by
dstats.q2_label (A1.5 precedence). E0 scoring through qscore.score_config (identity assertions incl. dt).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

import dcert as DC
import dstats as DS
import npcheck as NC
import qscore as SC

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'


def out_dir(att, drv, stem=None):
    d = RUNS / att / 'archive' / 'output' / f'{drv}_{stem or att}'
    return d if (d / 'result.json').exists() else None


def load(d, setting, form, scheme, factor=1):
    p = d / f'R_{setting}_{form}_{scheme}_{float(factor):g}.npz'
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


@contextlib.contextmanager
def _rules(R):
    """Temporarily install the lane's certificate classification / G-cert for this configuration."""
    form, square = str(R['form']), bool(R.get('square', False))
    t_prod = float(np.asarray(R['tol'])[1])
    old_c, old_g = SC.certified, SC.gcert_ok
    if form == 'LSPG' and square:
        SC.certified = lambda c, f: np.isfinite(np.asarray(c, float)) & (np.asarray(c, float) <= DC.ETA_W_BAR)
        SC.gcert_ok = lambda R_, f, tolf: DC.gcert(R_['raw'], R_['tolratio'], t_prod * np.asarray(R_['scale'], float))
    elif form == 'GAL':
        SC.gcert_ok = lambda R_, f, tolf: bool(old_g(R_, f, tolf) and DC.gcert(R_['raw'], R_['tolratio'],
                                                                                t_prod * np.asarray(R_['scale'], float)))
    try:
        yield
    finally:
        SC.certified, SC.gcert_ok = old_c, old_g


def q1(R, rows):
    """Q1 (and in-config offline agreement) of one configuration; the in-file Q2 population is ignored here (ladder Q2
    is computed separately)."""
    R2 = dict(R)
    R2['q2_case_ids'] = np.array([], dtype=str)
    expect = dict(cases=[str(x) for x in R['case_ids']], dev=[str(x) for x in R['dev_case_ids']], q2=[], qpp=None,
                  form=str(R['form']), steps=int(R['steps']))
    with _rules(R):
        return SC.qualify(R2, rows, expect)


def ladder_q2(R, Rn, Rnn, check_ids=None, second_to_top=False):
    """A1.5 Q2 of rung R from its successor Rn and check Rnn (None = missing / ladder top)."""
    if Rn is None:
        return dict(label=DS.q2_label(True, None, np.nan, None, ladder_top=True), top=True)
    Rp = int(R['Rp'])
    ids = [str(x) for x in R['case_ids']]
    assert [str(x) for x in Rn['case_ids']] == ids, 'successor population differs'
    for k in ('Rp', 'M', 'scheme', 'form', 'dt', 'policy', 'mode_hash'):
        assert str(R[k]) == str(Rn[k]), ('successor differs in', k)
    d = [NC.fdist(NC.decode257(R['W'][j], Rp), NC.decode257(Rn['W'][j], Rp), SC.n0_of(R, j)) for j in range(len(ids))]
    sv = bool(np.all(Rn['valid']) and np.isfinite(Rn['W']).all())
    rho = np.asarray(R['rho'], float).ravel()
    rho_f = rho[~np.isnan(rho)]
    rho_max = float(np.max(rho_f)) if rho_f.size else float('nan')
    chk = None
    if Rnn is not None:
        dev = [str(x) for x in R['dev_case_ids']] if check_ids is None else check_ids
        jj = [ids.index(c) for c in dev]
        dc = [NC.fdist(NC.decode257(Rn['W'][j], Rp), NC.decode257(Rnn['W'][j], Rp), SC.n0_of(R, j)) for j in jj]
        chk = (bool(all(Rnn['valid'][j] for j in jj) and np.isfinite(Rnn['W'][jj]).all()), float(np.max(dc)) if dc else np.inf)
    lab = DS.q2_label(True, (sv, float(np.max(d)) if d else np.inf), rho_max, chk)
    if second_to_top and lab != 'fail':
        lab = 'unverified (ladder top)'                 # A1.5: the top two rungs of a ladder are never qualified
    return dict(label=lab, worst_dist_qp=float(np.max(d)), rho_max=rho_max, rho_trivial=int(np.sum(np.isnan(rho))),
                check=chk, successor_valid=sv)


def scores(R):
    return SC.score_config(R, float(R['dt']))
