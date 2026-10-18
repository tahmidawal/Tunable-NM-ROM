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
import json
import re
from pathlib import Path

import numpy as np

import dcert as DC
import dstats as DS
import npcheck as NC
import qscore as SC

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'


def out_dir(att, drv, stem=None):
    """Archive of a logical job: the latest collected retry '<job>r<k>' (DESIGN A7.1; the output stem is the config stem
    = the logical job name) whose result is complete, else the original attempt."""
    stem = stem or att
    cands = sorted((p for p in RUNS.glob(f'{att}r*') if re.fullmatch(rf'{re.escape(att)}r\d+', p.name)),
                   key=lambda p: int(p.name[len(att) + 1:]), reverse=True) + [RUNS / att]
    for c in cands:
        for st_, base_ in [(x, y) for y in ('archive', 'slim') for x in (stem, c.name)]:     # slim = offloaded raw (brief 16)
            d = c / base_ / 'output' / f'{drv}_{st_}'
            if (d / 'result.json').exists() and json.loads((d / 'result.json').read_text()).get('complete'):
                return d
    for base_ in ('archive', 'slim'):
        d = RUNS / att / base_ / 'output' / f'{drv}_{stem}'
        if (d / 'result.json').exists():
            return d
    return None


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
        SC.gcert_ok = lambda R_, f, tolf: DC.gcert(R_['raw'], R_['tolratio'], t_prod * np.asarray(R_['scale'], float),
                                                    np.asarray(R_['scale'], float))
    elif form == 'GAL':
        SC.gcert_ok = lambda R_, f, tolf: bool(old_g(R_, f, tolf) and DC.gcert(R_['raw'], R_['tolratio'],
                                                                                t_prod * np.asarray(R_['scale'], float),
                                                                                np.asarray(R_['scale'], float)))
    try:
        yield
    finally:
        SC.certified, SC.gcert_ok = old_c, old_g


DEV6 = [f'dev6|{i}' for i in range(6)]
ALL38 = DEV6 + [f'val32|{i}' for i in range(32)]


def expected_ids(spec):
    """Registered populations (DESIGN 2.4, make_configs arms): all38, dev6, dev2, or an explicit list."""
    if spec == 'all38':
        return list(ALL38)
    if spec == 'dev6':
        return list(DEV6)
    if spec == 'dev2':
        return DEV6[:2]
    return [f'{c}|{i}' for c, i in spec]


def q1(R, rows, cases_spec):
    """Q1 (and in-config offline agreement) of one configuration against its REGISTERED population (cases_spec from
    the config arm; tighter on its dev6 members); the in-file Q2 population is ignored here (ladder Q2 separately)."""
    R2 = dict(R)
    R2['q2_case_ids'] = np.array([], dtype=str)
    cases = expected_ids(cases_spec)
    expect = dict(cases=cases, dev=[c for c in cases if c in DEV6], q2=[], qpp=None, form=str(R['form']), steps=int(R['steps']))
    with _rules(R):
        return SC.qualify(R2, rows, expect)


IDENT = ('Rp', 'M', 'scheme', 'form', 'dt', 'policy', 'mode_hash', 'steps')


def same_config(R, Rn):
    return all(str(R[k]) == str(Rn[k]) for k in IDENT) and [str(x) for x in R['case_ids']] == [str(x) for x in Rn['case_ids']]


def ladder_q2(R, Rn, Rnn, check_ids=None, position='interior'):
    """A1.5 Q2 of rung R from its successor Rn and check Rnn. position: 'top' / 'second' (the top two rungs of a
    ladder: always 'unverified (ladder top)', A1.5 precedence) or 'interior' (a missing successor / check is
    'unverified (successor invalid)' / 'unverified (finer rule unavailable)'). Successor and check must be the same
    numerical configuration and population except the rule (IDENT), else the label is 'blocked (successor identity)'."""
    if position in ('top', 'second'):
        return dict(label='unverified (ladder top)')
    if Rn is None:
        return dict(label='unverified (successor invalid)')
    Rp = int(R['Rp'])
    ids = [str(x) for x in R['case_ids']]
    if not same_config(R, Rn) or (Rnn is not None and not same_config(R, Rnn)):
        return dict(label='blocked (successor identity)')
    d = [NC.fdist(NC.decode257(R['W'][j], Rp), NC.decode257(Rn['W'][j], Rp), SC.n0_of(R, j)) for j in range(len(ids))]
    sv = bool(np.all(Rn['valid']) and np.isfinite(Rn['W']).all())
    rho = np.asarray(R['rho'], float).ravel()
    rho_f = rho[~np.isnan(rho)]
    rho_max = float(np.max(rho_f)) if rho_f.size else float('nan')
    chk = None
    if Rnn is not None:
        dev = DEV6 if check_ids is None else check_ids
        if not all(c in ids for c in dev):
            return dict(label='unverified (finer rule unavailable)')
        jj = [ids.index(c) for c in dev]
        dc = [NC.fdist(NC.decode257(Rn['W'][j], Rp), NC.decode257(Rnn['W'][j], Rp), SC.n0_of(R, j)) for j in jj]
        chk = (bool(all(Rnn['valid'][j] for j in jj) and np.isfinite(Rnn['W'][jj]).all()), float(np.max(dc)) if dc else np.inf)
    if len(rho) != 6 * len(ids):
        return dict(label='unverified (successor invalid)', note='rho population incomplete')
    lab = DS.q2_label(True, (sv, float(np.max(d)) if d else np.inf), rho_max, chk)
    return dict(label=lab, worst_dist_qp=float(np.max(d)), rho_max=rho_max, rho_trivial=int(np.sum(np.isnan(rho))),
                check=chk, successor_valid=sv)


def scores(R):
    return SC.score_config(R, float(R['dt']))
