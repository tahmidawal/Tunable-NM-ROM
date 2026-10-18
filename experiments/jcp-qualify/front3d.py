"""3D front from speed3d's audited grid analysis (DESIGN 2.4, 6, A1.1, A2.1) and the Q2-3D job configs.

Candidate universe: speed3d `checks/analysis-grid.json` configurations that are eligible with a valid screening panel;
cost = screening ratio x t_A (speed3d's own plot axis); error = median / worst E0 over the 64 validation cases;
eps = median of the per-case eta. The dominance rule is qstats.dominates (DESIGN 6).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import qstats as QS

SPEED3D = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d')
GAUSS = ['gl12', 'gl16', 'gl20', 'gl24', 'gl32', 'gl40', 'gl48', 'gl56', 'gl64', 'gl80']
LATTICE = ['lat2048', 'lat4096', 'lat8192', 'lat16384', 'lat32768', 'lat65536', 'lat131072']
WINNERS = ['M2_512|gl24|fs|BDF2|0.0125|K1', 'M2_512|gl24|qn|CN|0.016666667|k4']
DESIGN_ROUND1 = ['W256|gl16|fs|BE|0.016666667|K1', 'W256|gl16|fs|BDF2|0.025|K2', 'W256|gl16|fs|CN|0.0125|K1',
                 'W256|gl20|fs|CN|0.0125|K1', 'W256|gl20|fs|CN|0.01|K1', 'W256|lat8192|fs|CN|0.01|K1',
                 'W512|gl24|qn|CN|0.025|k4', 'M2_512|gl24|qn|BDF2|0.025|k4', 'W512|lat16384|qn|CN|0.025|k4',
                 'W512|gl24|qn|CN|0.016666667|k4', 'M2_512|gl24|fs|BDF2|0.0125|K1', 'M2_512|gl24|qn|BDF2|0.016666667|k4',
                 'M2_512|gl24|fs|CN|0.01|K1', 'W512|gl24|fs|CN|0.01|K1', 'W768|lat16384|fs|CN|0.01|K1',
                 'W1024|lat32768|fs|CN|0.01|K1', 'M2_512|gl24|qn|CN|0.016666667|k4']
SETTINGS = {'M2_512': dict(name='M2_512', bank='M2', Rp=512, a=3, rule='gl32'),
            'W256': dict(name='W256', bank='W1024', Rp=256, a=6, rule='gl24'),
            'W512': dict(name='W512', bank='W1024', Rp=512, a=6, rule='lat32768'),
            'W768': dict(name='W768', bank='W1024', Rp=768, a=6, rule='lat32768'),
            'W1024': dict(name='W1024', bank='W1024', Rp=1024, a=6, rule='lat65536')}
CONTROL = dict(key='W512|lat8192|fs|BE|0.01|K1', qp='lat16384', qpp=None)


def successors(rule):
    lad = GAUSS if rule.startswith('gl') else LATTICE
    i = lad.index(rule)
    return (lad[i + 1] if i + 1 < len(lad) else None), (lad[i + 2] if i + 2 < len(lad) else None)


def candidates():
    G = json.loads((SPEED3D / 'checks' / 'analysis-grid.json').read_text())['configs']
    return {k: v for k, v in G.items() if v['eligible'] and v.get('ratio') and v.get('panel_valid')}


def points(C, stat):
    return {k: dict(c=v['ratio'] * v['t_A'], e=v[stat], eps=float(np.median(v['eta']))) for k, v in C.items()}


def fronts(C):
    fm, _ = QS.front(points(C, 'median'))
    fw, _ = QS.front(points(C, 'worst'))
    return fm, fw


def round1():
    C = candidates()
    fm, fw = fronts(C)
    keys = sorted(set(fm) | set(fw), key=lambda k: C[k]['ratio'] * C[k]['t_A'])
    keys += [w for w in WINNERS if w not in keys]
    out = []
    for k in keys:
        rule = k.split('|')[1]
        qp, qpp = successors(rule)
        out.append(dict(key=k, qp=qp, qpp=qpp, front=[x for x, f in (('median', fm), ('worst', fw)) if k in f],
                        winner=k in WINNERS))
    return out


def q3d_config(members, attempt, case_subset=None, control=True, pilot=None):
    sets = {}
    for m in members + ([CONTROL] if control else []):
        sn, rule = m['key'].split('|')[:2]
        s = sets.setdefault(sn, dict(SETTINGS[sn], reduced=[], extra=[]))
        for r in (rule, m['qp'], m.get('qpp')):
            if r and r != s['rule'] and r not in s['extra']:
                s['extra'].append(r)
    cfg = dict(attempt=attempt, driver='q3run', mode='q2', cohort_seed=923801, cohort_count=64, trust_fraction=.05,
               settings=list(sets.values()), members=[dict(key=m['key'], qp=m['qp'], qpp=m['qpp']) for m in members],
               qpp_cases=list(range(8)), timing=dict(cases=4, reps=2, burn=.1, seed=20261018))
    if control:
        cfg['control'] = CONTROL
    if case_subset is not None:
        cfg['case_subset'] = case_subset
        cfg['qpp_cases'] = [c for c in range(8) if c in case_subset]
    if pilot:
        cfg['pilot'] = pilot
    return cfg


def smoke_config():
    """smk3 (DESIGN 8): members 14 and 16 and the under-resolved control on all 64 cases (cheap), memory pilot with
    W1024 lat131072."""
    mem = [m for m in round1() if m['key'] in ('W512|gl24|fs|CN|0.01|K1', 'W1024|lat32768|fs|CN|0.01|K1')]
    return q3d_config(mem, 'smk3', control=True,
                      pilot=[dict(setting='W1024', rule='lat131072'), dict(setting='W1024', rule='lat65536')])
