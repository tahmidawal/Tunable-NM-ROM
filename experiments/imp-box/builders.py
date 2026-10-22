"""imp-box query builders and representation loaders (split out of ladders.py so that the frozen query-defining
source set is small and stable; DESIGN A6/A7). Used by ladders, timing, diag, controls and the test stage."""
from __future__ import annotations

import json

import numpy as np

import podsine as PS
import rom as RM
import spec as SP

RULES = (4, 6, 8, 10, 12)
R_POD_COLS = 384


# ------------------------------------------------------------------------------------------------ spaces -------

def pod_spaces(out, key, rules=RULES + (14,)):
    Phi = np.load(out / f'{key}_Phi.npy')
    kq = json.loads((out / 'pod.json').read_text())[key]['k_q']
    R = min(len(Phi), R_POD_COLS)
    return {q: PS.pod_space(Phi[:R], kq, q, name=key) for q in rules}, Phi, kq


def bank_spaces(out, rules=RULES + (14,)):
    import bank as BK
    b = BK.load(out / 'bank.pkl')
    return {q: BK.bank_space('B', b['params'], b['T'], b['k_q'], q) for q in rules}, b


def space_of(sps, q):
    return sps[12] if q == 'T12' else sps[q]


def form_of(q):
    return 'tensor' if q == 'T12' else 'point'


# ------------------------------------------------------------------------------------------------ builders -----

def build_spec(c):
    xf = c.get('xform', 'fft')                     # A11: transform implementation knob ('fft' | 'dense')
    if c['fam'] == 'spec':
        return SP.make_fom(c['N'], c['nsteps'], cert=c.get('cert', False), xform=xf)
    return SP.make_fom_ad(c['N'], c['tol'], cert=c.get('cert', False), xform=xf)


def build_reduced(sps, Phi=None):
    def b(c):
        sp = space_of(sps, c['q'])
        if c['fam'] == 'trans':
            return PS.make_trans_query(sp, Phi, c['r'], c['nsteps'], form=form_of(c['q']), cert=c.get('cert', False),
                                       perturb_cert=c.get('perturb', 0.0))
        return RM.make_query(sp, c['r'], c['st'], c['nsteps'], form=form_of(c['q']), tol=c.get('tol', 'prod'),
                             cert=c.get('cert', False), perturb_cert=c.get('perturb', 0.0))
    return b


def build_ecsw(sub_spaces):
    def b(c):
        sp = sub_spaces[(c['r'], c['kappa'])]
        return RM.make_query(sp, c['r'], 'gal_etdrk4', c['nsteps'], form='point', cert=c.get('cert', False))
    return b


