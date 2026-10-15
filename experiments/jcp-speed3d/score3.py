"""jcp-speed3d offline scoring (CPU, offline.py first): every saved ROM coefficient trajectory of a pulled job against the
E0 second-order reference (primary; refstore read-only from the E0 lane, store version pinned, field 'finest') and the
first-order 3D reference (513 nodes, BE dt0/4; PROVISIONAL, continuity; cohort 923801 only).

Metric (both): e(t_k) = ||F(t_k) - R(t_k)|| / ||R(t_0)|| on the 63^3 lattice, e = max_{k>=1}. F = W G_lat^T with G_lat the
bank rows at the lattice points (features at x = k/64 times the rotation's first R' columns, as Step 2's score3d).
Controls K4: case swap (case 0's field against case 1's reference) and time shift (outputs rolled by one) must change
the score by > 1e-6. Gate: the job's in-job lattice distances to ADT are recomputed here for the first key of each tag.

    python score3.py --run runs/<attempt>/archive/output [--out checks/scores-<attempt>.json]
"""
from __future__ import annotations

import offline  # noqa: F401  (CPU, highest, x64)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E0DIR = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
REF1 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/'
            'quadrature-burgers3d/runs/ref1/code/output/ref_923801.npz')
REF1_SHA = '36e84e3dbac2f5cf1b3350c311c1f16d17ecfcd4dc217ded836ba30613c6cc8b'
STORE_VERSION = '3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
STATUS_SHA = 'af7d824e94b42afb83753d6b11a0c4d71407cff0923ebee9c245eb8d51f81005'
COHORT = {923801: 'val64', 923901: 'test32'}
BANKS = {'M2_512': ('M2', 512), 'W256': ('W1024', 256), 'W512': ('W1024', 512), 'W768': ('W1024', 768), 'W1024': ('W1024', 1024)}


def refstore():
    if str(E0DIR) not in sys.path:
        sys.path.insert(0, str(E0DIR))
    import refstore as RS
    assert Path(RS.__file__).resolve().parent == E0DIR
    assert RS.store_version() == STORE_VERSION
    import hashlib
    st = hashlib.sha256((E0DIR / 'store' / 'STATUS.json').read_bytes()).hexdigest()
    assert st == STATUS_SHA, 'E0 STATUS.json differs from the pin'
    return RS


class Refs:
    def __init__(self, seed, tab):
        import c3common as C3
        self.RS, self.seed, self.coh = refstore(), seed, COHORT[seed]
        self.tab, self.C3 = tab, C3
        self.e0, self.st = {}, {}
        self.r1 = None
        if seed == 923801:
            assert C3.sha_file(REF1) == REF1_SHA
            self.r1 = np.load(REF1)

    def get(self, j):
        if j not in self.e0:
            R, meta = self.RS.load(3, self.coh, int(j), 'ST', field='finest')
            u0 = np.asarray(self.C3.C.initial_interior(65, self.tab, j))
            assert abs(meta['judge']['n0'] / np.linalg.norm(u0) - 1) <= 1e-12
            assert float(np.abs(np.asarray(R[0]) - u0).max()) <= 1e-12 * float(np.abs(u0).max())
            st = self.RS.reference_status(3, self.coh, int(j), 'ST')
            self.e0[j] = np.asarray(R)
            self.st[j] = dict(eps_ref=st['eps_ref'], eligible=bool(st['eligible']))
        return self.e0[j]


def err(F, R):
    n0 = np.linalg.norm(R[0])
    e = [float(np.linalg.norm(F[k] - R[k]) / n0) for k in range(6)]
    return e, max(e[1:])


_G = {}


def glat(setting):
    if setting not in _G:
        import c3common as C3
        bname, Rp = BANKS[setting]
        bk = C3.load_bank(bname)
        X = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
        _G[setting] = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :Rp], X))     # (R', 63^3)
    return _G[setting]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    import c3common as C3
    rp = Path(a.run)
    res = json.loads((rp / 'result.json').read_text())
    cfg = res['config']
    seed = int(cfg['cohort_seed'])
    tab = C3.C.table(seed, int(cfg['cohort_count']))
    refs = Refs(seed, tab)
    Wz = np.load(rp / f"W_{res['mode']}.npz")
    out = dict(run=str(rp), job_id=res.get('job_id'), seed=seed, cohort=refs.coh, store_version=STORE_VERSION, status_sha=STATUS_SHA,
               label=('E0 second-order reference R_ST (conditional Richardson estimate, not a rigorous bound; its '
                      'cross-check rule was amended after data, E0 A10); first-order scores PROVISIONAL'),
               scores={}, controls={}, gates={})
    keys = [k for k in Wz.files if not k.startswith(('cases|', 'wall|', 'fomfield|'))]
    for key in keys:
        setting = key.split('|')[0]
        G = glat(setting)
        ids = [int(x) for x in Wz[f'cases|{key}']]
        A = Wz[key]
        rec = {}
        for ci, j in enumerate(ids):
            R = refs.get(j)
            F = A[ci] @ G
            e_t, e = err(F, R)
            r = dict(e_t=e_t, e=e, eps_ref=refs.st[j]['eps_ref'], eligible=refs.st[j]['eligible'])
            if refs.r1 is not None:
                e1t, e1 = err(F, np.asarray(refs.r1[f'c{j}']))
                r['first_order'] = dict(e_t=e1t, e=e1)
            rec[j] = r
        out['scores'][key] = rec
        print(key, 'median e', np.median([v['e'] for v in rec.values()]), flush=True)
    # K4 controls on the first key with >= 2 cases
    k0 = next(k for k in keys if len(Wz[f'cases|{k}']) >= 2)
    s0 = k0.split('|')[0]
    ids = [int(x) for x in Wz[f'cases|{k0}']]
    G = glat(s0)
    ca, cb = (0, 1) if {0, 1} <= set(ids) else (ids[0], ids[1])     # A1.10: case 0's field vs case 1's reference
    F0 = Wz[k0][ids.index(ca)] @ G
    e00 = err(F0, refs.get(ca))[1]
    out['controls']['K4_cases'] = [ca, cb]
    out['controls']['K4_case_swap'] = abs(err(F0, refs.get(cb))[1] - e00)
    out['controls']['K4_time_shift'] = abs(err(np.roll(F0, 1, 0), refs.get(ids[0]))[1] - e00)
    out['controls']['K4_fires'] = bool(out['controls']['K4_case_swap'] > 1e-6 and out['controls']['K4_time_shift'] > 1e-6)
    # gate: in-job ADT distances reproduced for one spec per grid rule
    if res['mode'] == 'grid':
        worst = 0.
        n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in res['cohort']['cases']}
        for sn, sv in res['settings'].items():
            for rule, rr in sv['rules'].items():
                key = next(iter(rr['dist_to_adt']))
                parts = key.split('|')
                ak = f"ad|LSPG|{parts[1]}|{parts[2]}|tight"
                Wa, Wb = Wz[f'{sn}|{rule}|{key}'], Wz[f'{sn}|{rule}|{ak}']
                ids = [int(x) for x in Wz[f'cases|{sn}|{rule}|{key}']]
                G = glat(sn)
                d = np.array([np.linalg.norm((Wa[i] - Wb[i]) @ G, axis=1) / n0[j] for i, j in enumerate(ids)])
                worst = max(worst, float(np.abs(d - np.asarray(rr['dist_to_adt'][key])).max()))
        out['gates']['dist_recomputed_worst_abs'] = worst
        out['gates']['dist_ok'] = worst <= 1e-10
    dest = Path(a.out or HERE / 'checks' / f"scores-{cfg['attempt']}.json")
    dest.write_text(json.dumps(out, default=float) + '\n')
    print(dest, out['controls'], out['gates'])


if __name__ == '__main__':
    main()
