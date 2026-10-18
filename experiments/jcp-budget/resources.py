"""Smoke-derived cost certificates (DESIGN 9, A2.7; speed3d / qualify resources.py pattern). ESTIMATES, not bounds: the
scheduler time limit is the enforced bound. Writes checks/resources-<attempt>.json with projection_hours and
time_limit_hours = ceil_0.25(1.5 x projection); cluster/stage.py refuses a production attempt whose time limit does not
cover 1.5 x the projection or whose configs / runs differ from the certificate.

Measurements (smk1 2D, smk2 3D; GPU type recorded, H200):
  references   seconds per (dim, horizon) case from the fd step metas (all attempts of a case, incl. retries);
               T = 1.0 projected as 2.2 x the T = 0.5 case time (twice the stages, no refinement, 10 % margin);
               refinement risk: every T = 0.5 case charged x 1.3; PS from the smoke's PS step if present, else
               PS_CASE_H per case (E0: 0.17 h per case at T = 0.25 -> 0.4 h at T = 0.5 incl. escalation reserve)
  b2run        seconds per (member, case) of every phase from the B2 / B7 npz 'seconds' fields and result.json
               elapsed; an unmeasured member takes the slowest measured member (conservative); the full factorial's
               38 corners vs the pairwise 22 scale the corner time linearly; 256^2 = 1024^2 time; 4096^2 x MESH4096
               plus the pilot's measured bank build; + per-r setup (bank, LS, tables) measured
  b3run        seconds per (member, case) from the rollout rows (seconds_first, incl. compile) x cases

    python resources.py --attempt b1024 --run b2run:configs/b1024.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MESH4096 = 2.0
PS_CASE_H = 0.4
NC = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}


def out_dir(att):
    return HERE / 'runs' / att / 'archive' / 'output'


def ncases(c):
    return NC[c] if isinstance(c, str) else len(c)


def ref_case_seconds(att='smk1', dim=2):
    secs = []
    for js in out_dir(att).glob(f'ref_refrun_*/{dim}d/*/T*/c*.json'):
        m = json.loads(js.read_text())
        secs.append((m['horizon'], m['seconds']))
    assert secs, f'no reference timing in {att}'
    return max(s for _, s in secs), secs


def b2_rates(att='smk1'):
    d = next(out_dir(att).glob('b2run_*b1024*'))
    res = json.loads((d / 'result.json').read_text())
    ncase = ncases(res['config']['cases'])
    per_member, corners = {}, None
    for f in d.glob('B2_*.npz'):
        z = np.load(f)
        corners = len(z['corners'])
        per_member[f.stem[3:]] = float(np.nansum(z['seconds'])) / len(z['case_ids'])          # s per case (all corners)
    b7 = {mk: v['seconds'] / v['n'] for mk, v in res['b7'].items()}
    total = res['elapsed_seconds']
    accounted = sum(per_member.values()) * ncase + sum(v * ncase for v in b7.values())
    overhead = max(0., total - accounted)                       # setup, stars, checks, controls, B8, compile
    return dict(corner_case=max(per_member.values()), corners=corners, b7_case=max(b7.values()), overhead=overhead,
                ncase=ncase, members=len(per_member))


def smoke_phases(att='smk1'):
    """Per bank prefix r: phase times from the smoke log's progress lines ('STAR r case t', 'B2 member case t',
    'B7 member case t'; t = seconds since the step began): star+setup, corners per member-case, the check / M / control
    phase (between the last B2 line and the first B7 line) per member-check-case, and B7 per member-case."""
    log = '\n'.join(p_.read_text() for p_ in (HERE / 'runs' / att / 'archive' / 'logs').glob('*.out'))
    blk = log.split('STEP b2run configs/smk1_b1024.json')[1].split('STEP b2run')[0]
    import re
    ev = [(m.group(1), m.group(2), m.group(3), float(m.group(4))) for m in
          re.finditer(r'^(STAR|B2|B7) (\S+) (\S+) ([0-9.]+)$', blk, re.M)]
    d = json.loads((next(out_dir(att).glob('b2run_*b1024*')) / 'result.json').read_text())
    ncase, nchk = ncases(d['config']['cases']), len(d['config']['b2'].get('check_cases', 'dev2') == 'dev2' and [0, 1] or [])
    out = {}
    t_prev = 0.
    for r in sorted({int(m['Rp']) for m in d['config']['members']}):
        mks = [mk for mk in d['b2'][f'R{r}']['members']]
        st = [t for k, a, b, t in ev if k == 'STAR' and a == str(r)]
        b2 = [t for k, a, b, t in ev if k == 'B2' and a in mks]
        b7 = [t for k, a, b, t in ev if k == 'B7' and a in mks]
        out[r] = dict(star_setup=st[-1] - t_prev, corner_member_case=(b2[-1] - st[-1]) / (len(mks) * ncase),
                      check_member_case=(b7[0] - b2[-1]) / (len(mks) * max(1, nchk)),
                      b7_member_case=max(d['b7'][mk]['seconds'] / d['b7'][mk]['n'] for mk in mks), members=len(mks))
        t_prev = b7[-1]
    nref = len(d['b8'].get('refs_found', [])) or 1                 # B8 runs only on cases with a reference
    out['b8_member_case'] = max(0., d['elapsed_seconds'] - t_prev) / (max(1, len(d['config']['members'])) * nref)
    return out


def b2_model(cfg):
    """Production b2run projection from the smoke phases: an unmeasured r takes the slowest measured r; the full factorial
    has 38 corners, pairwise 22 (corner time scaled linearly); checks scale with the Q-corner count (17 full / 6
    pairwise) and the check cases; stars and setup scale with the cases; 4096^2 x MESH4096 plus the pilot."""
    ph = smoke_phases()
    meas = {r: v for r, v in ph.items() if isinstance(r, int)}
    slow = max(meas.values(), key=lambda v: v['corner_member_case'])
    n = ncases(cfg['cases'])
    nchk = ncases(cfg['b2'].get('check_cases', 'dev6'))
    full = cfg['b2'].get('design') == 'full'
    cf, qf = (1. if full else 22 / 38), (1. if full else 6 / 17)
    t = 0.
    for r in sorted({int(m['Rp']) for m in cfg['members']}):
        v = meas.get(r, slow)
        k = sum(int(m['Rp']) == r for m in cfg['members'])
        t += v['star_setup'] * n / 2 + k * n * v['corner_member_case'] * cf + k * nchk * v['check_member_case'] * qf
        if 'b7' in cfg:
            t += k * n * v['b7_member_case'] * float(cfg['b7']['horizon']) / .5
    if 'b8' in cfg:
        t += len(cfg['members']) * n * ph['b8_member_case']
    if cfg['mesh'] == 4096:
        t = t * MESH4096 + pilot_seconds()
    return t + 600


def pilot_seconds(att='smk1'):
    d = next(out_dir(att).glob('b2run_*pilot4096*'), None)
    if d is None:
        return 3600.
    return json.loads((d / 'result.json').read_text())['elapsed_seconds']


def fd_worst_extra(case_s, dim, refine, hf):
    """Enumerated worst retry path of ONE case beyond its nominal triple (refrun.solve_case): every base level retried
    (+1), a tight base triple (x 1.5), the refinement level (r = 3.4 x 0.86 of the case time in 2D, 5.1 x 0.86 in 3D)
    retried once (+r), and a tight re-solve of the refined triple (1.5 x (0.98 + r)). Units of the nominal case time."""
    r = ((3.4 if dim == 2 else 5.1) * .86) if refine else 0.
    return (1. + 1.5 + (r + r + 1.5 * (.98 + r) if refine else 0.)) * case_s * hf


def memory_note(runs):
    """Expected device peaks from the smokes (A2.7): reference levels scale with the grid size; ROM jobs with the bank."""
    out = {}
    for r in runs:
        drv, cf = r.split(':', 1)
        cfg = json.loads((HERE / cf).read_text())
        if drv == 'ref/refrun' and cfg['mode'] == 'fd' and int(cfg['dim']) == 2:
            pk = max((a['peak_bytes'] or 0) for js in out_dir('smk1').glob('ref_refrun_*_ref/2d/*/T*/c*.json')
                     for a in json.loads(js.read_text())['attempts'])
            out[r] = dict(measured_peak_GiB_L8192=pk / 2 ** 30, refinement_estimate_GiB=pk / 2 ** 30 * (12288 / 8192) ** 2,
                          device_GiB=138101653504 / 2 ** 30)
        elif drv == 'ref/refrun' and cfg['mode'] == 'fd' and int(cfg['dim']) == 3:
            pk = max((a['peak_bytes'] or 0) for js in out_dir('smk2').glob('ref_refrun_*_ref/3d/*/T*/c*.json')
                     for a in json.loads(js.read_text())['attempts'])
            out[r] = dict(measured_peak_GiB_L512=pk / 2 ** 30, refinement_estimate_GiB=pk / 2 ** 30 * (768 / 512) ** 3,
                          note='cubic scaling of the measured L512 peak to the L768 refinement level (output buffer alone '
                               '37 GiB); E0 refined 0 of 200 3D case-kinds', device_GiB=138101653504 / 2 ** 30)
        elif drv == 'ref/refrun' and cfg['mode'] == 'ps':
            E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
            d_ = int(cfg['dim'])
            sub = {2: '2d/val32', 3: '3d/val64'}[d_]
            pk = max(r_['peak_bytes'] for f in (E0 / 'store' / sub / 'PS').glob('*.json') for r_ in json.loads(f.read_text())['runs']
                     if r_.get('peak_bytes'))
            out[r] = dict(E0_measured_peak_GiB_T025=pk / 2 ** 30, horizon_note='the PS state is the coefficient array; the '
                          'horizon adds only lattice outputs (11 x the lattice)', device_GiB=138101653504 / 2 ** 30)
        elif drv == 'b3run':
            res_ = json.loads((next(out_dir('smk2').glob('b3run_*')) / 'result.json').read_text())
            pk = max(v.get('peak', -1) for v in res_.get('memory', {}).values())
            out[r] = dict(smoke_peak_GiB=pk / 2 ** 30, estimate_GiB=1.25 * pk / 2 ** 30,
                          note='same settings and rule tables as the smoke (W256, W512 with gl56 / gl64); one more member '
                               '(W256 lat8192, smaller tables) and 16 instead of 2 cases (arrays of a few MB)',
                          device_GiB=138101653504 / 2 ** 30)
        elif drv == 'b2run':
            pr = json.loads((next(out_dir('smk1').glob('b2run_*pilot4096*')) / 'result.json').read_text())
            pk = max(v.get('peak', -1) for x in pr['pilot'].values() for v in x.values() if isinstance(v, dict))
            out[r] = dict(pilot4096_peak_GiB=pk / 2 ** 30, device_GiB=138101653504 / 2 ** 30)
    return out


def project(drv, cfg):
    if drv == 'ref/refrun':
        dim = int(cfg['dim'])
        smk = 'smk1' if dim == 2 else 'smk2'
        case_s, _ = ref_case_seconds(smk, dim)
        if cfg['mode'] == 'fd':
            # nominal case time (smoke) x horizon factor, plus a reserve for up to two cases taking the full retry path:
            # a tight re-run of the triple (x 1.5) and the refinement level (12288^2 at dt = 0.05/240: about
            # (12288/8192)^2 x 1.5 = 3.4 x the finest-level time, the finest level being ~86 % of the case; 2D only)
            hf = 1. if cfg['horizon'] <= .5 else 2.2
            return len(cfg['cases']) * case_s * hf + 600                 # the worst-path reserve is added per JOB (main)
        if cfg['mode'] == 'ps':
            # no H200 PS measurement: E0's 0.17 A100-h per case at T = 0.25, x 2 for T = 0.5, x 3 for a stability rerun
            # (worst case), no H200 speed credit (job audit refs2d-1)
            return len(cfg['cases']) * .17 * 3600 * 2 * 3 * (2. if cfg['horizon'] > .5 else 1.) + 900
        return 1800.
    if drv == 'b2run' and 'b2' in cfg:
        return b2_model(cfg)
    if drv == 'b2run':
        r = b2_rates()
        n = ncases(cfg['cases'])
        t = 0.
        if 'b2' in cfg:
            corners = 38 if cfg['b2'].get('design') == 'full' else 22
            t += len(cfg['members']) * n * (r['corner_case'] / r['corners']) * corners
        if 'b7' in cfg:                       # B7 cost scales with the horizon (smoke measured at T = 0.5)
            t += len(cfg['members']) * n * r['b7_case'] * float(cfg['b7']['horizon']) / .5
        t += r['overhead'] * (n / r['ncase']) * (len(cfg['members']) / r['members'])
        if cfg['mesh'] == 4096:
            t = t * MESH4096 + pilot_seconds()
        return t
    if drv == 'b3run':
        d = next(out_dir('smk2').glob('b3run_*'))
        res = json.loads((d / 'result.json').read_text())
        per = sum(r_['seconds_first'] for r_ in res['rows']) / max(1, len(res['config']['case_subset']))
        per *= len(cfg['members']) / len(res['config']['members'])
        return per * len(cfg['case_subset']) + res['elapsed_seconds'] * .2
    raise KeyError(drv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--gpu', default='h200', help='the GPU type the smoke timings were measured on (H200); bound into the certificate')
    a = ap.parse_args()
    assert a.gpu == 'h200', 'timings were measured on H200; another GPU type needs its own smoke'
    tot, det, cfgs = 0., {}, {}
    fd_n, fd_par = 0, None
    for r in a.run:
        drv, cf = r.split(':', 1)
        p = HERE / cf
        cfg = json.loads(p.read_text())
        s = project(drv, cfg)
        det[r] = s / 3600
        tot += s
        cfgs[cf] = hashlib.sha256(p.read_bytes()).hexdigest()
        if drv == 'ref/refrun' and cfg['mode'] == 'fd':
            fd_n += len(cfg['cases'])
            d_ = int(cfg['dim'])
            fd_par = (ref_case_seconds('smk1' if d_ == 2 else 'smk2', d_)[0], d_, bool(cfg.get('allow_refine')),
                      1. if cfg['horizon'] <= .5 else 2.2)
    if fd_par is not None:                       # up to two cases of the JOB take the enumerated worst retry path
        ex = min(fd_n, 2) * fd_worst_extra(*fd_par)
        det['worst_path_reserve'] = ex / 3600
        tot += ex
    proj = tot / 3600
    tl = math.ceil(1.5 * proj * 4) / 4
    mem = memory_note(a.run)
    out = dict(attempt=a.attempt, runs=a.run, configs=cfgs, gpu=a.gpu, projection_hours=proj, memory=mem, time_limit_hours=max(tl, .5), detail=det,
               label='estimate from the smokes (not a bound)')
    dest = HERE / 'checks' / f'resources-{a.attempt}.json'
    dest.write_text(json.dumps(out, indent=1) + '\n')
    print(dest, json.dumps(out))


if __name__ == '__main__':
    main()
