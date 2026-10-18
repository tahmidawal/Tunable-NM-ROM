"""jcp-fom2d C2 analysis (DESIGN §5 as amended by A1.3, A1.4, A2.1–A2.3). CPU only.

    python c2analyze.py --c2 runs/<a>/archive/output/c2time_<cfg> [--c2 ...] --plan checks/c2plan.json --out checks/c2-analysis.json

`c2plan.json` (written by make_c2configs.py) lists, per mesh, the targets (key = <identity>@<mesh>|<H|X>), their groups
(family / union x criterion) and the K2 universe. Per stage-2 data only:
  r_{B,i}  = median over B's valid repetitions on case i of t_B / (t_A1 + t_A2)/2   (case valid: >= 2 valid repetitions)
  cost(B)  = median over the common-valid cases I of r_{B,i};  S(X vs Y) = cost(Y) / cost(X)
  interval = 10,000 bootstrap draws of I (same IDs for X and Y), percentile 95 %; |I| < 5 -> 'invalid timing'
A vetoed subject (identity failure anywhere) voids every claim using it; a vetoed or failed A voids the whole panel.
Verdicts: ROM faster iff lower >= 1; FOM faster iff upper < 1; else unresolved; '>= 2x' pass iff lower >= 2, fail iff
upper < 2. Crossover brackets per identity over the meshes where it is qualified, never interpolated. Every number is
conditional on the dev6 timing workload; every matched speed-up is an upper bound on the claimed ROM speed-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NBOOT = 10000
MIN_CASES = 5


def per_case(stage2, lb):
    by = {}
    for x in stage2:
        if x['B'] == lb:
            by.setdefault(x['case'], []).append(x)
    out = {}
    for c, xs in by.items():
        v = [x['ratio'] for x in xs if x['valid']]
        out[c] = float(np.median(v)) if len(v) >= 2 else None
    return out


def speedup(rom, fom, seed=0):
    """S = cost(FOM)/cost(ROM) on the common valid cases, with the paired case bootstrap."""
    I = sorted(c for c in rom if rom[c] is not None and fom.get(c) is not None)
    if len(I) < MIN_CASES:
        return dict(status='invalid timing', n=len(I))
    r = np.array([rom[c] for c in I])
    f = np.array([fom[c] for c in I])
    S = float(np.median(f) / np.median(r))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(I), size=(NBOOT, len(I)))
    bs = np.median(f[idx], axis=1) / np.median(r[idx], axis=1)
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    return dict(status='ok', n=len(I), cases=I, S=S, lo=lo, hi=hi, cost_rom=float(np.median(r)), cost_fom=float(np.median(f)),
                faster=('ROM faster' if lo >= 1 else ('FOM faster' if hi < 1 else 'unresolved')),
                two_x=('pass' if lo >= 2 else ('fail' if hi < 2 else 'inconclusive')))


def bracket(seq):
    """seq: [(mesh, class)] in mesh order -> a bracket statement between actually run meshes (no interpolation)."""
    cls = [c for _, c in seq]
    if not seq:
        return 'no qualified mesh'
    if all(c == 'ROM faster' for c in cls):
        return f'ROM faster at every run mesh ({", ".join(str(m) for m, _ in seq)})'
    if all(c == 'FOM faster' for c in cls):
        return f'FOM faster at every run mesh; crossover not reached by {seq[-1][0]}^2'
    parts = '; '.join(f'{m}^2: {c}' for m, c in seq)
    firsts = [i for i, c in enumerate(cls) if c == 'ROM faster']
    rev = any(cls[i] == 'ROM faster' and cls[j] == 'FOM faster' for i in range(len(cls)) for j in range(i + 1, len(cls)))
    if rev:
        return f'reversal ({parts})'
    if firsts and all(c == 'FOM faster' for c in cls[:firsts[0]]) and firsts[0] > 0:
        return f'crossover between {seq[firsts[0] - 1][0]}^2 (FOM faster) and {seq[firsts[0]][0]}^2 (ROM faster) ({parts})'
    return f'unresolved endpoints ({parts})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c2', action='append', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    jobs = {}
    for p in a.c2:
        r = json.loads((Path(p) / 'result.json').read_text())
        assert r.get('complete') is True, p
        jobs[int(r['mesh'])] = r
    gpus = {r['gpu'] for r in jobs.values()}
    res = dict(gpus=sorted(gpus), one_gpu_type=len(gpus) == 1, meshes={}, cells={}, hcomp={}, k2={}, crossover={},
               label='conditional on the dev6 timing workload; prepared-input latencies; every matched speed-up is an upper '
                     'bound on the claimed ROM speed-up (finite ladder, screening-selected comparator)')
    for N, r in sorted(jobs.items()):
        A = r['config']['A']['label']
        veto = set(r.get('veto', []))
        ctl = r.get('controls', {})
        panel_ok = bool(A not in veto and r['gates'].get('G-RR') and r['gates'].get('G-FR')
                        and ctl.get('T-dbl', {}).get('fires') and ctl.get('T-fault', {}).get('fires')
                        and ctl.get('T-id', {}).get('fires'))
        idf = ctl.get('T-id', {}).get('row', {}).get('B')
        veto_claims = veto - ({idf} if idf else set())          # the T-id control subject is vetoed by construction
        res['meshes'][N] = dict(job=r.get('job_id'), gpu=r['gpu'], panel_ok=panel_ok, veto=sorted(veto), gates=r['gates'],
                                controls={k: v.get('fires') for k, v in ctl.items()},
                                frozen_selection_sha256=r.get('frozen_selection_sha256'),
                                stage0={k: v['cost'] for k, v in r['stage0'].items()},
                                prep=r.get('prep'))
        st2 = r['stage2']
        for tk, t in plan['meshes'][str(N)]['targets'].items():
            rom = per_case(st2, t['rom_label'])
            for gid, g in t['groups'].items():
                sel = r['selection'].get(gid, {}).get('selected')
                cell = dict(target=tk, group=gid, family=g['family'], criterion=g['criterion'], selected=sel,
                            labels=g.get('labels', {}))
                if not panel_ok:
                    cell.update(status='panel invalid')
                elif sel is None:
                    cell.update(status=r['selection'].get(gid, {}).get('status', 'no selection (invalid timing or no candidate)'))
                elif t['rom_label'] in veto_claims or sel in veto_claims:
                    cell.update(status='vetoed (identity)')
                else:
                    cell.update(speedup(rom, per_case(st2, sel)), selected_labels=g.get('candidate_labels', {}).get(sel, {}))
                res['cells'][f'{N}|{gid}'] = cell
    # H-comp at 4096 and 1024 (H populations, joint criterion), per family and union
    for N in (4096, 1024):
        if N not in jobs:
            continue
        for tk, t in plan['meshes'][str(N)]['targets'].items():
            if not tk.endswith('|H'):
                continue
            for gid, g in t['groups'].items():
                if g['criterion'] != 'joint':
                    continue
                c = res['cells'][f'{N}|{gid}']
                res['hcomp'][f'{N}|{gid}'] = dict(target=tk, family=g['family'], verdict=c.get('two_x', c.get('status')),
                                                  S=c.get('S'), lo=c.get('lo'), hi=c.get('hi'), matched=c.get('selected'))
    # K2 at 4096: every member of the universe, union joint criterion
    if 4096 in jobs:
        uni = plan['meshes']['4096'].get('k2_universe')
        rows = {}
        for tk in uni or []:
            gid = plan['meshes']['4096']['targets'][tk]['union_joint_group']
            c = res['cells'].get(f'4096|{gid}', {})
            rows[tk] = dict(S=c.get('S'), lo=c.get('lo'), hi=c.get('hi'), status=c.get('status', 'ok'), two_x=c.get('two_x'))
        if not plan['meshes']['4096'].get('k2_universe_complete'):
            verdict = 'inconclusive (no complete qualification manifest; front members only)'
        elif any(v['two_x'] == 'pass' for v in rows.values()):
            verdict = 'does not fire'
        elif rows and all(v['two_x'] == 'fail' for v in rows.values()):
            verdict = 'fires'
        else:
            verdict = 'inconclusive'
        res['k2'] = dict(verdict=verdict, members=rows, universe_complete=plan['meshes']['4096'].get('k2_universe_complete'))
    # crossover brackets per identity (X population), per family (joint = (b); same-mesh = (a)) and union
    for ident, ent in plan.get('identities', {}).items():
        out = {}
        for fam in ('UB', 'CC', 'PS', 'union'):
            for crit in ('joint', 'same_mesh'):
                seq = []
                for N in sorted(jobs):
                    tk = ent.get('targets', {}).get(str(N))
                    if tk is None:
                        continue
                    gid = plan['meshes'][str(N)]['targets'][tk]['group_ids'].get(f'{fam}|{crit}')
                    c = res['cells'].get(f'{N}|{gid}') if gid else None
                    if c and c.get('status') == 'ok':
                        seq.append((N, c['faster']))
                    elif c:
                        seq.append((N, c.get('status')))
                ok = [(m, c) for m, c in seq if c in ('ROM faster', 'FOM faster', 'unresolved')]
                out[f'{fam}|{crit}'] = dict(sequence=seq, bracket=bracket(ok))
        res['crossover'][ident] = out
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('K2', res.get('k2', {}).get('verdict'), 'panels', {N: v['panel_ok'] for N, v in res['meshes'].items()})


if __name__ == '__main__':
    main()
