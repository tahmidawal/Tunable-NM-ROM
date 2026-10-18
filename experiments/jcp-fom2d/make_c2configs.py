"""jcp-fom2d C2 configurations and analysis plan (DESIGN §5, A1.3, A1.4, A2.1–A2.3, A3.4). CPU only.

    python make_c2configs.py --match checks/match-<x>.json --targets checks/targets-<x>.json --ladder <dir> [...]
                             --k2 checks/k2-universe.json --prefix c2 --out-plan checks/c2plan.json

Per ROM mesh N, configs/<prefix><N>.json holds:
  A        the P1 reference ROM (R384, gauss128, frozen, LSPG-CN-R, dt0) under label 'A'
  roms     every ROM identity with a target at N (label = identity), with its H-population errors on the timing cases
  foms     every FOM candidate of any group, with its ladder errors on the timing cases (G-FR)
  groups   per target key (H and X populations) x family x criterion (joint, median_only, worst_only, same_mesh,
           same_mesh_unmatched) with >= 1 candidate; union groups (pool_by_family) for joint and same_mesh
The plan records, per mesh, the target -> group map, empty groups ("no match"), the K2 universe at 4096 and per identity
the X-population target at each mesh (crossover).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import match2 as M2

HERE = Path(__file__).resolve().parent
FAMS = ('UB', 'CC', 'PS')
CRITS = ('joint', 'median_only', 'worst_only', 'same_mesh', 'same_mesh_unmatched')
TIMING_CASES = [['dev6', j] for j in range(6)]
A_SPEC = dict(label='A', Rp=384, rule='gauss128', policy='frozen', scheme='CNR', dt=0.005, form='LSPG', level='prod')
EDGES = {256: [0, 128, 256, 384, 512], 1024: [0, 128, 256, 384, 512], 4096: [0, 128, 256, 384, 512]}


def ladder_scores(dirs):
    out = {}
    for d in dirs:
        r = json.loads((Path(d) / 'result.json').read_text())
        for k, v in r['configs'].items():
            out[k] = {p['case']: p['scores']['ST']['e'] for p in v['production']}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--ladder', action='append', required=True)
    ap.add_argument('--k2', required=True, help='{"identities": [...], "complete": bool, "source": "..."}')
    ap.add_argument('--prefix', default='c2')
    ap.add_argument('--meshes', default='256,1024,4096')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out-plan', required=True)
    a = ap.parse_args()
    M = json.loads(Path(a.match).read_text())
    assert not M['blocked'], 'the C1 matcher is blocked (acceptance failed): no C2 configs'
    assert M.get('extension_complete') is True, 'the A3.2 extension round is required and not complete: no C2 configs'
    T = json.loads(Path(a.targets).read_text())['targets']
    L = ladder_scores(a.ladder)
    K2 = json.loads(Path(a.k2).read_text())
    plan = dict(match=a.match, targets=a.targets, meshes={}, identities={})
    adm = M['admissibility']
    src = {}
    for d in a.ladder:
        for k in json.loads((Path(d) / 'result.json').read_text())['configs']:
            src[k] = Path(d)
    cheapest = {}
    for k, v in adm.items():
        if v['admissible'] and (v['family'] not in cheapest or (v['cost'], k) < (adm[cheapest[v['family']]]['cost'], cheapest[v['family']])):
            cheapest[v['family']] = k
    ann = {}

    def annotate(k):
        """Per-candidate labels for whichever comparator C2 selects (A3.4): edge reasons, boundary, PS anchor."""
        if k not in ann:
            f = adm[k]['family']
            ann[k] = dict(edge=M2.edge_reasons(k, M['edges']), boundary=cheapest.get(f) == k, upper_bound=True,
                          anchor=M2.anchor(k, adm, src) if f == 'PS' else None)
        return ann[k]
    for N in [int(x) for x in a.meshes.split(',')]:
        tks = [k for k, t in T.items() if t['mesh'] == N]
        roms, groups, ptargets = {}, [], {}
        for tk in tks:
            t = T[tk]
            idt = t['identity']
            if idt not in roms:
                h = T[f'{idt}@{N}|H']
                roms[idt] = dict(label=idt, **{k: t['config'][k] for k in ('Rp', 'rule', 'policy', 'scheme', 'dt')},
                                 form=t['config'].get('form', 'LSPG'), level=t['config'].get('level', 'prod'),
                                 target_e={f'{c}|{j}': h['e'][f'{c}|{j}'] for c, j in TIMING_CASES})
            m = M['matches'][tk]
            ent = dict(rom_label=idt, population=t['population'], groups={}, group_ids={}, empty={}, status=m.get('status'))
            if m.get('status') == 'ok':
                for f in FAMS:
                    for cr in CRITS:
                        cands = m[f]['candidates'][cr]
                        gid = f'{tk}|{f}|{cr}'
                        if not cands:
                            ent['empty'][f'{f}|{cr}'] = 'no match' if cr != 'same_mesh' else m[f]['same_mesh_status']
                            continue
                        groups.append(dict(id=gid, target=idt, family=f, criterion=cr, candidates=cands))
                        ent['groups'][gid] = dict(family=f, criterion=cr, candidate_labels={k: annotate(k) for k in cands})
                        ent['group_ids'][f'{f}|{cr}'] = gid
                for cr in ('joint', 'same_mesh'):
                    cands = sorted({k for f in FAMS for k in m[f]['candidates'][cr]})
                    gid = f'{tk}|union|{cr}'
                    if not cands:
                        ent['empty'][f'union|{cr}'] = 'no match'
                        continue
                    groups.append(dict(id=gid, target=idt, family='union', criterion=cr, candidates=cands, pool_by_family=True))
                    ent['groups'][gid] = dict(family='union', criterion=cr, candidate_labels={k: annotate(k) for k in cands})
                    ent['group_ids'][f'union|{cr}'] = gid
                ent['union_joint_group'] = ent['group_ids'].get('union|joint')
            ptargets[tk] = ent
            if t['population'] == 'X':
                plan['identities'].setdefault(idt, dict(targets={}))['targets'][str(N)] = tk
        fkeys = sorted({k for g in groups for k in g['candidates']})
        foms = {k: dict(ladder_e={f'{c}|{j}': L[k][f'{c}|{j}'] for c, j in TIMING_CASES}) for k in fkeys}
        fault_fom = fkeys[0] if fkeys else None
        cfg = dict(attempt=f'{a.prefix}{N}', mesh=N, edges=EDGES[N], cases=TIMING_CASES, A=A_SPEC, roms=list(roms.values()),
                   foms=foms, groups=groups, reps=a.reps, burn=0.1, seed=N,
                   controls=dict(doubled_cases=[['dev6', 0], ['dev6', 1]], fault_subject=fault_fom,
                                 identity_fault_subject=fault_fom))
        (HERE / 'configs' / f'{a.prefix}{N}.json').write_text(json.dumps(cfg, indent=1) + '\n')
        md = dict(targets=ptargets, n_foms=len(fkeys), n_groups=len(groups))
        if N == 4096:
            uni = [f'{i}@4096|H' for i in K2['identities']]
            md.update(k2_universe=[u for u in uni if u in ptargets], k2_universe_complete=bool(K2['complete'])
                      and all(u in ptargets for u in uni), k2_source=K2.get('source'))
        plan['meshes'][str(N)] = md
        print(N, 'roms', len(roms), 'foms', len(fkeys), 'groups', len(groups))
    Path(a.out_plan).write_text(json.dumps(plan, indent=1) + '\n')


if __name__ == '__main__':
    main()
