"""jcp-fom2d matching targets (DESIGN §4). CPU only.

  python targets.py p1 --out checks/targets-p1.json
      PREVIEW targets: the numerically qualified P1 configurations (jcp-p1-scaling checks/analysis.json), per-case E0 R_ST
      scores recomputed here from P1's saved lattice fields (runs/acc<mesh>/archive/output/p1acc_acc<mesh>/R_<cfg>_<scheme>.npz).
      Label: "preview against P1 configurations; superseded by FRONT-2D". Selects nothing.
  python targets.py front --front <FRONT-2D.json> --commit <sha> --out checks/targets-front.json
      the jcp-qualify front (format read when the file exists; see DESIGN amendments).

A target: {mesh, case_ids, e {cid: e}, eta {cid}, eligible {cid}, median, worst, config, source}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import refstore as RS

P1 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling')


def target_from_fields(f257, case_ids, mesh, config, source):
    e, eta, el = {}, {}, {}
    for i, cid in enumerate(case_ids):
        coh, c = cid.split('|')
        s = RS.score(f257[i], 2, coh, int(c), 'ST')
        e[cid], eta[cid], el[cid] = s['e'], s['eps_ref'], s['eligible']
    ids = [c for c in case_ids if el[c]]
    return dict(mesh=int(mesh), case_ids=list(case_ids), e=e, eta=eta, eligible=el,
                median=float(np.median([e[c] for c in ids])) if ids else None,
                worst=float(np.max([e[c] for c in ids])) if ids else None,
                n_eligible=len(ids), config=config, source=source,
                status='ok' if len(ids) >= 3 else 'population too small')


def identity(cfg):
    """Configuration identity, mesh excluded (A1.4): bank, r, rule, mode policy, scheme, dt, form, tolerance level."""
    return (f"{cfg.get('bank', 'dn256b')}|R{cfg['Rp']}|{cfg['rule']}|{cfg['policy']}|{cfg['scheme']}|{cfg['dt']:.8g}|"
            f"{cfg.get('form', 'LSPG')}|{cfg.get('level', 'prod')}")


def with_populations(T):
    """A2.3: every target in two variants, <identity>@<mesh>|H (its published population) and |X (the intersection of the
    identity's eligible cases over every mesh where it is present)."""
    by = {}
    for t in T.values():
        by.setdefault(identity(t['config']), []).append(t)
    out = {}
    for idt, ts in by.items():
        inter = set.intersection(*[{c for c in t['case_ids'] if t['eligible'][c]} for t in ts])
        assert len({t['mesh'] for t in ts}) == len(ts), f'identity collision at one mesh: {idt}'
        for t in ts:
            out[f"{idt}@{t['mesh']}|H"] = dict(t, identity=idt, population='H')
            ids = [c for c in t['case_ids'] if c in inter]
            x = dict(t, identity=idt, population='X', case_ids=ids, e={c: t['e'][c] for c in ids},
                     eta={c: t['eta'][c] for c in ids}, eligible={c: t['eligible'][c] for c in ids},
                     meshes_present=sorted(u['mesh'] for u in ts))
            el = [c for c in ids if x['eligible'][c]]
            x.update(median=float(np.median([x['e'][c] for c in el])) if el else None,
                     worst=float(np.max([x['e'][c] for c in el])) if el else None, n_eligible=len(el),
                     status='ok' if len(el) >= 3 else 'population too small')
            out[f"{idt}@{t['mesh']}|X"] = x
    return out


def p1(out):
    an = json.loads((P1 / 'checks' / 'analysis.json').read_text())
    T = {}
    for lab, v in an['configurations'].items():
        if not v.get('numerically_qualified'):
            continue
        mesh, cfg, scheme = lab.split('|')
        f = P1 / 'runs' / f'acc{mesh}' / 'archive' / 'output' / f'p1acc_acc{mesh}' / f'R_{cfg}_{scheme}.npz'
        z = np.load(f)
        assert int(z['mesh']) == int(mesh) and str(z['scheme']) == scheme, (lab, z['mesh'], z['scheme'])
        ids = [str(x) for x in z['case_ids']]
        t = target_from_fields(z['f257'], ids, mesh, dict(mesh=int(mesh), Rp=int(z['Rp']), rule=str(z['rule']),
                                                          policy=str(z['policy']), scheme=scheme, dt=float(z['dt'])),
                               dict(file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        # consistency with P1's own analysis (same store): medians/worst over its population must agree
        t['p1_median'], t['p1_worst'] = v['e0_median'], v['e0_worst']
        assert t['median'] is not None and abs(t['median'] - v['e0_median']) <= 1e-12 and \
            abs(t['worst'] - v['e0_worst']) <= 1e-12, (lab, t['median'], v)
        T[lab] = t
    Path(out).write_text(json.dumps(dict(source='preview against P1 configurations; superseded by FRONT-2D',
                                         store_version=RS.store_version(), targets=with_populations(T)), indent=1) + '\n')
    for lab, t in T.items():
        print(lab, t['n_eligible'], t['median'], t['worst'])


def front(path, commit, out, k2_out):
    """FRONT-2D.json (jcp-qualify front2d.py schema): per mesh, members + other_qualified entries with a full manifest,
    labels and per-case E0 ST errors (38 cases). Every numerically qualified entry (both labels) becomes a target; the K2
    universe is every numerically qualified linear-span entry at 4096 (all of them: the bank span is linear); it is
    complete iff the file lists members, other_qualified and unqualified for 4096."""
    raw = Path(path).read_bytes()
    F = json.loads(raw)
    sha_file = hashlib.sha256(raw).hexdigest()
    T, k2 = {}, []
    sv = None
    for mesh, M in F['meshes'].items():
        if 'members' not in M:
            continue
        for ent in M['members'] + M['other_qualified']:
            lab, man = ent['labels'], ent['manifest']
            if not (lab.get('numerically_qualified_accuracy_job') and lab.get('numerically_qualified_timing_job')):
                continue
            sv = sv or man['e0_store_version']
            assert man['e0_store_version'] == sv
            pc = ent['errors']['per_case']
            ids = sorted(pc, key=lambda c: (c.split('|')[0] != 'dev6', int(c.split('|')[1])))
            cfg = dict(mesh=int(mesh), Rp=int(man['Rp']), rule=man['rule'], policy='frozen', scheme=man['scheme'],
                       dt=float(man['dt']), form=man['form'], level=man['level'], bank='ckpt:' + man['checkpoint_sha256'][:12],
                       budget=man.get('budget'), fallback=man.get('fallback'), front_key=ent['config'])
            t = dict(mesh=int(mesh), case_ids=ids, e={c: pc[c]['e'] for c in ids}, eta={c: pc[c]['eps_ref'] for c in ids},
                     eligible={c: bool(pc[c]['eligible']) for c in ids}, config=cfg,
                     source=dict(file=str(path), sha256=sha_file, commit=commit, front_member=bool(lab.get('fronts')),
                                 fronts=lab.get('fronts'), front_timing=ent.get('timing')))
            el = [c for c in ids if t['eligible'][c]]
            t.update(median=float(np.median([t['e'][c] for c in el])) if el else None,
                     worst=float(np.max([t['e'][c] for c in el])) if el else None, n_eligible=len(el),
                     status='ok' if len(el) >= 3 else 'population too small')
            assert abs(t['median'] - ent['errors']['median']) <= 1e-12 and abs(t['worst'] - ent['errors']['worst']) <= 1e-12, ent['config']
            T[f"{mesh}|{ent['config']}"] = t
            if int(mesh) == 4096:
                k2.append(identity(cfg))
    assert sv == RS.store_version(), ('FRONT store version differs from the active E0 store', sv)
    M4 = F['meshes'].get('4096', F['meshes'].get(4096, {}))

    def qualified_anywhere(v):            # an 'unqualified' entry that still carries a true numerical-qualification label
        if isinstance(v, dict):
            return any((('numerically_qualified' in str(k)) and x is True) or qualified_anywhere(x) for k, x in v.items())
        if isinstance(v, list):
            return any(qualified_anywhere(x) for x in v)
        return False
    hidden = [k for k, v in (M4.get('unqualified') or {}).items() if qualified_anywhere(v)]
    complete = all(k in M4 for k in ('members', 'other_qualified', 'unqualified')) and not hidden
    if hidden:
        print('K2 universe incomplete: numerically qualified entries listed as unqualified (e.g. unusable timing):', hidden)
    Path(out).write_text(json.dumps(dict(source=f'FRONT-2D ({path}, sha256 {sha_file}, commit {commit})',
                                         store_version=RS.store_version(), targets=with_populations(T)), indent=1) + '\n')
    Path(k2_out).write_text(json.dumps(dict(identities=sorted(set(k2)), complete=bool(complete),
                                            source=f'FRONT-2D sha256 {sha_file}'), indent=1) + '\n')
    print(len(T), 'targets;', len(set(k2)), 'K2 identities; complete', complete)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['p1', 'front'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--front')
    ap.add_argument('--commit')
    ap.add_argument('--k2-out')
    a = ap.parse_args()
    if a.kind == 'p1':
        p1(a.out)
    else:
        front(a.front, a.commit, a.out, a.k2_out)


if __name__ == '__main__':
    main()
