"""imp-box stage `test` (test-once; DESIGN §2, A2.2, A2.6, A3.6, A3.7, A4). Runs only with the committed frozen manifest
(MF-LEAK) whose artifact hashes match the job's files (MF-FROZEN). Generates and gates the test references (U_test),
evaluates the frozen finalists on test32 (feasibility by the three-way rule with U_test, Q1, Q2 on test reached states)
and re-times each frozen validation pair on test[0:8] only if both members are test-feasible and qualified. Never
selects anything. Writes test.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import finalize as FZ
import ladder as LD
import refgate as RG
import score as SC


HERE = Path(__file__).resolve().parent


def preflight(frozen, out, W, lev):
    """MF-LEAK: --frozen must be the staged committed copy runs/frozen/<W>_<lev>.json of the code tree and equal the job's
    frozen.json; MF-FROZEN: every frozen artifact hash must match."""
    if not frozen:
        raise SystemExit('MF-LEAK: --frozen (the committed copy of frozen.json) is required')
    fz = Path(frozen).resolve()
    if fz != (HERE / 'runs' / 'frozen' / f'{W}_{lev}.json').resolve():
        raise SystemExit('MF-LEAK: --frozen must be the staged committed copy under code/runs/frozen/')
    for f_ in (fz, HERE / 'runs' / 'frozen' / 'INDEX.json'):      # committed: blob hash in the staged commit's tree
        if not committed(f_):
            raise SystemExit(f'MF-LEAK: {f_.name} is not the committed content of the staged commit')
    idx = json.loads((HERE / 'runs' / 'frozen' / 'INDEX.json').read_text())
    cell = f'{W}_{lev}'
    if cell not in idx.get('cells', {}) or idx['cells'][cell] != hashlib.sha256(fz.read_bytes()).hexdigest():
        raise SystemExit('MF-LEAK: the campaign freeze index does not list this frozen manifest')
    for rc in idx.get('retained', []):                   # campaign completeness: every retained cell frozen + committed
        f_ = HERE / 'runs' / 'frozen' / f'{rc}.json'
        if rc not in idx['cells'] or not committed(f_) or \
                hashlib.sha256(f_.read_bytes()).hexdigest() != idx['cells'][rc]:
            raise SystemExit(f'MF-LEAK: retained cell {rc} is not frozen and committed in the index')
    if cell not in idx.get('retained', []):
        raise SystemExit('MF-LEAK: this cell is not in the retained list of the freeze index')
    job = out / 'frozen.json'
    if not fz.exists() or not job.exists() or \
            hashlib.sha256(fz.read_bytes()).hexdigest() != hashlib.sha256(job.read_bytes()).hexdigest():
        raise SystemExit('MF-LEAK: committed frozen manifest missing or different from the job frozen.json')
    fr = json.loads(fz.read_text())
    check_artifacts(fr, out)
    return fr


def committed(path):
    """True iff `path` (inside the staged code tree) has the git blob hash listed for it in .git_tree.txt, which
    submit.sh writes from `git ls-tree -r <commit>` of the lane at the staged commit."""
    tree = HERE / '.git_tree.txt'
    if not tree.exists() or not Path(path).exists():
        return False
    rel = str(Path(path).resolve().relative_to(HERE))
    data = Path(path).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    for line in tree.read_text().splitlines():
        meta, name = line.split('\t', 1)
        if name.endswith('/' + rel) or name == rel:
            return meta.split()[2] == blob
    return False


def check_artifacts(fr, d):
    for name, h in fr['artifacts'].items():
        if hashlib.sha256((Path(d) / name).read_bytes()).hexdigest() != h:
            raise SystemExit(f'MF-FROZEN: artifact {name} changed since the freeze')


def leak_controls(out, W, lev):
    """MF-LEAK / MF-FROZEN fixtures (controls stage): no manifest -> refuse; an artifact with one changed byte ->
    refuse (checked on a temporary copy; the real artifacts are untouched)."""
    import shutil
    import tempfile
    res = {}
    msgs = []
    for fz in (None, out / 'frozen.json'):              # no manifest; an uncommitted copy outside the staged tree
        try:
            preflight(fz, out, W, lev)
            msgs.append(None)
        except SystemExit as e:
            msgs.append(str(e))
    res['MF_LEAK'] = dict(fires=bool(all(m is not None for m in msgs)), msgs=msgs)
    fr = json.loads((out / 'frozen.json').read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(out / 'frozen.json', td / 'frozen.json')
        if not fr['artifacts']:
            res['MF_FROZEN'] = dict(fires=False, msg='no artifacts')
            return res
        name = min(fr['artifacts'], key=lambda nm: (out / nm).stat().st_size)     # copy only the smallest artifact
        shutil.copy(out / name, td / name)
        one = dict(artifacts={name: fr['artifacts'][name]})
        try:
            check_artifacts(one, td)
            positive = True
        except SystemExit:
            positive = False
        b = bytearray((td / name).read_bytes())
        b[-1] ^= 1
        (td / name).write_bytes(bytes(b))
        try:
            check_artifacts(one, td)
            res['MF_FROZEN'] = dict(fires=False, artifact=name, positive=positive)
        except SystemExit as e:
            res['MF_FROZEN'] = dict(fires=True, artifact=name, msg=str(e), positive=positive)
    return res


def pair_matched(ea, eb):
    """A3.7 rule: a frozen pair is matched on test only if both members are test-feasible ('meets') and qualified."""
    return bool(ea and eb and all(x['feasibility'] == 'meets' and x['qualified'] for x in list(ea) + list(eb)))


def test_fixtures():
    """Forced test-Q2 failure -> not matched (no reselection: the stage only iterates frozen finalists)."""
    ok = dict(feasibility='meets', qualified=True)
    q2fail = dict(feasibility='meets', qualified=False)
    miss = dict(feasibility='unresolved', qualified=True)
    t = dict(clean_matched=pair_matched([ok], [ok]), q2_fail_not_matched=not pair_matched([ok], [q2fail]),
             infeasible_not_matched=not pair_matched([miss], [ok]), empty_not_matched=not pair_matched([], [ok]))
    t['fires'] = bool(all(t.values()))
    return t


def check_inputs(fr, out, D):
    """Frozen descriptor hashes and query-defining source identity (code audit 2)."""
    import cell as CL
    import timing as TM
    for k, h in fr['cohorts']['cell_hashes'].items():
        if CL.sha_arr(D[k]) != h:
            raise SystemExit(f'MF-FROZEN: descriptors of split {k} differ from the freeze')
    if TM.source_identity() != fr['source']:
        raise SystemExit('MF-FROZEN: query-defining source files differ from the freeze')
    cj = out / 'controls.json'
    if not (cj.exists() and json.loads(cj.read_text()).get('all_fire', False)):
        raise SystemExit('test refused: the cell did not pass its mandatory controls')


def stage_test(a, out, D):
    import cell as CL
    import timing as TM
    fr = preflight(a.frozen, out, a.W, a.lev)
    check_inputs(fr, out, D)
    Ut, Bt, rec = CL.gate_split(a, D['test'], 'test')
    el = np.array(rec['eligible'])
    U_test = RG.allowance(rec['eps'], rec['eligible'])
    np.savez(out / 'refs_test.npz', U=Ut, eps=np.array(rec['eps']), eligible=el)
    if not rec['scorable']:                          # persisted unscorable result; nothing evaluated (code audit 2)
        (out / 'test.json').write_text(json.dumps(TM.jsonable(dict(U_test=U_test, ref=rec, scorable=False,
                                                                     entries={}, pairs={})), indent=1))
        return
    c = TM.Ctx.__new__(TM.Ctx)
    c.a, c.out, c.U = a, out, U_test
    c.Dv, c.Uv = D['test'][el], Ut[el]
    c.Dd, c.Ud = c.Dv[:8], c.Uv[:8]
    c.L = None
    c.cache_off, c.cache_qual, c.cache_q2, c.cache_scr = {}, {}, {}, {}
    res = dict(U_test=U_test, ref=rec, scorable=rec['scorable'], entries={}, pairs={},
               claim_context=dict(test=f"{int(el.sum())} of {len(el)}", test_eligible=int(el.sum()), test_total=len(el),
                                  val_context=fr.get('claim_context'), training_gates=fr.get('training_gates')))
    arms = sorted({k.split('|')[0] for k in fr['selection']})
    for name in arms:
        arm = TM.Arm(name, c)
        for k, v in fr['selection'].items():
            if k.split('|')[0] != name:
                continue
            _, tau, stat = k.split('|')
            ent = []
            for f in v['finalists']:
                fk = f['key']
                if fk not in c.cache_off:
                    O, st = FZ.run_unbatched(arm.build(f['cfg']), c.Dv)
                    e = np.where(LD.stats_valid(f['cfg'], st), SC.err(O, c.Uv), np.inf)
                    c.cache_off[fk] = (O, st, dict(worst=float(np.max(e)), median=float(np.median(e)), e=e.tolist()))
                    c.cache_qual[fk] = TM.jsonable(arm.qualify(f['cfg'], O, st))
                O, st, sc = c.cache_off[fk]
                ql = c.cache_qual[fk]
                feas = SC.three_way(sc[stat], U_test, float(tau))
                ok = bool(ql['Q1']['passed'] and (ql['Q2'] is None or ql['Q2']['passed']))
                ent.append(dict(key=fk, test=sc, feasibility=feas, qualified=ok, qual=ql))
            res['entries'][k] = dict(finalists=ent, val_status=v['status'],
                                     boundary_labels=v.get('boundary_labels', []),
                                     construction_failures=v.get('construction_failures', 0))
        res.setdefault('_arms_done', []).append(name)
        del arm
    # frozen validation pairs: re-time only when both members are test-feasible and qualified
    for k, p in fr['pairs'].items():
        name, tau, stat = k.split('|')
        ea = [x for kk, v in res['entries'].items() for x in v['finalists'] if x['key'] == p['competitor_key']
              and kk.endswith(f'|{tau}|{stat}')]
        eb = [x for x in res['entries'].get(k, {}).get('finalists', []) if x['key'] == p['arm_key']]
        good = pair_matched(ea[:1], eb[:1]) and res['scorable']
        res['pairs'][k] = dict(S_val=p['S_val'], numerically_matched=good, matched_on_test=False,
                               reason=None if good else 'not test-feasible / not qualified / unscorable',
                               val_excluded=p.get('excluded', []), val_arm_status=p.get('arm_status'))
        if good:
            fa = ea[0]['key'].split('|')[0]
            ca = next(f['cfg'] for v in fr['selection'].values() for f in v['finalists'] if f['key'] == p['competitor_key'])
            cb = next(f['cfg'] for f in fr['selection'][k]['finalists'] if f['key'] == p['arm_key'])
            qa, qb = TM.Arm(fa, c).build(ca), TM.Arm(name, c).build(cb)
            r_, OA, OB = TM.timed_pair(qa, qb, c.Dv[:8])
            tia = FZ.t_id(OA, c.cache_off[p['competitor_key']][0][:8])
            tib = FZ.t_id(OB, c.cache_off[p['arm_key']][0][:8])
            dA, dB = TM.t_dbl_raw(qa, c.Dv), TM.t_dbl_raw(qb, c.Dv)       # same validity contract as val (audit 5)
            valid = bool(tia <= 1e-12 and tib <= 1e-12 and r_['compiles_in_timed_region'] == 0
                         and dA['passed'] and dB['passed'])
            res['pairs'][k].update(T_dbl_A=dA, T_dbl_B=dB, matched_on_test=valid,
                                   reason=None if valid else 'timing invalid (T-id / compile / T-dbl)')
            res['pairs'][k].update(S_test=(r_['S'] if valid else None), timing_valid=valid,
                                   label='speed-up of the validation-selected pair', T_id_A=tia, T_id_B=tib,
                                   compiles=r_['compiles_in_timed_region'], raw=r_['raw'])
    (out / 'test.json').write_text(json.dumps(TM.jsonable(res), indent=1))
