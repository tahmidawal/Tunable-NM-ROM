"""jcp-robust E2 3D rule selection per (bank, width, spec, family) (DESIGN 3.3; A1.8, A2.5, A2.6, A3.3). CPU (offline first).

From the two selection jobs of one bank (e2q<b>a: W256, W512; e2q<b>b: W768, W1024; d3run 'ladder' blocks) and their E0
scores: a candidate rung is qualified iff, on every one of the 64 cases, its production run is valid, the pooled
non-stationary fraction (fresh in-job certificate, recomputed from the saved gam_cert rows) is <= 1 %, the ADT tight
anchor is valid and within 2.5e-4 (recomputed), tight/tighter <= 2.5e-5 on cases 0-7, its successor rung is valid and
within 2.5e-4 (recomputed), rho against the successor <= 0.116, and (cross-family) it is within 2.5e-4 of gl64 on every
case. The target check is gl56 vs gl64 <= 2.5e-5 on cases 0-7 with both valid; if it fails, the pick is a diagnostic pick
on the own-family criteria (A2.5). Search order = the candidate lists; three-state: an undetermined candidate stops the
search ("unavailable"); "none within ladder" only when every candidate is definitively rejected.

    python select3d.py <bank> <e2q<b>a dir> <e2q<b>b dir>
"""
import offline  # noqa: F401

import json
import math
import sys
from pathlib import Path

import numpy as np

import e1stats as E
import e2score3 as ES
from make_configs_e2_3d import GAUSS, GAUSS_CAND, LAT, LAT_CAND, SPECS

HERE = Path(__file__).resolve().parent


def num(x):
    return E.num(x)


def main():
    b = int(sys.argv[1])
    dirs = [Path(x) for x in sys.argv[2:4]]
    import c3common as C3
    import score3 as SC
    rep = dict(bank=b, slots=[], cells={}, target={}, jobs={})
    for d in dirs:
        res = json.loads((d / 'result.json').read_text())
        cfg = res['config']
        ES.install(cfg)
        rep['jobs'][res['config']['attempt']] = dict(job_id=res['job_id'], complete=bool(res.get('complete')),
                                                     backend=res['backend'], memory_rejected=res.get('memory_rejected', []))
        Wz = np.load(d / f"W_{res['mode']}.npz")
        smoke = str(cfg.get('attempt', '')).startswith('smk')
        job_ok = bool(res.get('complete') and (d / 'COMPLETE').exists() and res['backend'] == 'gpu'
                      and not res.get('memory_rejected') and (smoke or [int(x) for x in res['cohort']['cases']] == list(range(64))))
        rep['jobs'][res['config']['attempt']]['ok'] = job_ok
        tab = C3.C.table(int(cfg['cohort_seed']), int(cfg['cohort_count']))
        cases = [int(x) for x in res['cohort']['cases']]
        n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in cases}
        lad = {}
        for blk in res['blocks']:
            if blk['kind'] == 'ladder':
                lad.update(blk['result'])

        def dist(sname, A_, B_, ids=None):
            G = SC.glat(sname)
            ids = cases if ids is None else ids
            return [max(float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[j] for k in range(1, 6)) for i, j in enumerate(ids)]

        for st in cfg['settings']:
            sname, wd = st['name'], int(st['Rp'])
            for spec in SPECS:
                g = {rl: lad.get(f'{sname}|{rl}|{spec}') for rl in GAUSS + LAT}
                W = {rl: Wz[f'{sname}|{rl}|{spec}'] for rl in GAUSS + LAT if f'{sname}|{rl}|{spec}' in Wz.files}
                tgt_ok = None
                if g.get('gl56') and g.get('gl64') and 'gl56' in W and 'gl64' in W:
                    dchk = dist(sname, W['gl56'][:8], W['gl64'][:8], cases[:8])
                    tgt_ok = bool(all(g['gl56']['valid'][:8]) and all(g['gl64']['valid']) and max(dchk) <= E.BAR_TC)
                    rep['target'][f'{sname}|{spec}'] = dict(dist_check=dchk, certified=tgt_ok, all_target_valid=all(g['gl64']['valid']))
                for fam, cand, ladder in (('gauss', GAUSS_CAND, GAUSS), ('lattice', LAT_CAND, LAT)):
                    cells = {}
                    for rl in cand:
                        x = g.get(rl)
                        nxt = ladder[ladder.index(rl) + 1]
                        if x is None or rl not in W or nxt not in W or g.get(nxt) is None:
                            cells[rl] = dict(qualified=None, own=None, status='missing')
                            continue
                        rows = [r_ for r_ in res['rows'] if r_.get('tag') == f'{sname}|{rl}' and r_.get('spec') == spec and 'gam_cert' in r_]
                        gam = [num(v) for r_ in rows for v in r_['gam_cert']]
                        nonst = (sum(1 for v in gam if not (np.isfinite(v) and v <= E.GAMMA)) / len(gam)) if (gam and len(rows) == len(cases)) else None
                        sp = spec.split('|')
                        ka = f"{sname}|{rl}|ad|LSPG|{sp[1]}|{float(sp[2]):.8g}|tight"
                        kb = f"{sname}|{rl}|ad|LSPG|{sp[1]}|{float(sp[2]):.8g}|tighter"
                        d_adt = dist(sname, W[rl], Wz[ka]) if ka in Wz.files else None
                        # code audit 2 #24: tight/tighter recomputed and compared; registered tighter cases 0-7
                        tcs = [j for j in range(8) if j in cases]
                        ib = [int(v) for v in Wz['cases|' + kb]] if kb in Wz.files else None
                        d_tt = (dist(sname, [Wz[ka][cases.index(j)] for j in ib], Wz[kb], ib) if (ib == tcs and ka in Wz.files)
                                else None)
                        bad_agree = []
                        for nm_, off, job in (('adt', d_adt, x['dist_adt']), ('tt', d_tt, x['adt_tight_tighter'])):
                            if off is None or len(off) != len(job):
                                bad_agree.append((nm_, 'missing'))
                            else:
                                bad_agree += [(nm_, i) for i, (o_, j_) in enumerate(zip(off, job))
                                              if np.isfinite(o_) and abs(o_ - num(j_)) > 1e-9 * abs(num(j_)) + 1e-13]
                        d_next = dist(sname, W[rl], W[nxt])
                        d_tgt = dist(sname, W[rl], W['gl64']) if 'gl64' in W else None
                        tt = d_tt
                        if bad_agree or not job_ok:
                            cells[rl] = dict(qualified=None, own=None, status='integrity: ' + (str(bad_agree[:3]) if bad_agree else 'job'))
                            continue
                        qual, own, status = E.draw_qualified(
                            dict(valid=x['valid'], valid_tight=x['adt_valid'], valid_tighter=x['adt_tighter_valid'],
                                 valid_succ=g[nxt]['valid']),
                            dict(tight=d_adt, tt=tt, succ=d_next, rho_succ=[max(num(v) for v in rr) for rr in x['rho']],
                                 # 3D selection's cross-family criterion is the rollout distance to gl64 only (A1.8,
                                 # the dims (3M) predicate); no rho against the target is registered, so it is not tested
                                 target=d_tgt, rho_tgt=[0.] * len(cases), nonst_frac=nonst),
                            (None if tgt_ok is None or tgt_ok is False else bool(all(g['gl64']['valid']))),
                            len(cases), len(tcs))
                        cells[rl] = dict(qualified=qual, own=own, status=status, m=x.get('m'),
                                         dist_next_max=max(d_next), dist_tgt_max=max(d_tgt) if d_tgt else None)
                    key = 'qualified' if tgt_ok else 'own'
                    pick, outcome = None, 'none within ladder'
                    for rl in cand:
                        v = cells[rl][key]
                        if v is True:
                            pick, outcome = rl, ('selected' if tgt_ok else 'diagnostic pick (target unverified)')
                            break
                        if v is None:
                            outcome = f'unavailable (evidence for {rl} undetermined)'
                            break
                    rep['cells'][f'{sname}|{spec}|{fam}'] = cells
                    rep['slots'].append(dict(width=wd, spec=spec, family=fam, rule=pick, outcome=outcome))
                    print(sname, spec, fam, pick, outcome, flush=True)
    (HERE / 'checks' / f'select3d-{b}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
