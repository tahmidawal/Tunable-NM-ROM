"""imp-fronts Phase B scoring of one bank's §B5 policy result on the fresh TEST references (CPU, NumPy only).

    python phaseb/score_b.py --label K1 --bank rom/inputs/banks/s1/BANK.json \
        --policy runs/rom/POL-K1-out/output/fpolicy_pol1 --refs runs/refT/pulled/refslim \
        --ledger runs/refT/pulled/REF-LEDGER.json [--out checks/phaseb/score_K1.json]

Per scheduled test case (Amendment 10's 24):
  * the scored terminal run of the policy (result.json policy[*].scored) is loaded from runs/<key>.npz (key verified) and
    decoded on the 257^2 lattice with the SAME bank (floors_b.use: BANK.json hashes checked, and equal to the hashes the
    policy job recorded in result.json['bank']);
  * e = project metric max_{k=1..5} ||v(t_k) - u(t_k)|| / ||u(t_0)|| against the fresh reference (floors_b.load_fresh_ref:
    descriptor and t0 identity checked); invalid / missing scored run -> e = inf;
  * eligible = reference outcome 'accepted' AND REF-LEDGER outcome 'accepted'; eps_ref from the reference;
  * qualified exactly as rom/fpolicy.summarise: terminal == CONVERGED and q1_terminal is True and the pooled
    non-stationary fraction (sum of 'noncertified' / sum of 'steps' over the scored records of ALL cases of the job)
    <= 1 %; recomputed here from the recorded records and asserted equal to the recorded flag;
  * tau_4096 of the decoded scored fields (floors_b.tau_of_coefficients; inf if invalid).
Writes checks/phaseb/score_<label>.json (per case + per-cohort summary).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(HERE))
import floors_b as B  # noqa: E402
import fcommon as F  # noqa: E402

NONST_POOLED = 0.01
REFERENCED = B.REFERENCED


def npz_name(key):
    return key.replace('|', '__').replace('.', 'p') + '.npz'


def pooled(res):
    recs = res['records']
    term = [recs.get(c['scored']) for c in res['policy'] if c.get('scored')]
    tot = sum(int(r['steps']) for r in term if r and 'steps' in r)
    bad = sum(int(r['noncertified']) for r in term if r and 'noncertified' in r)
    frac = bad / tot if tot else None
    return dict(fraction=frac, steps=tot, bad=bad, ok=frac is not None and frac <= NONST_POOLED)


def decode_and_measure(W, u, n0, qualified, decode=None):
    """-> dict(valid, e, tau4096, qualified). Audit codex-code-B-5 row 7: any non-finite coefficient, decoded field,
    error or tail forces valid = False, e = tau4096 = inf and qualified = False."""
    bad = dict(valid=False, e=math.inf, tau4096=math.inf, qualified=False)
    W = np.asarray(W, float)
    if W.shape != (6, 384) or not np.isfinite(W).all():
        return dict(bad, why='non-finite or malformed coefficients')
    V = (decode or (lambda w: F.NC.decode257(w, 384)))(W)
    if not np.isfinite(V).all():
        return dict(bad, why='non-finite decoded field')
    e, t = F.err(V, u, n0), B.tau(V, n0)
    if not (math.isfinite(e) and math.isfinite(t)):
        return dict(bad, why='non-finite error or tail')
    return dict(valid=True, e=e, tau4096=t, qualified=bool(qualified))


def score(label, bank, policy_dir, refs, ledger):
    pdir = Path(policy_dir)
    res = json.loads((pdir / 'result.json').read_text())
    if not res.get('complete', False) and not (pdir / 'COMPLETE').exists():
        raise SystemExit('policy result is not complete')
    b = B.use(bank)
    rb = res['bank']
    if (rb['ckpt_sha256'], rb['rotation_sha256']) != (b['ckpt_sha256'], b['rotation_sha256']):
        raise SystemExit(f'bank {label} differs from the bank the policy job recorded')
    tab = B.FC.load()
    if res.get('table_sha256') != tab['sha256']:
        raise SystemExit('policy job used another fresh-cohort table')
    led = {(r['cohort'], r['case']): r for r in json.loads(Path(ledger).read_text())['cases']}
    pol = {c['case']: c for c in res['policy']}
    pl = pooled(res)
    if abs((pl['fraction'] or 0.) - (res['pooled_nonstationary']['fraction'] or 0.)) > 1e-15 or \
            pl['ok'] != res['pooled_nonstationary']['ok']:
        raise SystemExit('pooled non-stationary fraction does not reproduce')
    cases = []
    for coh in B.FC.COHORTS:
        for j in [r['j'] for r in tab['cohorts'][coh] if r['split'] == 'test'][:REFERENCED[coh]]:
            cid = f'{coh}|{j}'
            u, m = B.load_fresh_ref(refs, coh, j, tab)
            lrow = led.get((coh, j), {})
            p = pol.get(cid)
            row = dict(case=cid, cohort=coh, j=j, eps_ref=m['eps_ref'], ref_outcome=m['outcome'],
                       ledger_outcome=lrow.get('outcome'), eligible=bool(m['eligible'] and lrow.get('outcome') == 'accepted'))
            if p is None:
                row.update(terminal=None, scored=None, valid=False, e=math.inf, tau4096=math.inf, qualified=False,
                           q1_terminal=None, why='no policy record')
                cases.append(row)
                continue
            key = p.get('scored')
            rec = res['records'].get(key) if key else None
            q = bool(p['terminal'] == 'CONVERGED' and p.get('q1_terminal') is True and pl['ok'])
            if q != bool(p.get('qualified')):
                raise SystemExit(f'{cid}: qualification does not reproduce fpolicy.summarise')
            row.update(terminal=p['terminal'], scored=key, q1_terminal=p.get('q1_terminal'), qualified=q,
                       uncertified_start=p.get('uncertified_start'), rounds=p.get('rounds'))
            valid = bool(key and rec and rec.get('valid') and p['terminal'] != 'INVALID')
            if valid:
                z = np.load(pdir / 'runs' / npz_name(key))
                if str(z['key']) != key:
                    raise SystemExit(f'{cid}: run file key {z["key"]} != {key}')
                W = np.asarray(z['W'], float)
                row.update(decode_and_measure(W, u, m['n0'], q),
                           W_sha256=__import__('hashlib').sha256(np.ascontiguousarray(W).tobytes()).hexdigest())
            else:
                row.update(valid=False, e=math.inf, tau4096=math.inf, qualified=False, why='invalid or missing scored run')
            cases.append(row)
    summ = {}
    for coh in B.FC.COHORTS:
        el = [r for r in cases if r['cohort'] == coh and r['eligible']]
        es = [r['e'] for r in el]
        summ[coh] = dict(n_eligible=len(el), median_e_diagnostic=float(np.median(es)) if es else None,
                         worst_e=float(np.max(es)) if es else None, qualified=sum(r['qualified'] for r in el),
                         invalid=sum(not r['valid'] for r in el),
                         terminals={t: sum(r['terminal'] == t for r in el) for t in sorted({r['terminal'] for r in el}, key=str)},
                         median_tau4096=float(np.median([r['tau4096'] for r in el])) if el else None)
    return dict(label=label, bank=b, policy_dir=str(pdir), policy_job=res.get('job_id'), policy_gpu=res.get('gpu'),
                policy_sha256=res.get('policy_sha256'), selected=res['config'].get('selected'),
                table_sha256=tab['sha256'], pooled_nonstationary=pl, cases=cases, summary=summ,
                metric='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1; eps_ref conditional Richardson estimate')


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return 'inf'
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('label', 'bank', 'policy', 'refs', 'ledger'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    out = score(a.label, a.bank, a.policy, a.refs, a.ledger)
    p = Path(a.out or LANE / 'checks' / 'phaseb' / f'score_{a.label}.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(out), indent=1) + '\n')
    for c, v in out['summary'].items():
        print(c, f"n={v['n_eligible']} med_e={v['median_e_diagnostic']:.4g} worst={v['worst_e']:.4g} "
                 f"qual={v['qualified']} inv={v['invalid']} {v['terminals']}")
    print('pooled', out['pooled_nonstationary'], '->', p)


if __name__ == '__main__':
    main()
