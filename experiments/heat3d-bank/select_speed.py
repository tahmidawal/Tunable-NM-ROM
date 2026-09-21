"""Addendum-1 selection (DESIGN.md), VALIDATION summaries only. Usage: select_speed.py <run>/speed_vp_R*_K* ...
Writes selection_addendum1.json and configs/final01b.json (panel B) or exits without a config if nothing qualifies."""
import json, re, sys
from pathlib import Path
import make_configs as MC

GATE, BAR = 0.006, 0.008
HERE = Path(__file__).resolve().parent; cands = []
for d in (Path(a).resolve() for a in sys.argv[1:]):
    s = json.loads((d / 'summary.json').read_text()); m = re.match(r'speed_vp_R(\d+)_K(\d+)', d.name); R, K = int(m[1]), int(m[2])
    tr = json.loads((HERE / 'inputs' / f'vp_R{R}' / 'training.json').read_text())
    floor = {f['intervals']: f['worst'] for f in tr['bank']['validation_floor_other_grids']}
    assert all(mesh['cohort'] == 'validation' for mesh in s['meshes'])
    rows = {mesh['intervals']: {r['method']: r for r in mesh['rows']} for mesh in s['meshes']}
    e = dict(R=R, K=K, run=str(d.relative_to(HERE)), gate=floor[128] <= GATE, accurate_q=None, per_q={})
    for q in MC.SPEED_Q[R]:
        nm = f'nmrom_q{q}_field_cn_tol1e-4_chol'
        if nm not in rows[128]: continue
        e['per_q'][q] = {n: dict(err=rows[n][nm]['error_all_times_worst'], failures=rows[n][nm]['failures'], ms=rows[n][nm]['device_ms_median']) for n in (64, 128)}
        if e['accurate_q'] is None and all(v['err'] <= BAR and v['failures'] == 0 for v in e['per_q'][q].values()):
            e['accurate_q'] = q; e['ms_128'] = rows[128][nm]['device_ms_median']
    cands.append(e)
ok = [c for c in cands if c['gate'] and c['accurate_q'] is not None]
if not ok:
    (HERE / 'selection_addendum1.json').write_text(json.dumps(dict(rule='addendum 1: nothing qualified; no panel B', candidates=cands), indent=1) + '\n')
    sys.exit('addendum 1: nothing qualified; no panel B')
best = min(c['ms_128'] for c in ok); ch = min([c for c in ok if c['ms_128'] <= 1.05 * best], key=lambda c: (c['R'], c['K']))
R, K, qa = ch['R'], ch['K'], ch['accurate_q']
arms = [dict(name=f'nmrom_q{q}_field_{st}', q=q, opt=dict(init='field', tolerance=1e-4, cholesky=True, **({'stepping': 'exact_direct'} if st == 'direct_tol1e-4_chol' else {})))
        for q in sorted({0, qa}) for st in ('cn_tol1e-4_chol', 'direct_tol1e-4_chol')]
cfg = json.loads((HERE / 'configs' / 'final01.json').read_text())
cfg['model'] = dict(kind='mlp', bank=f'vp_R{R}/bank.pkl', head=f'vp_R{R}/head_K{K}.pkl'); cfg['rom_arms'] = arms
(HERE / 'configs' / 'final01b.json').write_text(json.dumps(cfg, indent=1) + '\n')
(HERE / 'selection_addendum1.json').write_text(json.dumps(dict(rule='addendum 1 rules 1-3 with field_cn_tol1e-4_chol', chosen=dict(R=R, K=K, accurate_q=qa, fast_q=0),
                                                             candidates=cands, final_arms=[a['name'] for a in arms]), indent=1) + '\n')
print(dict(R=R, K=K, accurate_q=qa), [a['name'] for a in arms])
