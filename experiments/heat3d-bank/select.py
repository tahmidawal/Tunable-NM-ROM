"""Applies the pre-registered selection rule (DESIGN.md) to VALIDATION summaries only and writes configs/final01.json +
selection.json. Usage: select.py <run>/<val panel dir> [...]   (each dir holds summary.json; training.json next to the run)."""
import json, re, sys
from pathlib import Path
import make_configs as MC

GATE, BAR = 0.006, 0.008
HERE = Path(__file__).resolve().parent
cands, log = [], []
for d in (Path(a).resolve() for a in sys.argv[1:]):
    s = json.loads((d / 'summary.json').read_text()); m = re.match(r'val_vp_R(\d+)_K(\d+)', d.name); R, K = int(m[1]), int(m[2])
    tr = json.loads(next(d.parent.glob(f'trained_vp_R{R}-training.json')).read_text())
    floor = {f['intervals']: f['worst'] for f in tr['bank']['validation_floor_other_grids']}
    assert all(mesh['cohort'] == 'validation' for mesh in s['meshes']), 'selection must read validation summaries only'
    rows = {mesh['intervals']: {r['method']: r for r in mesh['rows']} for mesh in s['meshes']}
    entry = dict(R=R, K=K, run=str(d.relative_to(HERE)), bank_floor=floor, gate=floor[128] <= GATE, accurate_q=None)
    ladder = MC.LADDER[R]['ladder']
    def ok(name): return all(name in rows[n] and rows[n][name]['error_all_times_worst'] <= BAR and rows[n][name]['failures'] == 0 for n in (64, 128))
    for q in ladder:
        if q + K > R: continue
        if ok(f'nmrom_q{q}_field_cn'):
            entry['accurate_q'] = q; entry['ms_128'] = rows[128][f'nmrom_q{q}_field_cn']['device_ms_median']; break
    entry['errors_128_cn'] = {q: rows[128][f'nmrom_q{q}_field_cn']['error_all_times_worst'] for q in ladder if f'nmrom_q{q}_field_cn' in rows[128]}
    entry['top_q'] = max(q for q in ladder if q + K <= R); entry['ok_variants'] = [v for v in ('cn_dt0.05', 'cn_dt0.1') if entry['accurate_q'] is not None and ok(f"nmrom_q{entry['accurate_q']}_field_{v}")]
    log.append(entry); cands.append(entry)
passing = [c for c in cands if c['gate'] and c['accurate_q'] is not None]
if passing:
    best = min(c['ms_128'] for c in passing)
    tied = [c for c in passing if c['ms_128'] <= 1.05 * best]; chosen = min(tied, key=lambda c: (c['R'], c['K'])); rule = 'rule 1-3 (gate + <=0.8% + fastest)'
else:
    gated = [c for c in cands if c['gate']]
    if not gated:   # stop rule: no bank passes the 0.6 % floor gate -> the sealed cohorts stay closed (Codex finding 1)
        (HERE / 'selection.json').write_text(json.dumps(dict(rule='STOP: no bank passed the floor gate; no final config written', candidates=log), indent=1) + '\n')
        sys.exit('STOP: no bank passed the 0.6 % floor gate; sealed cohorts stay closed')
    chosen = min(gated, key=lambda c: min(c['errors_128_cn'].values())); chosen['accurate_q'] = min(chosen['errors_128_cn'], key=chosen['errors_128_cn'].get)
    rule = 'rule 5 (no gated candidate reached 0.8%: lowest validation error among gated banks)'
R, K, qa, qt = chosen['R'], chosen['K'], chosen['accurate_q'], chosen['top_q']
arms = [a for a in MC.rom_arms(sorted({0, qa, qt}), [qa]) if (a['q'] in (0, qa, qt) and ('_dt' not in a['name'] or any(a['name'].endswith(v) for v in chosen['ok_variants'])))]
cfg = MC.panel(f'vp_R{R}', K, [], [], [('sealed_921099_never_opened', 921099, 64), ('paper_benchmark_920399_repeated', 920399, 64)], [32, 64, 128], 128, 128, 128, 32)   # 31^3 audit sub-grid (addendum 1)
cfg['rom_arms'] = arms
(HERE / 'configs' / 'final01.json').write_text(json.dumps(cfg, indent=1) + '\n')
(HERE / 'selection.json').write_text(json.dumps(dict(rule=rule, chosen=dict(R=R, K=K, accurate_q=qa, top_q=qt, fast_q=0), candidates=log, final_arms=[a['name'] for a in arms]), indent=1) + '\n')
print(rule, dict(R=R, K=K, accurate_q=qa, top_q=qt), [a['name'] for a in arms])
