"""Addendum-2 selection (DESIGN.md), VALIDATION summary only. Usage: select_speed2.py runs/speed2/speed2_vp_R256_K16
Rewrites configs/final01b.json (panel B) and writes selection_addendum2.json."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; d = Path(sys.argv[1]).resolve(); s = json.loads((d / 'summary.json').read_text())
assert all(m['cohort'] == 'validation' for m in s['meshes'])
rows = {m['intervals']: {r['method']: r for r in m['rows']} for m in s['meshes']}
cfg_sp = {a['name']: a for a in json.loads((HERE / 'configs' / 'speed2_vp_R256_K16.json').read_text())['rom_arms']}
out = {}
for fam, ref in (('cn', 'nmrom_q160_field_cn_tol1e-4_chol'), ('direct', 'nmrom_q160_field_direct_tol1e-4_chol')):
    cap = {n: min(0.008, 1.05 * rows[n][ref]['error_all_times_worst']) for n in (64, 128)}
    names = [k for k in rows[128] if k.startswith('nmrom_q160_') and (('_cn_' in k) == (fam == 'cn'))]
    elig = [k for k in names if all(rows[n][k]['error_all_times_worst'] <= cap[n] and rows[n][k]['failures'] == 0 for n in (64, 128))]
    if rows[64][ref]['failures'] == 0 and rows[128][ref]['failures'] == 0 and ref not in elig: elig.append(ref)
    table = {k: dict(err64=rows[64][k]['error_all_times_worst'], err128=rows[128][k]['error_all_times_worst'], fail=[rows[n][k]['failures'] for n in (64, 128)],
                     ms128=rows[128][k]['device_ms_median'], eligible=k in elig) for k in names}
    win = min(elig, key=lambda k: rows[128][k]['device_ms_median']) if elig else ref
    out[fam] = dict(winner=win, reference=ref, table=table)
arms = [cfg_sp[out[f]['winner'].replace('q160', f'q{q}')] for q in (0, 160) for f in ('cn', 'direct')]
cfg = json.loads((HERE / 'configs' / 'final01b.json').read_text()); cfg['rom_arms'] = arms
(HERE / 'configs' / 'final01b.json').write_text(json.dumps(cfg, indent=1) + '\n')
(HERE / 'selection_addendum2.json').write_text(json.dumps(dict(rule='addendum 2', families=out, final_b_arms=[a['name'] for a in arms]), indent=1) + '\n')
print({f: out[f]['winner'] for f in out}); print([a['name'] for a in arms])
