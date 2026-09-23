"""Numbers for the LAB-LOG entry, generated from runs/<panel>/summary.json (never hand-typed)."""
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent.parent
PANELS = [('pn1024b', 1024), ('pn2048d', 2048), ('pn4096d', 4096)]
KEY = ['nmrom_q0_cn', 'nmrom_q32_cn', 'nmrom_q0_field_direct_tol1e-4_chol', 'nmrom_q32_field_direct_tol1e-4_chol',
       'linear_bank_moments_BASELINE', 'linear_bank_moments_cn_BASELINE', 'pod32_galerkin_cn', 'pod128_galerkin_cn', 'pod128_lspg_cn',
       'pod128_galerkin_exact', 'qm16_field_direct_tol1e-4_chol', 'qm32_cn', 'qm32_cn_noautotune',
       'op_fno', 'op_unet', 'op_transolver', 'op_deeponet', 'dst_exact_CONTROL', 'coarse64_fom_cncg_dt0.05_rtol1e-3_CONTROL']
out = []
for job, n in PANELS:
    p = LANE / 'runs' / job / 'summary.json'
    if not p.exists():
        continue
    s = json.loads(p.read_text()); rows = {x['method']: x for x in s['rows']}
    out.append(f"**{n}²** (`{job}`, job {s['metadata']['job_id']}, {s['metadata']['gpu'].split(',')[0]}, status {s['status']}): " + '; '.join(
        f"`{k}` {100*rows[k]['worst_all_times']:.4f} % / {rows[k]['device_ms_median']:.2f} ms"
        + (f" / {rows[k]['speedup']:.1f}×{'†' if rows[k].get('no_fom_as_accurate') else ''} vs `{rows[k]['fom_chosen']}`" if rows[k].get('speedup') else '')
        for k in KEY if k in rows) + '.')
print('\n\n'.join(out))
