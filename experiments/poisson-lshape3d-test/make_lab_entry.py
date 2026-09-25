"""poisson-lshape3d-test: print the LAB-LOG entry's numbers table from reports/summary.json (no hand-typed numbers).

    python make_lab_entry.py > <tmp>
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / 'reports' / 'summary.json').read_bytes()
S = json.loads(raw)


def p(x):
    return '—' if x is None else (f'{x:.3f}' if x < 1 else f'{x:.2f}')


def x(v):
    return '—' if v is None else (f'{v:.0f}' if v >= 100 else f'{v:.1f}')


print(f"summary.json sha256 `{hashlib.sha256(raw).hexdigest()}`. GPU-query scope; worst same-grid error %; speedup vs "
      "the row's FOM (fastest tested CG at least as accurate as the accurate arm, same job).\n")
print("| problem | mesh | cohort | accurate % | acc × | fast % | fast × | FOM (worst %, ms) | acc/fast ms | job | GPU | gates+audit |")
print("|---|---|---|---:|---:|---:|---:|---|---|---|---|---|")
for r in S['rows']:
    m = f"{r['intervals']}{'²' if r['problem'] == 'lshape' else '³'}"
    for c in ('development', 'test'):
        e = r[c]
        if e is None:
            print(f"| {r['problem']} | {m} | test | NOT RUN/COLLECTED | | | | | | | | |")
            continue
        a, f, fom = e['row']['arms']['accurate'], e['row']['arms']['fast'], e['row']['fom']
        print(f"| {r['problem']} | {m} | {c} ({e['cases']}) | {p(a['worst_pct'])} | {x(a['speedup'])} | {p(f['worst_pct'])} | "
              f"{x(f['speedup'])} | CG {fom['tolerance']:g} ({p(fom['worst_pct'])}, {fom['ms']:.2f}) | {a['ms']:.3f}/{f['ms']:.3f} | "
              f"{e['job_id']} | {e['gpu']} | {'PASS' if e['usable'] else 'FAIL'} |")
        if 'also:R32_linear' in e['row']['arms']:
            b = e['row']['arms']['also:R32_linear']
            print(f"| {r['problem']} | {m} | {c}: span R'=32 | | | {p(b['worst_pct'])} | {x(b['speedup'])} | | {b['ms']:.3f} | | | |")
