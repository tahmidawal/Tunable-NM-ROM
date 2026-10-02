"""Cross-mesh bars (DESIGN 8: B2 mesh invariance; B4 flat cost from the 4096^2 job's same-GPU panel) and the pre-registered recommended off-mesh rule per
setting, from the development/validation summaries only (checks/dv256|dv1024|dv4096-summary.json).

    python select_rule.py --tag dv --out checks/selection-dv.json

Recommended rule := the smallest-m Gauss or Fibonacci rollout arm (point form) that passes B1 (where dense ran), B3 and
B5 at all three meshes; None if no arm passes. Run once, before any test job; its output is frozen in DESIGN.md.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESHES = (256, 1024, 4096)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='dv')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    S = {L: json.loads((HERE / f'checks/{a.tag}{L}-summary.json').read_text()) for L in MESHES}
    out = dict(source={L: dict(attempt=S[L]['attempt'], job_id=S[L]['job_id'], result_sha256=S[L]['result_sha256'],
                               failed_gates=S[L]['failed_gates']) for L in MESHES},
               B2={}, B4={}, per_arm={}, recommended={})
    for s in ('acc', 'fast', 'head'):
        names = sorted(set.intersection(*[set(S[L]['arms'].get(s, {})) for L in MESHES]))
        for name in names:
            ent = {L: S[L]['arms'][s][name] for L in MESHES}
            st = [ent[L]['all']['ref_ST_evolved']['worst'] if ent[L]['all'].get('ref_ST_evolved') else None for L in MESHES]
            b2 = None
            if all(v is not None for v in st):
                b2 = dict(worst_ST=dict(zip(MESHES, st)), ratio=max(st) / min(st), passed=bool(max(st) / min(st) <= 1.02))
            # B4 from the same-GPU cross-mesh panel of the 4096^2 job (Codex design audit finding 3)
            b4 = S[4096]['timing'].get('B4_same_gpu', {}).get(f'{s}|{name}')
            rho = {}
            rule = ent[MESHES[0]]['rule']
            for L in MESHES:
                rr = S[L]['rho'].get(s, {}).get('rules', {})
                key = name if name in rr else rule
                rho[L] = rr.get(key, {}).get('cont', {}).get('max')
            b5 = all(v is not None and v <= .116 for v in rho.values())
            b1 = [ent[L].get('B1', {}).get('passed') for L in MESHES]
            b3 = [ent[L].get('B3', {}).get('passed') for L in MESHES]
            out['B2'].setdefault(s, {})[name] = b2
            out['B4'].setdefault(s, {})[name] = b4
            out['per_arm'].setdefault(s, {})[name] = dict(kind=ent[MESHES[0]]['kind'], rule=rule, m=ent[MESHES[0]]['m'],
                                                          B1=b1, B3=b3, B5=b5, rho_cont_max=rho)
        cands = [(v['m'], n) for n, v in out['per_arm'][s].items()
                 if v['kind'] == 'point' and (v['rule'] or '').startswith(('gauss', 'fib')) and not n.startswith('ctrl')
                 and n != 'gref' and all(x is not False for x in v['B1']) and all(v['B3']) and v['B5']]
        out['recommended'][s] = min(cands)[1] if cands else None
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['recommended']))


if __name__ == '__main__':
    main()
