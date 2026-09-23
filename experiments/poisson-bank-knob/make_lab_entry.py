"""Print the generated tables used in the lab-log entry and final report (from reports/summary.json only)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / 'reports' / 'summary.json'
S = json.loads(P.read_text())
PRIMARY = ('linear-rung', 'nm-rom')


def main():
    print(f"summary.json sha256 {hashlib.sha256(P.read_bytes()).hexdigest()}")
    print()
    print('**Step 0 — stage split (median ms, separately jitted stages; fused GPU-query and host copies alongside):**')
    print()
    print('| mesh | arm | project+start | LM solve | y elim | reconstruction u=Gc | fused GPU | host copies | dominant |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---|')
    for m in S['meshes']:
        for nm in ('R512_q256', 'R512_q0', 'R512_linear', 'R128_linear'):
            p = m['profile'][nm]
            print(f"| {m['intervals']}² | `{nm}` | {p['project_and_start']:.2f} | {p['lm_solve']:.2f} | {p['y_elimination_and_map']:.2f} | "
                  f"{p['reconstruction']:.2f} | {p['fused_gpu_ms']:.2f} | {p['host_copies_ms']:.2f} | {p['dominant']} |")
    print()
    print('**Table-1 settings (pre-registered rule) per mesh:**')
    print()
    print('| mesh | GPU | job | accurate arm | err % | GPU ms | fast arm | err % | GPU ms | FOM (CG) | FOM err % | FOM ms | × accurate | × fast | gates |')
    print('|---|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|')
    for m in S['meshes']:
        t = m['table1']
        bad = [k for k, v in m['gates'].items() if not v]
        print(f"| {m['intervals']}² | {m['gpu']} | {m['job_id']} | `{t['accurate']}` | {t['accurate_worst_error']*100:.3f} | {t['accurate_gpu_ms']:.2f} | "
              f"`{t['fast']}` | {t['fast_worst_error']*100:.3f} | {t['fast_gpu_ms']:.2f} | `{t['fom']}` | {t['fom_worst_error']*100:.3f} | "
              f"{t['fom_gpu_ms']:.1f} | {t['accurate_speedup']:.1f} | {t['fast_speedup']:.1f} | {'all pass' if not bad else 'FAIL: ' + ', '.join(bad)} |")
    print()
    for fam in ('linear-rung', 'nm-rom'):
        title = {'linear-rung': "bank-span linear rung q=R'", 'nm-rom': 'head-only q=0'}[fam]
        print(f"**R' ladder — {title} (worst err % / floor % / GPU ms / × vs own matched CG):**")
        print()
        meshes = S['meshes']
        print('| R\' | ' + ' | '.join(f"{m['intervals']}²" for m in meshes) + ' |')
        print('|---:|' + '---|' * len(meshes))
        for Rp in (512, 384, 256, 128, 64, 32):
            cells = []
            for m in meshes:
                s = next(x for x in m['subjects'].values() if x['family'] == fam and x['Rp'] == Rp and (fam == 'linear-rung' or x['q'] == 0))
                cells.append(f"{s['worst_error']*100:.3f} / {s['floor']*100:.3f} / {s['gpu_ms']:.2f} / {s['speedup_vs_own_matched_cg']:.0f}× (`{s['own_matched_cg']}`)")
            print(f"| {Rp} | " + ' | '.join(cells) + ' |')
        print()
    print('**Verdict inputs:**')
    for m in S['meshes']:
        v = S['verdict'][str(m['intervals'])]
        print(f"- {m['intervals']}²: " + '; '.join(f"{k}: monotone err {x['monotone_error']}, monotone cost {x['cost_monotone']}, R'=512/R'=32 cost {x['cost_range']:.2f}×"
                                                   for k, x in v.items()))
    print()
    print('**Gates:**')
    for m in S['meshes']:
        g = m['gate_breakdown']
        print(f"- {m['intervals']}² ({m['design']}): {m['gates']}; audit {m['audit']['verdict']}; parity "
              f"{', '.join(f'{p['worst_field_relative']:.1e}' for p in m['parity']) or 'n/a'}; primary-arm neighbour worst {g['neighbour_primary_worst']:.3f}, "
              f"failing primary {g['neighbour_primary_failing'] or 'none'}, failing reference {g['neighbour_other_failing'] or 'none'}, drift failing {g['drift_failing'] or 'none'}")
    print()
    for m in S['meshes']:
        print(f"- {m['intervals']}²: {m['attempt']}/{m['output']} result.json sha256 {m['result_sha256']}, commit {m['commit']}, GPU {m['gpu']} {m['gpu_uuid']}")


if __name__ == '__main__':
    main()
