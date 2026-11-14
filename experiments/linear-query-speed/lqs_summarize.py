"""linear-query-speed: tables from the saved result JSON of one job (no number is typed by hand).

usage: lqs_summarize.py <results dir of a job> <gpu key in comparators.json> <out.md>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p):
    return json.loads(Path(p).read_text())


def find_key(d, mesh):
    for k in (str(mesh), f'{mesh}^2', f'{mesh}x{mesh}'):
        if k in d:
            return d[k]
    return None


def f(x, nd=3):
    return '-' if x is None else (f'{x:.{nd}f}' if abs(x) < 100 else f'{x:.1f}')


def main():
    res, gpu, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    comp = load(HERE / 'comparators.json')
    frozen = load(res / 'frozen.json') if (res / 'frozen.json').exists() else None
    L = [f'# Poisson 2D query time, GPU type `{gpu}` (job results `{res.name}`)', '']
    if frozen:
        L += [f"Frozen from development cases only: F = `{frozen['F']['variant']}`, "
              f"S = `{(frozen.get('S') or {}).get('variant')}` "
              f"(selection meshes {frozen.get('selection_meshes')}, timing mesh {frozen.get('timing_mesh')}).", '']
    meshes = sorted(int(p.name.split('-')[-1]) for p in res.glob('p2d-*') if (p / 'COMPLETE').exists())
    rows_main, rows_stage, rows_gate, notes = [], [], [], []
    for n in meshes:
        R = load(res / f'p2d-{n}' / 'result.json')
        S = {s['name']: s for s in R['summary']}
        sub = {s['name']: s for s in R['subjects']}
        prof = R.get('profile_median_ms', {})
        stored = None
        sp = HERE / 'paper_reference' / f'p2t{n}-stored.json'
        if sp.exists():
            stored = load(sp)
        cg = comp['poisson2d'].get(gpu, {})
        cud = find_key(cg.get('cudss_solve', {}), n)
        amg = find_key(cg.get('amgx_solve_1e-6', {}), n)
        rg = R.get('repro_gate') or {}
        rows_gate.append(f"| {n}^2 | {R.get('nvidia_smi', '').split(',')[0]} | {R.get('job_id')} | "
                         f"{rg.get('stored_available')} | {rg.get('passed')} | "
                         f"{max([r.get('max_abs_difference') or 0 for r in rg.get('rows', [])], default=None)} | "
                         f"{all(r.get('passes', True) for r in R['dev_gate']['rows'])} |")
        for arm in R['arms']:
            base = arm['name']
            P = S[base]
            var = [s for s in R['summary'] if s['kind'] == 'variant' and s['name'].startswith(base + '__')]
            so = [s for s in R['summary'] if s['kind'] == 'solve_only' and s['name'].startswith(base + '__')]
            rows_main.append(
                f"| {n}^2 | {arm['label']} | paper path | {P['worst_error']:.6f} | {P['mean_error']:.6f} | 0 | "
                f"{f(P['median_fused_device_ms'])} | {f(P.get('A1_ms'))}/{f(P.get('A2_ms'))} | "
                f"{f(cud)} | {f(amg)} | {f(P['median_fused_device_ms'] / cud if cud else None, 2)} |")
            pp = prof.get(base, {})
            rows_stage.append(f"| {n}^2 | {arm['label']} | paper path | "
                              f"{f(pp.get('project_and_start'))} | - | {f(pp.get('reconstruction'))} | - |")
            for v in var:
                tag = v['variant']
                role = []
                if frozen and tag == frozen['F']['variant']:
                    role.append('F')
                if frozen and frozen.get('S') and tag == frozen['S']['variant'] and 'sub' in tag:
                    role.append('S')
                info = sub[v['name']].get('info', {})
                rows_main.append(
                    f"| {n}^2 | {arm['label']} | {tag}{' [' + ','.join(role) + ']' if role else ''} "
                    f"(p={info.get('p')}, q={info.get('q')}) | {v['worst_error']:.6f} | {v['mean_error']:.6f} | "
                    f"{v['max_deviation_from_paper']:.1e} | {f(v['median_fused_device_ms'])} | - | "
                    f"{f(cud)} | {f(amg)} | {f(v['median_fused_device_ms'] / cud if cud else None, 2)} |")
                pv = prof.get(v['name'], {})
                rows_stage.append(f"| {n}^2 | {arm['label']} | {tag} | {f(pv.get('project_and_solve'))} | "
                                  f"{f(pv.get('core_contraction'))} | {f(pv.get('field_products'))} | "
                                  f"{f(v.get('median_input_ms'))}/{f(v.get('median_output_ms'))} |")
            for v in so:
                rows_main.append(f"| {n}^2 | {arm['label']} | SOLVE ONLY, no field ({v['name'].split('__')[1]}) | "
                                 f"- | - | - | {f(v['median_fused_device_ms'])} | - | {f(cud)} | {f(amg)} | "
                                 f"not comparable |")
            if stored:
                sa = stored['arms'][base]
                notes.append(f"- {n}^2 {arm['label']}: stored paper run (job {stored['job_id']}, A100 80GB): "
                             f"{json.dumps({k: v for k, v in sa.items() if not isinstance(v, (list, dict))})}")
        c = R.get('compression', {})
        notes.append(f"- {n}^2 offline: bank build + compression {c.get('seconds')} s for the compression; "
                     f"range-finder residual {c.get('range_finder_relative_residual')}; "
                     f"device memory {R.get('device_memory')}")
    L += ['## Query time and error (test cohort, 32 cases; GPU-query = device f64 source -> device f64 full field)',
          '',
          '| mesh | arm | path | worst err | mean err | max field dev. from paper path | ms (median) | '
          'paper A1/A2 ms | cuDSS ms | AmgX ms | ours / cuDSS |', '|' + '---|' * 11] + rows_main + ['']
    L += ['## Stage medians, ms (separately jitted and synced; they do not sum to the fused time)', '',
          '| mesh | arm | path | projection + solve | core contraction | reconstruction / field products | '
          'host->device / device->host copy |', '|' + '---|' * 7] + rows_stage + ['']
    L += ['## Gates', '', '| mesh | GPU | job | stored value exists | repro gate | max abs diff | dev gate (all) |',
          '|' + '---|' * 7] + rows_gate + ['']
    L += ['## Notes', ''] + notes + ['']
    out.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
