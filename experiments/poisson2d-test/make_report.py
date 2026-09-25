"""Generate reports/summary.json + the test-vs-development report from the pulled cluster JSONs.

No hand-typed numbers: every value comes from runs/<attempt>/archive/output/{result.json,audit.json}
(sha256 recorded) and from the parent lane's reports/summary.json (development rows, sha256 checked).
Rules are DESIGN.md's, frozen before any test job: accurate = R512_linear, fast = R128_linear,
FOM = fastest tested CG whose worst test error <= the accurate arm's worst test error, same job.

    /home/tahmid/Dev/.venv/bin/python make_report.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PARENT = EXP / 'poisson-bank-knob' / 'reports' / 'summary.json'
PARENT_SHA = 'f19d01721a0ba999518eeb3f05ee49ac4f0c10f52ca43313d35ce4c08940f67b'
RUNS = ['p2t256', 'p2t1024', 'p2t2048', 'p2t4096']
ACC, FAST, HEADREF = 'R512_linear', 'R128_linear', 'R512_q0'
TEST_SEED, TEST_COUNT = 2026092501, 32
REPORT = '2026-09-25-poisson2d-test-vs-dev.md'
LIM = 1.10


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def test_params():
    """Recompute the cohort from the sampler itself (not from the job) to check the job used the frozen seed."""
    sys.path.insert(0, str(EXP / 'wave2d-rom-latent-stepping' / 'deps' / 'multistage-precision'))
    import ms_parametric as mp
    cx, cy, w, a, _ = mp.sample_params(TEST_SEED, TEST_COUNT)
    return np.column_stack((cx, cy, w, a))


def one(att, params_expected):
    d = HERE / 'runs' / att / 'archive' / 'output'
    if not (d / 'result.json').exists():
        return dict(attempt=att, missing=True)
    R = json.loads((d / 'result.json').read_text())
    A = json.loads((d / 'audit.json').read_text())
    logs = sorted((HERE / 'runs' / att / 'archive' / 'logs').glob('*.out'))
    logtext = ''.join(p.read_text() for p in logs)
    checks = dict(
        complete=bool(R['complete']), backend_gpu=R['backend'] == 'gpu', x64=bool(R['x64']),
        precision_highest=R['matmul_precision'] == 'highest', not_smoke=not R['smoke'],
        log_has_jax_backend_gpu='jax_backend=gpu' in logtext, log_all_done='ALL-DONE' in logtext,
        cohort_seed=R['cohort']['seed'] == TEST_SEED and R['cohort']['count'] == TEST_COUNT,
        cohort_params_match_sampler=bool(np.array_equal(np.asarray(R['cohort']['parameters']), params_expected)),
        audit_pass=A['verdict'] == 'PASS',
        commit_matches_staged=(R['commit'] == (HERE / 'runs' / att / 'COMMIT.txt').read_text().strip()),
        a100_80gb=('A100' in R['gpu'] and '80GB' in R['gpu']))
    n = R['intervals']
    inv = R['invocations'] + R['slow_invocations']
    subj = {}
    for nm in dict.fromkeys(x['name'] for x in inv):
        rows = [x for x in inv if x['name'] == nm]
        errs = {x['case']: x['same_grid_error'] for x in rows}
        subj[nm] = dict(name=nm, family=rows[0]['family'], Rp=rows[0].get('Rp'), q=rows[0].get('q'),
                        tolerance=rows[0].get('tolerance'), worst_error=max(errs.values()),
                        median_error=med(list(errs.values())), cases=len(errs),
                        gpu_ms=med([x['fused_device_seconds'] for x in rows]) * 1e3,
                        total_ms=med([x['total_seconds'] for x in rows]) * 1e3,
                        gpu_ms_A1=(med([x['fused_device_seconds'] for x in rows if x['phase'] == 'romA1']) or 0) * 1e3,
                        gpu_ms_A2=(med([x['fused_device_seconds'] for x in rows if x['phase'] == 'romA2']) or 0) * 1e3,
                        samples=len(rows),
                        cg_iterations=med([x['iterations'] for x in rows if x.get('iterations') is not None]),
                        cg_iterations_max=max([x['iterations'] for x in rows if x.get('iterations') is not None] or [0]))
    cgs = sorted([s for s in subj.values() if s['family'] == 'cg'], key=lambda s: s['gpu_ms'])
    acc, fast = subj[ACC], subj[FAST]
    fom = next((c for c in cgs if c['worst_error'] <= acc['worst_error']), None)
    for s in subj.values():
        if s['family'] != 'cg':
            s['floor'] = R['floors'][str(s['Rp'])]['worst']
            own = next((c for c in cgs if c['worst_error'] <= s['worst_error']), None)
            s['own_matched_cg'] = own['name'] if own else None
            s['speedup_vs_own_matched_cg'] = own['gpu_ms'] / s['gpu_ms'] if own else None
            s['speedup_vs_table1_fom'] = fom['gpu_ms'] / s['gpu_ms'] if fom else None
    t1 = dict(accurate=ACC, fast=FAST, fom=fom['name'] if fom else None,
              accurate_worst_error=acc['worst_error'], accurate_median_error=acc['median_error'],
              accurate_gpu_ms=acc['gpu_ms'], accurate_total_ms=acc['total_ms'],
              fast_worst_error=fast['worst_error'], fast_median_error=fast['median_error'],
              fast_gpu_ms=fast['gpu_ms'], fast_total_ms=fast['total_ms'],
              fom_worst_error=fom['worst_error'] if fom else None, fom_gpu_ms=fom['gpu_ms'] if fom else None,
              fom_total_ms=fom['total_ms'] if fom else None,
              accurate_speedup=fom['gpu_ms'] / acc['gpu_ms'] if fom else None,
              fast_speedup=fom['gpu_ms'] / fast['gpu_ms'] if fom else None,
              accurate_speedup_total=fom['total_ms'] / acc['total_ms'] if fom else None,
              fast_speedup_total=fom['total_ms'] / fast['total_ms'] if fom else None,
              head_reference_worst_error=subj[HEADREF]['worst_error'])
    # informational only (NOT used for the row): what the parent's selection rule would pick on test
    cand = [s for s in subj.values() if s['family'] in ('nm-rom', 'linear-rung') and not (s['family'] == 'nm-rom' and s['q'] > 0)]
    rule_acc = min(cand, key=lambda s: (s['worst_error'], s['gpu_ms']))
    rule_fast = min([s for s in cand if s['worst_error'] <= subj[HEADREF]['worst_error']], key=lambda s: s['gpu_ms'])
    primary = {ACC, FAST}
    ng, dg = R['neighbour_gate'], R['drift_gate']
    gb = dict(
        neighbour_table1_arms_worst=max([r['ratio'] for r in ng['rows'] if r['name'] in primary] or [float('nan')]),
        neighbour_fom_worst=max([r['ratio'] for r in ng['rows'] if fom and r['name'] == fom['name']] or [float('nan')]),
        neighbour_failing=[f"{r['name']}@{r['phase']} {r['ratio']:.3f}" for r in ng['rows'] if r['ratio'] > LIM],
        drift_table1_arms=[f"{r['name']} {r['ratio']:.3f}" for r in dg['rows'] if r['name'] in primary],
        drift_failing=[f"{r['name']} {r['ratio']:.3f}" for r in dg['rows'] if not (1 / LIM <= r['ratio'] <= LIM)])
    t1_gate_clean = (not any(x.split('@')[0] in primary or (fom and x.split('@')[0] == fom['name']) for x in gb['neighbour_failing'])
                     and not any(x.split(' ')[0] in primary for x in gb['drift_failing']))
    return dict(attempt=att, intervals=n, job_id=R['job_id'], gpu=R['gpu'], gpu_uuid=R['gpu_uuid'], commit=R['commit'],
                host=next((l.split()[0].split('=')[1] for l in logtext.splitlines() if l.startswith('host=')), None),
                checks=checks, gates=R['gates'], parity=R['parity'], audit=dict(verdict=A['verdict'], summary=A['summary']),
                gate_breakdown=gb, table1_arms_gate_clean=bool(t1_gate_clean),
                cohort=dict(seed=R['cohort']['seed'], count=R['cohort']['count'], sha256=R['cohort']['sha256'],
                            closest_linf_to_other_cohorts=R['cohort']['closest_linf_to_other_cohorts']),
                floors={k: v['worst'] for k, v in R['floors'].items()}, table1=t1,
                parent_rule_on_test=dict(accurate=rule_acc['name'], fast=rule_fast['name'],
                                         note='informational; the row uses the frozen DESIGN settings'),
                subjects=subj, device_memory=R.get('device_memory'), elapsed_seconds=R['elapsed_seconds'],
                bank_contraction_sha256=R['bank']['contraction_sha256'],
                result_sha256=sha(d / 'result.json'), audit_sha256=sha(d / 'audit.json'),
                outputs_sha256_file=sha(HERE / 'runs' / att / 'archive' / 'OUTPUTS.sha256'))


def dev_rows():
    assert sha(PARENT) == PARENT_SHA, 'parent summary.json changed'
    P = json.loads(PARENT.read_text())
    out = {}
    for m in P['meshes']:
        t = m['table1']
        assert t['accurate'] == ACC and t['fast'] == FAST, t
        out[m['intervals']] = dict(job_id=m['job_id'], gpu=m['gpu'], fom=t['fom'],
                                   accurate_worst_error=t['accurate_worst_error'], accurate_speedup=t['accurate_speedup'],
                                   fast_worst_error=t['fast_worst_error'], fast_speedup=t['fast_speedup'],
                                   fom_worst_error=t['fom_worst_error'], accurate_gpu_ms=t['accurate_gpu_ms'],
                                   fast_gpu_ms=t['fast_gpu_ms'], fom_gpu_ms=t['fom_gpu_ms'],
                                   bank_contraction_sha256=None, cases=12, result_sha256=m['result_sha256'])
    return out


def pct(x, d=2):
    return '—' if x is None else f'{x * 100:.{d}f}'


def sx(x):
    if x is None:
        return '—'
    return f'{x:.0f}×' if x >= 100 else f'{x:.1f}×'


def main():
    expected = test_params()
    meshes = [one(a, expected) for a in RUNS]
    dev = dev_rows()
    ok = [m for m in meshes if not m.get('missing')]
    for m in ok:
        dv = dev[m['intervals']]
        t = m['table1']
        m['development'] = dv
        m['test_over_dev'] = dict(
            accurate_error=t['accurate_worst_error'] / dv['accurate_worst_error'],
            fast_error=t['fast_worst_error'] / dv['fast_worst_error'],
            accurate_speedup=(t['accurate_speedup'] / dv['accurate_speedup']) if t['accurate_speedup'] else None,
            fast_speedup=(t['fast_speedup'] / dv['fast_speedup']) if t['fast_speedup'] else None,
            accurate_gpu_ms=t['accurate_gpu_ms'] / dv['accurate_gpu_ms'], fast_gpu_ms=t['fast_gpu_ms'] / dv['fast_gpu_ms'])
        m['row_status'] = ('VOID' if not all(m['checks'].values()) else
                           'final' if m['table1_arms_gate_clean'] else 'provisional (timing gate)')
    S = dict(description='Poisson 2D Table-1 settings on the held-out TEST cohort vs the development cohort',
             design='experiments/poisson2d-test/DESIGN.md', test_seed=TEST_SEED, test_count=TEST_COUNT,
             parent_summary=str(PARENT.relative_to(EXP.parent)), parent_summary_sha256=PARENT_SHA,
             settings=dict(accurate=ACC, fast=FAST, fom_rule='fastest tested CG with worst test error <= accurate worst test error, same job'),
             meshes=meshes)
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'reports' / 'summary.json').write_text(json.dumps(S, indent=1) + '\n')
    ssha = sha(HERE / 'reports' / 'summary.json')

    L = ['# Poisson 2D Table-1 settings on held-out test sources, beside the development values', '',
         f'The frozen Poisson 2D model at the paper Table 1 settings (accurate = bank span $R\'=512$, fast = $R\'=128$), '
         f'evaluated once on {TEST_COUNT} held-out test sources (seed {TEST_SEED}) against unpreconditioned CG timed in the '
         f'same job, next to the 12-source development values from `poisson-bank-knob`. Numbers are final unless a row is '
         f'marked provisional; generated by `experiments/poisson2d-test/make_report.py` from `reports/summary.json` '
         f'(sha256 `{ssha}`). Nothing here was tuned on test sources.', '',
         '## Table-1 format, test vs development', '',
         '| mesh | cohort | accurate err % | accurate speedup | fast err % | fast speedup | FOM err % | FOM (CG rtol) | job | status |',
         '|---|---|---:|---:|---:|---:|---:|---|---|---|']
    for m in ok:
        t, dv = m['table1'], m['development']
        n = m['intervals']
        L.append(f"| ${n}^2$ | test ({m['cohort']['count']}) | {pct(t['accurate_worst_error'])} | {sx(t['accurate_speedup'])} | "
                 f"{pct(t['fast_worst_error'])} | {sx(t['fast_speedup'])} | {pct(t['fom_worst_error'])} | `{t['fom']}` | "
                 f"{m['job_id']} | {m['row_status']} |")
        L.append(f"| | development (12) | {pct(dv['accurate_worst_error'])} | {sx(dv['accurate_speedup'])} | "
                 f"{pct(dv['fast_worst_error'])} | {sx(dv['fast_speedup'])} | {pct(dv['fom_worst_error'])} | `{dv['fom']}` | "
                 f"{dv['job_id']} | paper Table 1 |")
    for m in meshes:
        if m.get('missing'):
            L.append(f"| — | test | — | — | — | — | — | — | {m['attempt']} | NOT RUN / NOT PULLED |")
    L += ['', '## Times and ratios (test)', '',
          '| mesh | GPU | accurate ms | fast ms | FOM ms | accurate / fast median err % | complete-query speedup acc / fast | test÷dev: acc err, fast err, acc speedup, fast speedup |',
          '|---|---|---:|---:|---:|---:|---:|---|']
    for m in ok:
        t, r = m['table1'], m['test_over_dev']
        L.append(f"| ${m['intervals']}^2$ | {m['gpu']} ({m['host']}) | {t['accurate_gpu_ms']:.2f} | {t['fast_gpu_ms']:.2f} | "
                 f"{t['fom_gpu_ms']:.1f} | {pct(t['accurate_median_error'])} / {pct(t['fast_median_error'])} | "
                 f"{sx(t['accurate_speedup_total'])} / {sx(t['fast_speedup_total'])} | "
                 f"{r['accurate_error']:.2f}, {r['fast_error']:.2f}, {r['accurate_speedup']:.2f}, {r['fast_speedup']:.2f} |")
    L += ['', "## The $R'$ ladder on test (bank-span linear rung, plus head-only at $R'=512$)", '',
          "| mesh | arm | worst err % | median err % | floor % | GPU ms | own matched CG | × vs own matched CG | × vs Table-1 FOM |",
          '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for m in ok:
        for s in sorted([s for s in m['subjects'].values() if s['family'] in ('linear-rung', 'nm-rom')],
                        key=lambda s: (s['family'] != 'linear-rung', -s['Rp'])):
            L.append(f"| ${m['intervals']}^2$ | `{s['name']}` | {pct(s['worst_error'], 3)} | {pct(s['median_error'], 3)} | "
                     f"{pct(s['floor'], 3)} | {s['gpu_ms']:.2f} | `{s['own_matched_cg']}` | {sx(s['speedup_vs_own_matched_cg'])} | "
                     f"{sx(s['speedup_vs_table1_fom'])} |")
    L += ['', '## CG ladder on test', '', '| mesh | CG rtol | worst err % | GPU ms | median iterations | max iterations |',
          '|---|---:|---:|---:|---:|---:|']
    for m in ok:
        for s in sorted([s for s in m['subjects'].values() if s['family'] == 'cg'], key=lambda s: -s['tolerance']):
            L.append(f"| ${m['intervals']}^2$ | {s['tolerance']:g} | {pct(s['worst_error'], 3)} | {s['gpu_ms']:.1f} | "
                     f"{s['cg_iterations']:.0f} | {s['cg_iterations_max']:.0f} |")
    L += ['', '## Gates, audit and provenance', '']
    for m in ok:
        g = m['gate_breakdown']
        L += [f"- **${m['intervals']}^2$** (`{m['attempt']}`, job {m['job_id']}, {m['gpu']} `{m['gpu_uuid']}`, commit `{m['commit'][:10]}`): "
              f"checks {'all pass' if all(m['checks'].values()) else 'FAIL ' + str([k for k, v in m['checks'].items() if not v])}; "
              f"gates {m['gates']}; audit {m['audit']['verdict']} — {m['audit']['summary']}; "
              f"parity {', '.join(f"{p['candidate']} {p['worst_field_relative']:.1e}" for p in m['parity']) or 'not run (original bank not kept)'}; "
              f"neighbour worst on Table-1 arms {g['neighbour_table1_arms_worst']:.3f}, on the FOM {g['neighbour_fom_worst']:.3f}; "
              f"neighbour failing: {', '.join(g['neighbour_failing']) or 'none'}; drift of Table-1 arms {', '.join(g['drift_table1_arms'])}; "
              f"drift failing: {', '.join(g['drift_failing']) or 'none'}; Table-1 arms gate-clean: {m['table1_arms_gate_clean']}. "
              f"Parent selection rule applied to test (informational): accurate `{m['parent_rule_on_test']['accurate']}`, fast `{m['parent_rule_on_test']['fast']}`. "
              f"Cohort sha256 `{m['cohort']['sha256'][:16]}…`, closest $L^\\infty$ distance to any other cohort "
              f"{min(m['cohort']['closest_linf_to_other_cohorts'].values()):.4f}. result.json sha256 `{m['result_sha256']}`; "
              f"audit.json `{m['audit_sha256']}`; development result.json `{m['development']['result_sha256']}`."]
    L += ['', '## Glossary', '',
          '- **accurate / fast setting**: the two operating points of the one frozen model printed in paper Table 1. Accurate = '
          "the bank-span linear rung with all $R'=512$ bank columns; fast = the same with the first $R'=128$ columns of the "
          'rotated (nested) bank.',
          "- **$R'$**: number of bank columns kept at query time (nested truncation of the rotated bank); the cost knob.",
          '- **bank-span linear rung**: the reduced solution is the weak-form least-squares fit in the span of the kept bank columns '
          '(one triangular solve); no head, no latent code.',
          '- **head-only `R512_q0`**: the nonlinear head at the full bank with no correction directions; kept as a reference.',
          '- **err %**: relative $L^2$ error against the exact discrete solution (DST-I solve) on the same mesh; **worst** = maximum over the cohort.',
          '- **floor %**: best possible error in the span of the kept columns (projection of the truth); the linear rung sits on it.',
          '- **FOM**: the full-order comparator, unpreconditioned conjugate gradients (CG) on the same mesh; **FOM (CG rtol)** = the '
          'relative-residual tolerance chosen by the rule "fastest tested CG whose worst error is at most the accurate setting\'s".',
          '- **speedup**: FOM GPU-query time / setting GPU-query time, both measured in the same job on the same GPU.',
          '- **GPU-query ms**: median device time of one query between the host-to-device source copy and the device-to-host field copy; '
          '**complete-query** adds those copies.',
          '- **A–B–A**: timing design — all ROM arms (A1), then all CG tolerances (B), then the ROM arms again (A2).',
          '- **drift gate**: A2 median / A1 median per arm must be within 1.10× either way; **neighbour gate**: median after a slow '
          'predecessor / median after a fast one must be ≤ 1.10.',
          '- **provisional (timing gate)**: a Table-1 arm or the FOM failed a timing gate; the errors stand, the milliseconds carry that caveat.',
          '- **NumPy audit**: an independent re-solve of the truth with a dense sine matrix, re-computing every recorded error and hash, with two negative controls.',
          '- **development cohort**: the 12 sources on which the settings were chosen and paper Table 1 was measured; **test cohort**: '
          f'{TEST_COUNT} fresh sources (seed {TEST_SEED}) from the same distribution, never used for any choice.',
          '- **own matched CG**: the fastest CG at least as accurate as that particular arm (used for the ladder table only).']
    (HERE / 'reports' / REPORT).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
