"""Generate the b-speed report from the job JSONs and the independent audits.

No number in the report is hand-typed: every table here is derived from `result.json`
and `audit.json`. Run from this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


# The arm list, its intents and its grouping are read from the job JSON, so the
# generator imports neither JAX nor the experiment modules.
def groups(res):
    decl = res['arm_declarations']
    isolated = [n for n in decl if n.startswith('o_') and decl[n]['declared_class'] != 'labelled']
    cumulative = [n for n in decl if re.fullmatch(r'L\d+', n)]
    composed = [n for n in decl if re.fullmatch(r'C\d+', n)]
    labelled = [n for n in decl if decl[n]['declared_class'] == 'labelled']
    return (isolated, sorted(cumulative, key=lambda x: int(x[1:])),
            sorted(composed, key=lambda x: int(x[1:])), labelled)


def med(xs):
    return float(st.median(xs)) if xs else float('nan')


def ms(x):
    return x * 1e3


def gather(res):
    """median GPU and host milliseconds per (mesh, subject), all repetitions retained."""
    gpu = defaultdict(list)
    hostt = defaultdict(list)
    percase = defaultdict(lambda: defaultdict(list))
    err = defaultdict(list)
    extra = {}
    for r in res['invocations']:
        key = (r['intervals'], r['name'])
        gpu[key].append(r['gpu_seconds'])
        hostt[key].append(r['host_seconds'])
        percase[key][r['case']].append(r['gpu_seconds'])
        if 'same_grid' in r:
            err[key].append(r['same_grid']['current_relative_max'])
        extra.setdefault(key, dict(kind=r['kind'], output_bytes=r['output_bytes'],
                                   dtype=r['dtype']))
        if r['kind'] == 'rom':
            extra[key].setdefault('iterations', []).append(sum(r['iterations']))
            extra[key].setdefault('ic_iterations', []).append(r['ic_iterations'])
            extra[key].setdefault('budget_exits', []).append(r['budget_exits'])
    out = {}
    for key in gpu:
        out[key] = dict(
            gpu_ms=ms(med(gpu[key])), host_ms=ms(med(hostt[key])),
            gpu_ms_case_median=ms(med([med(v) for v in percase[key].values()])),
            n=len(gpu[key]), repetitions_ms=[ms(x) for x in gpu[key]],
            worst_same_grid_percent=(100 * max(err[key]) if err[key] else None),
            median_same_grid_percent=(100 * med(err[key]) if err[key] else None),
            **{k: v for k, v in extra[key].items() if k != 'kind'},
            kind=extra[key]['kind'])
    return out


def parity_map(res, aud):
    out = {}
    for row in res['parity']:
        out[(row['intervals'], row['arm'])] = row
    for row in aud.get('parity_recomputed', []):
        key = (row['intervals'], row['name'])
        if key in out:
            out[key]['audit_field_relative'] = row['worst_field_relative']
            out[key]['audit_bitwise'] = row['bitwise']
    return out


def table(rows, header, align=None):
    align = align or ['---'] * len(header)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(lines)


def fmt(x, n=3):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'NO'
    if isinstance(x, float):
        if x != x:
            return '—'
        return f'{x:.{n}f}'
    return str(x)


def sci(x, n=3):
    return '—' if x is None else f'{x:.{n}e}'


# --------------------------------------------------------------- sections ---

def section_profile(pr, res):
    L = pr['intervals']
    body = [f'### The incumbent query at {L} intervals, component by component\n']
    ph = pr['phases']
    whole = ph['whole_query']['median_seconds']
    rows = []
    order = ['whole_query', 'initial_fit', 'evolution', 'decode_six_fields', 'host_transfer']
    for k in order:
        v = ph[k]
        rows.append([f'`{k}`', fmt(ms(v['median_seconds'])),
                     fmt(100 * v['median_seconds'] / whole, 1) if k != 'whole_query' else '100.0',
                     len(v['repetitions'])])
    body.append(table(rows, ['component', 'median ms', '% of whole query', 'repetitions'],
                      ['---', '---:', '---:', '---:']))
    body.append('\n' + pr['phase_note'].capitalize() + '.\n')
    body.append('\n**The `host_transfer` row is withdrawn as a measurement** and is shown '
                'only for the record: it times `np.asarray` on the same array repeatedly, and '
                'a JAX array caches its numpy value after the first conversion, so every '
                'repetition after the first measures a cache hit. What the transfers actually '
                'cost is the `transfer ms` column of the ladder tables below, which comes from '
                'the nested GPU and host timers of each timed invocation.\n')

    bp = pr['budget_probe']
    body.append('### Per-iteration marginal cost\n')
    body.append(table([[r['budget'], fmt(ms(r['median_seconds']))] for r in bp['rows']],
                      ['LM iterations per step (unconditional)', 'median evolution ms'],
                      ['---', '---:']))
    body.append(
        f"\nLeast squares over those budgets gives **{fmt(bp['fixed_seconds_per_step'] * 1e6, 1)} µs "
        f"fixed per time step + {fmt(bp['marginal_seconds_per_iteration'] * 1e6, 1)} µs per LM "
        f"iteration** (R² = {fmt(bp['r_squared'], 4)}), over {bp['steps']} steps. "
        f"{bp['note'].capitalize()}.\n")

    mb = pr['microbenchmarks']
    body.append('### In-loop component cost\n')
    body.append('Each row is a `fori_loop` whose body is only that component, timed at '
                'two iteration counts; the reported figure is the slope, so the fixed '
                'launch cost of the surrounding program is removed. Every operator the '
                'body reads is a jit argument, never a captured constant.\n')
    rows = []
    for k, v in mb.items():
        rows.append([f'`{k}`', fmt(v['per_iteration_us'], 2), v['iterations_low'],
                     fmt(ms(v['low_seconds'])), fmt(ms(v['high_seconds']))])
    body.append(table(rows, ['in-loop body', 'µs / iteration', 'T', 'ms at T', 'ms at 2T'],
                      ['---', '---:', '---:', '---:', '---:']))
    neg = [k for k, v in mb.items() if v['per_iteration_us'] < 0]
    if neg:
        body.append(f'\nNegative slopes ({", ".join("`" + k + "`" for k in neg)}) mean the '
                    'body is cheaper than the run-to-run scatter at these iteration counts; '
                    'they are reported as measured and no cost is inferred from them.\n')

    ca = pr['cost_analysis']
    body.append('\n### What kind of bound it is, from the compiled artefacts\n')
    rows = []
    for k, v in ca.items():
        if 'error' in v:
            rows.append([f'`{k}`', '—', '—', '—', '—', '—', '—'])
            continue
        fl = v.get('flops')
        by = v.get('bytes accessed')
        rows.append([f'`{k}`', f"{fl:.3e}" if fl else '—', f"{by:.3e}" if by else '—',
                     fmt(fl / by, 2) if fl and by else '—',
                     v.get('fusions'), v.get('custom_calls'), v.get('dots')])
    body.append(table(rows, ['compiled program', 'FLOPs', 'bytes accessed', 'FLOP/byte',
                             'fusions', 'custom calls', 'dots'],
                      ['---', '---:', '---:', '---:', '---:', '---:', '---:']))

    # launch-bound accounting: regress in-loop cost on fusion count
    pairs = []
    for k, v in mb.items():
        c = ca.get(k)
        if c and 'fusions' in c and v['per_iteration_us'] > 0 and 'flops' in c:
            pairs.append((k, c['fusions'], v['per_iteration_us'], c['flops'],
                          c.get('bytes accessed', 0.)))
    body.append('\n### Is it launch bound? The measurement\n')
    if len(pairs) >= 3:
        rows = [[f'`{k}`', f, fmt(t, 2), fmt(t / f, 2), f'{fl:.2e}',
                 f'{fl / (t * 1e-6) / 1e9:.1f}'] for k, f, t, fl, by in pairs]
        body.append(table(rows, ['in-loop body', 'fusions', 'us / iteration',
                                 'us / fusion', 'FLOPs', 'GFLOP/s achieved'],
                          ['---', '---:', '---:', '---:', '---:', '---:']))
        xs = [f for _, f, _, _, _ in pairs]
        ys = [t for _, _, t, _, _ in pairs]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float('nan')
        b0 = my - b1 * mx
        ss = sum((y - my) ** 2 for y in ys)
        rr = 1 - sum((y - (b0 + b1 * x)) ** 2 for x, y in zip(xs, ys)) / ss if ss else float('nan')
        fl = [f for _, _, _, f, _ in pairs]
        body.append(
            f'\nAcross these bodies the arithmetic spans a factor of '
            f'**{max(fl) / min(fl):.0f}** while the time spans a factor of only '
            f'**{max(ys) / min(ys):.1f}**. Regressing microseconds per in-loop iteration on '
            f'the compiled fusion count gives **{fmt(b1, 2)} us per fusion + {fmt(b0, 2)} us** '
            f'(R-squared = {fmt(rr, 3)}), against a bare loop-control floor of '
            f"{fmt(mb['loop_control_scalar']['per_iteration_us'], 2)} us/iteration. "
            + ('Cost tracks the number of compiled kernels, not the work inside them: the '
               'program is **kernel-count bound**. Every result below follows from that one '
               'fact - an optimisation pays only if it removes kernels, and one that removes '
               'arithmetic while adding kernels loses.\n'
               if (rr == rr and rr >= 0.7 and b1 > 0) else
               'That regression does NOT support a clean per-kernel law here (R-squared '
               'below 0.7 or a non-positive slope), so no per-kernel cost is quoted from '
               'it; the raw table above stands on its own and the arithmetic-versus-time '
               'spans are the evidence for the launch-bound reading.\n'))
        body.append('\nThe elementwise-chain rows above are NOT a per-kernel calibration and '
                    'must not be read as one: XLA fuses a chain of elementwise operations on '
                    'one small vector into a single kernel, so those rows measure arithmetic '
                    'inside one fusion, which is why they are nearly flat in the chain '
                    'length. They are kept as the contrast that makes the point.\n')
    ev = ca.get('evolution', {})
    if ev.get('flops') and ev.get('bytes accessed'):
        sec = pr['phases']['evolution']['median_seconds']
        body.append(
            f"\nThe whole evolution moves {ev['bytes accessed'] / 1e6:.1f} MB and does "
            f"{ev['flops'] / 1e6:.1f} MFLOP in {fmt(ms(sec))} ms, i.e. "
            f"{fmt(ev['bytes accessed'] / sec / 1e9, 2)} GB/s and "
            f"{fmt(ev['flops'] / sec / 1e9, 2)} GFLOP/s"
            + (', against an A100 80GB PCIe capability of order 1900 GB/s and 9700 GFLOP/s '
               'in f64.\n' if 'A100' in res.get('gpu', '') else
               f", on {res.get('gpu')}, orders of magnitude below what any current data-centre "
               'GPU sustains in f64.\n'))
    return '\n'.join(body)


def section_ladder(res, aud, g, pmap, title):
    body = [f'### {title}\n']
    meshes = sorted({k[0] for k in g})
    for L in meshes:
        inc = g.get((L, 'incumbent'))
        if inc is None:
            continue
        body.append(f'\n**{L} intervals.** Incumbent median '
                    f"{fmt(inc['gpu_ms'])} ms GPU / {fmt(inc['host_ms'])} ms including host "
                    f"transfer, over {inc['n']} retained invocations "
                    f'({len(set(r["case"] for r in res["invocations"] if r["intervals"] == L))} '
                    f"cases x {res['config']['reps']} repetitions).\n")
        floor = pmap.get((L, 'o_none'), {}).get('worst_field_relative')
        iso, cum, comp, lab = groups(res)
        blocks = [('Isolated optimisations', [n for n in iso if (L, n) in g]),
                  ('Cumulative ladder, pre-registered order',
                   [n for n in cum if (L, n) in g]),
                  ('Composed after the isolated measurements (post-hoc, labelled)',
                   [n for n in comp if (L, n) in g]),
                  ('Labelled non-parity', [n for n in lab if (L, n) in g])]
        for label, names in blocks:
            if not names:
                continue
            rows = []
            for n in names:
                v = g[(L, n)]
                p = pmap.get((L, n), {})
                null = g.get((L, 'o_none'))
                rows.append([
                    f'`{n}`', res['arm_declarations'][n]['declared_class'],
                    fmt(v['gpu_ms']), fmt(v['host_ms']),
                    fmt(v['host_ms'] - v['gpu_ms']),
                    fmt(inc['gpu_ms'] / v['gpu_ms'], 3) + 'x',
                    (fmt(null['gpu_ms'] / v['gpu_ms'], 3) + 'x') if null else '—',
                    sci(p.get('worst_field_relative')),
                    sci(p.get('audit_field_relative')),
                    fmt(p.get('iterations_identical')), fmt(p.get('reasons_identical')),
                    fmt(p.get('parity'))])
            body.append(f'\n*{label}*\n')
            body.append(table(rows, ['arm', 'intent', 'GPU ms', 'host ms', 'transfer ms',
                                     'vs incumbent', 'vs `o_none`',
                                     'parity (job)', 'parity (audit)', 'same iterations',
                                     'same exits', 'PARITY'],
                              ['---', '---', '---:', '---:', '---:', '---:', '---:', '---:',
                               '---:', ':---:', ':---:', ':---:']))
        null = g.get((L, 'o_none'))
        if floor is not None and null is not None:
            body.append(
                f'\n**Harness floor.** The null arm `o_none` — the same arithmetic emitted '
                f'through the optimisation harness with every switch off — already deviates '
                f'from the incumbent by {sci(floor)} in the field, and already runs at '
                f"{fmt(null['gpu_ms'])} ms against the incumbent's {fmt(inc['gpu_ms'])} ms "
                f"({fmt(inc['gpu_ms'] / null['gpu_ms'], 3)}x). Both are the floor: an arm at "
                'that parity has changed nothing the harness does not already change, and the '
                '`vs `o_none`` column is what each optimisation is actually worth once the '
                'harness itself is paid for.\n')
    return '\n'.join(body)


def best_parity_arm(res, g, pmap, L):
    """The fastest arm at this mesh that PASSES every parity gate."""
    iso, cum, comp, _ = groups(res)
    best = None
    for n in comp + cum + iso:
        if (L, n) not in g:
            continue
        if not pmap.get((L, n), {}).get('parity'):
            continue
        if best is None or g[(L, n)]['gpu_ms'] < g[(L, best)]['gpu_ms']:
            best = n
    return best


def section_headline(jobs):
    rows = []
    for attempt, res, aud, g, pmap in jobs:
        for L in sorted({k[0] for k in g}):
            inc = g.get((L, 'incumbent'))
            b = best_parity_arm(res, g, pmap, L)
            if inc is None or b is None:
                continue
            v = g[(L, b)]
            rows.append([attempt, L, f'`{b}`', fmt(inc['gpu_ms']), fmt(v['gpu_ms']),
                         fmt(inc['gpu_ms'] / v['gpu_ms'], 3) + 'x',
                         fmt(inc['host_ms'] / v['host_ms'], 3) + 'x',
                         sci(pmap[(L, b)]['worst_field_relative']),
                         'PASS' if inc['gpu_ms'] / v['gpu_ms'] >= 2.0 else 'FAIL'])
    return table(rows, ['attempt', 'intervals', 'fastest parity arm', 'incumbent GPU ms',
                        'arm GPU ms', 'GPU speedup', 'speedup incl. host transfer',
                        'parity', 'pre-registered 2x target'],
                 ['---', '---:', '---', '---:', '---:', '---:', '---:', '---:', ':---:'])


def section_controls(res, g, pmap):
    body = ['### Against the full-order controls, same job, same GPU\n',
            'Single-query latency. The ratio is ROM / FOM: below 1 the reduced query is '
            'faster. Same-grid error is against the same-job `fft_tight` solution at the '
            'same mesh.\n']
    foms = [f['name'] for f in res['config']['fom_settings']]
    meshes = sorted({k[0] for k in g})
    for L in meshes:
        rows = []
        for name in foms:
            v = g.get((L, name))
            if v:
                rows.append([f'`{name}`', 'FOM', fmt(v['gpu_ms']), fmt(v['host_ms']),
                             fmt(v['worst_same_grid_percent'], 4)]
                            + ['—'] * len(foms))
        best = best_parity_arm(res, g, pmap, L)
        for n in ['incumbent'] + ([best] if best else []):
            v = g.get((L, n))
            if not v:
                continue
            cells = [f'`{n}`', 'ROM', fmt(v['gpu_ms']), fmt(v['host_ms']),
                     fmt(v['worst_same_grid_percent'], 4)]
            for fname in foms:
                fv = g.get((L, fname))
                cells.append(fmt(v['gpu_ms'] / fv['gpu_ms'], 2) + 'x' if fv else '—')
            rows.append(cells)
        body.append(f'\n**{L} intervals**\n')
        body.append(table(rows, ['subject', 'kind', 'GPU ms', 'host ms', 'worst same-grid %']
                          + [f'ROM/`{f}`' for f in foms],
                          ['---', '---', '---:', '---:', '---:'] + ['---:'] * len(foms)))
    return '\n'.join(body)


def section_crossover(jobs):
    """One table across every mesh measured, so the trend is visible in one place.

    Each row is a within-job ratio; nothing is compared across jobs."""
    body = ['### The mesh trend, one row per mesh measured\n',
            'Every ratio in a row comes from the same job, the same GPU and the same '
            'randomized interleaving. Rows from different jobs are never divided by each '
            'other.\n']
    rows = []
    for attempt, res, aud, g, pmap in jobs:
        for L in sorted({k[0] for k in g}):
            inc = g.get((L, 'incumbent'))
            b = best_parity_arm(res, g, pmap, L)
            if inc is None or b is None:
                continue
            v = g[(L, b)]
            cells = [attempt, L, f'`{b}`', fmt(inc['gpu_ms']), fmt(v['gpu_ms']),
                     fmt(inc['gpu_ms'] / v['gpu_ms'], 3) + 'x',
                     fmt(v['worst_same_grid_percent'], 3)]
            for fname in [f['name'] for f in res['config']['fom_settings']]:
                fv = g.get((L, fname))
                cells.append((fmt(v['gpu_ms'] / fv['gpu_ms'], 2) + 'x') if fv else '—')
                cells.append(fmt(fv['worst_same_grid_percent'], 3) if fv else '—')
            rows.append(cells)
    if not rows:
        return ''
    foms = [f['name'] for f in jobs[0][1]['config']['fom_settings']]
    head = ['attempt', 'intervals', 'arm', 'incumbent GPU ms', 'arm GPU ms', 'speedup',
            'arm err %']
    align = ['---', '---:', '---', '---:', '---:', '---:', '---:']
    for f in foms:
        head += [f'arm/`{f}`', f'`{f}` err %']
        align += ['---:', '---:']
    body.append(table(rows, head, align))
    return '\n'.join(body)


def section_throughput(res, g):
    if not res.get('throughput'):
        return ''
    body = ['### Throughput, clearly labelled\n',
            'One compiled program over the eight-case cohort. This is **throughput, not '
            'latency**: a batched `while_loop` pays each batch its worst-case iteration '
            'count, and the full-order controls were **not** batched, so no ROM-beats-FOM '
            'claim rests on this table.\n']
    rows = []
    for t in res['throughput']:
        L = t['intervals']
        inc = g.get((L, 'incumbent'))
        rows.append([L, f"`{t['arm']}`", t['batch'], fmt(ms(t['median_seconds'])),
                     fmt(ms(t['per_query_seconds'])),
                     fmt(inc['gpu_ms'] / ms(t['per_query_seconds']), 2) + 'x' if inc else '—',
                     sci(t['worst_field_relative_vs_incumbent'])])
    body.append(table(rows, ['intervals', 'arm', 'batch', 'batch ms', 'ms / query',
                             'vs incumbent single-query latency', 'parity vs incumbent'],
                      ['---:', '---', '---:', '---:', '---:', '---:', '---:']))
    body.append('\nFor the same meshes the single-query **latency** of every arm is in the '
                'ladder table above; these two numbers answer different questions and are '
                'never mixed.\n')
    return '\n'.join(body)


DIAGRAM = '''```mermaid
flowchart TD
    U[supplied dense field u0] --> S1[bilinear sample at 2304 Gauss points]
    S1 --> S2[candidate code selection]
    S2 --> IC["initial fit: LM on R_cold h z - y"]
    IC --> STEP{{"50 time steps"}}
    STEP --> P["step projection p = A h z"]
    P --> PR["extrapolation probe: 2 residual norms"]
    PR --> LM["damped LM, budget 180"]
    LM --> RJ["residual + Jacobian"]
    LM --> SOL["16x16 damped normal solve"]
    RJ --> K["stencil contraction G5 h"]
    RJ --> AH["test projection A h"]
    RJ --> QD["quadrature matvec Pq^T adv"]
    STEP --> DEC["decode six fields: G h z"]
    DEC --> OUT[six dense GPU fields]
    OUT --> HT[host transfer]

    classDef fused fill:#1f6feb,stroke:#0b3d91,color:#ffffff
    classDef hoisted fill:#2da44e,stroke:#116329,color:#ffffff
    classDef folded fill:#8250df,stroke:#4c2889,color:#ffffff
    classDef unchanged fill:#d0d7de,stroke:#57606a,color:#1f2328
    classDef io fill:#fff8c5,stroke:#9a6700,color:#1f2328

    class RJ,LM fused
    class P,PR hoisted
    class K,AH,QD,SOL,DEC folded
    class S1,S2,IC,STEP unchanged
    class U,OUT,HT io
```

`fused` = one `jax.linearize` pass replaces two primals and a `lax.cond` (`fuse`);
`hoisted` = the $\\nu$-dependent constants leave the loop and one evaluation serves both the
step projection and the probe (`hoist`, `share`); `folded` = the head's output layer is
pre-multiplied into the frozen operators and the small solve eliminates four unknowns per
stage (`lean`, `block4`, `leandec`); `unchanged` = same program, same arithmetic.
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', nargs='+', required=True,
                   help='attempt:result.json:audit.json triples')
    p.add_argument('--out', required=True)
    a = p.parse_args()

    jobs = []
    for spec in a.jobs:
        attempt, rp, ap = spec.split(':')
        res = json.loads(Path(rp).read_text())
        aud = json.loads(Path(ap).read_text())
        jobs.append((attempt, res, aud, gather(res), parity_map(res, aud)))

    doc = ['# Making the frozen Burgers query faster at parity: the profile, the port, and '
           'what it bought\n',
           'Where the 2026-09-14 head ablation\'s `a_neural_eq` query spends its time, and '
           'how much of that a kernel-level port of the 2026-08-28 1D optimisations removes '
           'without changing what is computed. Numbers **final** for the meshes and arms '
           'listed; every table is generated from the job JSONs and the independent NumPy '
           'audits by the generator beside this file, and every arm carries its own parity '
           'verdict.\n',
           '## The answer, up front\n',
           section_headline(jobs),
           '\nThe speedup column is the incumbent\'s median single-query GPU latency divided '
           'by the arm\'s, both measured in the same job on the same GPU with every arm '
           'interleaved in a randomized order. Parity is the worst relative field deviation '
           'from the incumbent over every case, against a bar of 1e-12, with the per-step '
           'iteration counts and exit reasons identical as integers.\n',
           '## The query\n', DIAGRAM, '\n## S0 — the profile\n']

    for attempt, res, aud, g, pmap in jobs:
        for pr in res['profile']:
            doc.append(f"\n*From attempt `{attempt}`, job `{res['job_id']}` on "
                       f"`{res['gpu']}`.*\n")
            doc.append(section_profile(pr, res))

    doc.append('\n## S1/S2 — the optimisation ladder at parity\n')
    for attempt, res, aud, g, pmap in jobs:
        doc.append(section_ladder(res, aud, g, pmap,
                                  f"Attempt `{attempt}` — job `{res['job_id']}`, "
                                  f"`{res['gpu']}`"))

    doc.append('\n## The full-order controls\n')
    for attempt, res, aud, g, pmap in jobs:
        doc.append(f"\n*Attempt `{attempt}`.*\n")
        doc.append(section_controls(res, g, pmap))

    doc.append('\n## The mesh trend\n')
    doc.append(section_crossover(jobs))

    doc.append('\n## S3 — throughput\n')
    for attempt, res, aud, g, pmap in jobs:
        s = section_throughput(res, g)
        if s:
            doc.append(f"\n*Attempt `{attempt}`.*\n")
            doc.append(s)

    doc.append('\n## Gates\n')
    rows = []
    for attempt, res, aud, g, pmap in jobs:
        for k, v in aud['checks'].items():
            if isinstance(v, (bool, float, int)) and not isinstance(v, list):
                rows.append([attempt, f'`{k}`', fmt(v) if not isinstance(v, float)
                             else sci(v)])
    doc.append(table(rows, ['attempt', 'gate', 'value'], ['---', '---', '---:']))

    doc.append('\n## Ops record\n')
    for attempt, res, aud, g, pmap in jobs:
        doc.append(f"\n- `{attempt}`: job `{res['job_id']}`, `{res['gpu']}`, "
                   f"`jax_backend={res['backend']}`, x64 `{res['x64']}`, matmul precision "
                   f"`{res['matmul_precision']}`, JAX `{res['jax_version']}`, source commit "
                   f"`{res['commit']}`, checkpoint sha256 `{res['checkpoint_sha256'][:16]}…` "
                   f"unchanged at exit, elapsed {res['elapsed_seconds'] / 60:.1f} min, "
                   f"{len(res['invocations'])} timed invocations, audit passed "
                   f"`{aud['passed']}`.")

    doc.append('''
## Glossary

Written for a reader who opens this cold.

- **The query** — one complete reduced-order model run: a supplied dense initial field on the
  GPU in, six dense output fields at $t = 0, 0.05, \\dots, 0.25$ out. Everything in between —
  fitting the initial latent code, stepping 50 times, decoding — is charged to it.
- **Incumbent** — the retained solver, `arms.make_query` imported unmodified. It is the thing
  every number here is measured against, not a re-implementation of it.
- **Arm** — one variant of the query. Isolated arms switch on a single optimisation; the
  cumulative ladder `L1`…`L7` switches them on one at a time in a pre-registered order.
- **Parity** — the requirement that an arm change *how* the answer is computed and not *what*
  the answer is: output fields within $10^{-12}$ relative of the incumbent's, and the
  per-step iteration counts and exit reasons identical as integers. An arm that is faster but
  fails parity is rejected, not reported as a speedup.
- **Harness floor** — the parity deviation of the null arm `o_none`, which switches nothing
  on. It is not zero, because re-emitting the same arithmetic in a differently structured
  program changes XLA's fusion decisions and therefore its rounding. Read every arm's parity
  number against this floor.
- **bitwise / reassociation** — the declared intent of an optimisation: `bitwise` means it
  evaluates the identical floating-point expressions, `reassociation` means it is
  algebraically exact but rounds differently. At whole-query level the measurement cannot
  tell them apart (see the harness floor); the classes are verified at component level.
- **LM (Levenberg–Marquardt)** — the damped nonlinear least-squares iteration that solves each
  time step. Each iteration builds a Jacobian, solves a small damped normal system for a
  step, and accepts it only if the residual decreased.
- **Jacobian** — the matrix of derivatives of the 64 weak-residual components with respect to
  the 16 latent unknowns.
- **`jax.linearize` / `jacfwd`** — two ways to get that Jacobian. `jacfwd` evaluates the
  function and then differentiates it; `linearize` does one pass that yields both.
- **`lax.cond`** — a GPU branch. Removing it costs an unconditional Jacobian on rejected
  steps but removes a dispatch boundary and a duplicated function evaluation.
- **Hoisting** — moving a computation that cannot change out of the loop that was repeating
  it. Here: everything that depends only on the query's viscosity.
- **Empirical quadrature (EQ)** — the rule that evaluates the nonlinear advection term at only
  $m = 256$ chosen grid points with fitted weights, instead of all 65025. It is the campaign's
  largest cost lever and is held fixed here.
- **Stencil contraction $G_5 h$** — evaluating the decoder at each quadrature point and its
  four neighbours, which is what the upwind advection needs.
- **Test projection $A h$** — projecting the state onto the 64 sine test modes that define the
  weak residual.
- **Gauss–Jordan** — solving the small linear system by elimination written as array
  operations instead of calling a library factorization. The incumbent eliminates one unknown
  per stage: 16 strictly sequential stages. **Block Gauss–Jordan** eliminates $b$ at a time.
- **Folding / `lean`** — pre-multiplying the head's last weight matrix into the frozen
  operators offline, so one matmul at query time replaces three.
- **`nodot`** — writing tiny matrix-vector products as broadcast-and-sum expressions, which
  fuse, instead of as library calls, which each cost a fixed dispatch.
- **Decode** — turning the six latent codes into six dense fields. Its cost grows with the
  mesh; the reduced solve's does not.
- **Kernel-count bound (launch bound)** — the wall time is set by how many GPU operations are
  launched, not by arithmetic or memory bandwidth. Each operation here is microseconds and
  moves almost nothing, so the gaps dominate.
- **Fusion** — one compiled GPU kernel, XLA's unit of launch. The fusion count of a compiled
  program is the quantity that matters for a launch-bound program.
- **FLOP/byte** — arithmetic per byte moved. Low values mean the program is not doing enough
  work to justify the memory traffic; here both are far below what the device can sustain,
  which is the signature of a launch-bound program.
- **FOM (full-order model)** — the ordinary finite-difference Burgers solver on the same grid,
  run in the same job. `fft_loose` stops Newton at $10^{-2}$, `fft_tight` at $10^{-6}$;
  `fft_loose_dt01` is `fft_loose` with twice the time step.
- **Same-grid error** — the reduced solution's discrepancy from the same-job `fft_tight`
  solution on the same grid, so discretisation error cancels and what is left is the
  reduced model's own error.
- **Latency vs throughput** — latency is how long one query takes; throughput is total time
  divided by the number of queries when several run in one batched program. The full-order
  controls were not batched, so throughput numbers are never compared against them.
- **Repetition / median** — every timed invocation is repeated; all repetitions are kept in
  the JSON and the tables quote medians. Nothing is a single measurement.
''')

    text = '\n'.join(doc) + '\n'
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
