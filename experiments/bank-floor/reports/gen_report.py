"""Generate the bank-floor tables and summary.json from the run JSONs. No hand-typed numbers.

usage: gen_report.py   (reads ../runs/<attempt>/result.json as listed in SOURCES below)
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs'
SOURCES = dict(rep_poisson='bfp02', rep_burgers='bfb03', solve_poisson='bfsp01', solve_burgers='bfsb01')


def load(att):
    p = RUNS / att / 'result.json'
    if not p.exists():
        return None, None
    return json.loads(p.read_text()), hashlib.sha256(p.read_bytes()).hexdigest()


def pct(x):
    return f'{100 * x:.4f}'


def order(tag):
    m = re.match(r'([a-z_]+?)(\d+)$', tag)
    fam = ['inc', 'ft', 'cat', 'pod', 'podraw', 'pod_sub', 'podraw_sub', 'random']
    return (fam.index(m.group(1)) if m and m.group(1) in fam else 99, int(m.group(2)) if m else 0)


def rep_table(res, primary, large):
    inc = res['arms']['inc512']['floors']
    rows = ['| arm | R | rank | cond | train mean-sq floor (rms %) | ' + primary + ' worst % | ' + primary
            + ' median % | ' + ' | '.join(f'{c} worst %' for c in large) + ' | vs inc512 (' + primary
            + ') | head-transplant defect worst % | decode FLOPs/field | full-bank operator entries | bank MB |',
            '|---|---:|---:|---:|---:|---:|---:|' + '---:|' * len(large) + '---:|---:|---:|---:|---:|']
    out = {}
    for tag in sorted(res['arms'], key=order):
        a = res['arms'][tag]
        f, c, b = a['floors'], a['cost'], a['basis']
        ratio = inc[primary]['worst'] / f[primary]['worst']
        cond = b.get('condition_number')
        rows.append(f"| `{tag}` | {c['R']} | {b['rank']} | {cond:.2e} | " if cond else
                    f"| `{tag}` | {c['R']} | {b['rank']} | orthonormal | ")
        rows[-1] += (f"{pct(f['train_full']['rms'])} | {pct(f[primary]['worst'])} | {pct(f[primary]['median'])} | "
                     + ' | '.join(pct(f[k]['worst']) for k in large)
                     + f" | {ratio:.2f}x | {pct(f['head_transplant_defect']['worst'])} | "
                     f"{c['decode_flops_per_field']:.3g} | {c['full_bank_operator_entries']:.3g} | "
                     f"{c['bank_bytes'] / 1e6:.0f} |")
        out[tag] = dict(R=c['R'], rank=b['rank'], primary_worst=f[primary]['worst'],
                        large_worst={k: f[k]['worst'] for k in large}, improvement_vs_inc512=ratio,
                        label=a.get('label'), fine_mesh=a.get('fine_mesh', {}).get('dev'),
                        p1_promoted=bool(f[primary]['worst'] <= inc[primary]['worst'] / 3
                                         and all(f[k]['worst'] <= inc[k]['worst'] / 2 for k in large)
                                         and b.get('rank_valid', True)))
    return '\n'.join(rows), out


def main():
    summary = dict(sources={})
    md = []
    for key, primary, large, title in (('rep_poisson', 'dev12', ['common256', 'fresh256'], 'Poisson 2D, 255 intervals'),
                                       ('rep_burgers', 'dev6', ['hold64'], 'Burgers 2D, 256 intervals')):
        res, h = load(SOURCES[key])
        if res is None:
            continue
        summary['sources'][key] = dict(attempt=SOURCES[key], result_sha256=h, job=res['slurm_job'],
                                       commit=res['source_commit'], gpu=res['gpu'], complete=res['complete'])
        table, out = rep_table(res, primary, large)
        summary[key] = out
        md += [f'### Floors — {title} (job {res["slurm_job"]}, complete={res["complete"]})', '', table, '']
    for key, title in (('solve_poisson', 'Poisson 2D'), ('solve_burgers', 'Burgers 2D')):
        res, h = load(SOURCES[key])
        if res is None:
            continue
        summary['sources'][key] = dict(attempt=SOURCES[key], result_sha256=h, job=res['slurm_job'],
                                       commit=res['source_commit'], gpu=res['gpu'], complete=res['complete'])
        rows, out = [], {}
        cohorts = list(next(iter(res['subjects'].values()))['results'])
        if key == 'solve_poisson':
            rows = ['| subject | kind | R | M | ' + ' | '.join(f'{c} worst %' for c in cohorts)
                    + ' | median total ms | median device ms |', '|---|---|---:|---:|' + '---:|' * len(cohorts) + '---:|---:|']
            for name, s in res['subjects'].items():
                r = s['results']
                rows.append(f"| `{name}` | {s['kind']} {s.get('form', '')} | {s.get('R', '-')} | {s.get('M', '-')} | "
                            + ' | '.join(pct(r[c]['worst']) for c in cohorts)
                            + f" | {r[cohorts[0]]['median_total_ms']:.3f} | {r[cohorts[0]]['median_device_ms']:.3f} |")
                out[name] = dict(kind=s['kind'], R=s.get('R'), worst={c: r[c]['worst'] for c in cohorts},
                                 median_total_ms=r[cohorts[0]]['median_total_ms'])
        else:
            rows = ['| subject | kind | R | M | ' + ' | '.join(f'{c} worst all-times % | {c} worst evolved %' for c in cohorts)
                    + ' | floor on dev6 % | budget exits (dev6) | median total ms (dev6) | reps |',
                    '|---|---|---:|---:|' + '---:|---:|' * len(cohorts) + '---:|---:|---:|---:|']
            for name, s in res['subjects'].items():
                r = s['results']
                be = sum(x['budget_exits'] for x in r['dev6']['solver']) if r['dev6']['solver'] else '-'
                fl = s.get('bank_floor_on_reference_fields', {}).get('dev6')
                rows.append(f"| `{name}` | {s['kind']} | {s.get('R', '-')} | {s.get('M', '-')} | "
                            + ' | '.join(f"{pct(r[c]['worst_all_times'])} | {pct(r[c]['worst_evolved'])}" for c in cohorts)
                            + f" | {pct(fl) if fl else '-'} | {be} | {r['dev6']['median_total_ms']:.1f} | {s['repetitions']} |")
                out[name] = dict(kind=s['kind'], R=s.get('R'),
                                 worst_all={c: r[c]['worst_all_times'] for c in cohorts},
                                 worst_evolved={c: r[c]['worst_evolved'] for c in cohorts},
                                 median_total_ms=r['dev6']['median_total_ms'])
        summary[key] = out
        md += [f'### Solved error and paired cost — {title} (job {res["slurm_job"]}, {res["gpu"]})', '',
               '\n'.join(rows), '']
    man = HERE.parent / 'CKPT-MANIFEST.json'
    if man.exists():
        m = json.loads(man.read_text())
        md += ['### Checkpoints (git-ignored `ckpt/`, hashes from `CKPT-MANIFEST.json`)', '',
               '| file | MB | SHA256 |', '|---|---:|---|']
        md += [f"| `{v['path'].split('bank-floor/')[-1]}` | {v['bytes'] / 1e6:.1f} | `{v['sha256']}` |" for k, v in m.items()]
        md += ['']
    (HERE / 'tables.generated.md').write_text('\n'.join(md) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
