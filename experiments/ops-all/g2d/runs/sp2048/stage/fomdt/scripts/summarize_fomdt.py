"""g2d fomdt: results.json from the pulled job records (runs/fd<L>/pull/out/result.json), with the anchor check against
burgers-bank-knob checks/bk<L>-summary.json. No number is typed.

    /home/tahmid/Dev/.venv/bin/python fomdt/scripts/summarize_fomdt.py
"""
import json
from pathlib import Path

F = Path(__file__).resolve().parents[1]
RUNS = F.parent / 'runs'
BK = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers-bank-knob/experiments/burgers-bank-knob/checks')
ANCH = ('lean_nt3e-3_l3e-3_dt005', 'lean_nt1e-2_l1e-2_dt01')


def main():
    res = dict(study='g2d fomdt: Burgers 2D lean Newton-BiCGStab at intermediate dt (dev6)', design='fomdt/DESIGN.md',
               metric='per case max over evolved times of ||u-u_ref||_2/||u_0||_2 vs same-grid fft_tight; worst/mean/median over the 6 cases',
               meshes={})
    for L in (256, 512, 1024, 2048, 4096):
        job = f'fd{L}'
        r = RUNS / job / 'pull/out/result.json'
        sub = RUNS / job / 'SUBMIT.log'
        rec = dict(job=job, slurm_id=sub.read_text().split()[-1] if sub.exists() else None)
        if not r.exists():
            rec['status'] = 'not collected'
            res['meshes'][str(L)] = rec
            continue
        d = json.loads(r.read_text())
        out = (RUNS / job / 'pull/job.out').read_text(errors='replace')
        rec.update(gpu=d['gpu'], backend_gpu='jax_backend=gpu' in out, complete=d.get('complete'), gates=d.get('gates'),
                   settings={n: {k: s[k] for k in ('dt', 'ntol', 'ltol', 'steps_per_output', 'worst_percent', 'mean_percent',
                                                    'median_percent', 'median_gpu_ms', 'converged', 'timed_invocations')}
                             for n, s in d['settings'].items()})
        bk = BK / f'bk{L}-summary.json'
        if bk.exists():
            t = json.loads(bk.read_text())['table']
            rec['anchor_check'] = {a: dict(this_job=rec['settings'][a]['worst_percent'], bank_knob=t[a]['worst_evolved_percent'],
                                           bank_knob_ms=t[a]['median_gpu_ms'],
                                           equal_to_4_significant=f"{rec['settings'][a]['worst_percent']:.4g}" == f"{t[a]['worst_evolved_percent']:.4g}")
                                   for a in ANCH if a in t}
        res['meshes'][str(L)] = rec
        print(f'== {L}^2 job {rec["slurm_id"]} {rec["gpu"]} complete {rec["complete"]} gates {rec["gates"]}')
        for n, s in sorted(rec['settings'].items(), key=lambda kv: kv[1]['median_gpu_ms']):
            print(f"  {n:28s} worst {s['worst_percent']:.4f}  mean {s['mean_percent']:.4f}  median {s['median_percent']:.4f}  {s['median_gpu_ms']:.2f} ms")
        print('  anchors', rec.get('anchor_check'))
    (F / 'results.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
