"""Final evaluation of trained operators on a cohort (PROTOCOL.md section 5): errors at 256^2, batch-1 timing,
and the two 1024^2 routes (zero-shot; coarse-then-bilinear). Refuses to overwrite an existing output file.

  evaluate.py --runs DIR [DIR ...] --cohort-dir DIR --cohort test64 --out FILE [--skip-1024]
Each DIR is a directory of run directories (each with best.pt + meta.json).
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import opnet


def stats(x):
    x = np.asarray(x, dtype=np.float64) * 1e3
    return dict(count=int(x.size), median_ms=float(np.median(x)), mean_ms=float(x.mean()), p05_ms=float(np.quantile(x, .05)),
                p95_ms=float(np.quantile(x, .95)), min_ms=float(x.min()), max_ms=float(x.max()))


def load(run):
    ck = torch.load(run / 'best.pt', map_location='cuda', weights_only=False)
    net = opnet.make(ck['config'])
    net.load_state_dict({k: v for k, v in ck['model'].items() if k != '_metadata'})
    net.eval()
    return net, ck


def errors(pred6, ref, n0):
    """pred6, ref (6,H,W) float64 on GPU -> list of 6 relative errors."""
    return ((pred6 - ref).square().sum(dim=(-2, -1)).sqrt() / n0).cpu().tolist()


def query_direct(net, u0, nu, norm):
    return torch.cat((u0[:, None], opnet.predict(net, u0, nu, norm)), dim=1)


def query_coarse(net, u0, nu, norm, stride):
    c = opnet.predict(net, u0[:, ::stride, ::stride], nu, norm)
    up = F.interpolate(c, size=u0.shape[-2:], mode='bilinear', align_corners=True)
    return torch.cat((u0[:, None], up), dim=1)


def run_cohort(net, norm, fields, nus, n0s, query, reps=5, burn=20):
    per, t = [], []
    with torch.no_grad():
        for c in range(len(fields)):
            ref = torch.from_numpy(np.ascontiguousarray(fields[c])).cuda()
            u0, nu = ref[:1].clone(), torch.tensor([nus[c]], dtype=torch.float64, device='cuda')
            if c == 0:
                for _ in range(burn):
                    query(net, u0, nu, norm)
            torch.cuda.synchronize()
            for r in range(reps):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = query(net, u0, nu, norm)
                torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
            n0 = float(ref[0].square().sum().sqrt())
            assert abs(n0 - n0s[c]) <= 1e-10 * n0
            e = errors(out[0], ref, n0)
            assert e[0] == 0.0 and np.isfinite(e).all()
            per.append(e[1:])
    ev = np.max(np.asarray(per), axis=1)
    return dict(worst_percent=100 * float(ev.max()), median_percent=100 * float(np.median(ev)),
                mean_percent=100 * float(ev.mean()), per_case_per_time=per, timing=stats(t))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--runs', nargs='+', required=True); a.add_argument('--cohort-dir', required=True)
    a.add_argument('--cohort', default='test64'); a.add_argument('--out', required=True)
    a.add_argument('--skip-1024', action='store_true')
    a = a.parse_args()
    out = Path(a.out)
    assert not out.exists(), 'the cohort is evaluated once; refusing to overwrite'
    opnet.configure()
    data = {}
    for L in (256,) if a.skip_1024 else (256, 1024):
        m = np.load(Path(a.cohort_dir) / f'{a.cohort}-L{L}.meta.npz')
        data[L] = (np.load(Path(a.cohort_dir) / f'{a.cohort}-L{L}.fields.npy', mmap_mode='r'), m['params'][:, 4], m['n0'])
    res = dict(cohort=a.cohort, gpu=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'),
               node=os.environ.get('SLURMD_NODENAME'), runs={})
    for d in a.runs:
        for run in sorted(Path(d).iterdir()):
            if not (run / 'best.pt').exists():
                continue
            net, ck = load(run)
            norm = ck['normalization']
            r = dict(arm=ck['arm'], rung=ck['rung'], seed=ck['seed'], best_step=ck['step'], validation=ck['validation'],
                     real_parameters=opnet.real_parameters(net),
                     train_result=json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else None)
            r['L256'] = run_cohort(net, norm, *data[256], query_direct)
            if not a.skip_1024:
                for name, q in (('L1024_zero_shot', query_direct),
                                ('L1024_coarse_bilinear', lambda n, u, v, nm: query_coarse(n, u, v, nm, 4))):
                    try:
                        r[name] = run_cohort(net, norm, *data[1024], q)
                    except torch.cuda.OutOfMemoryError as exc:
                        r[name] = dict(error=str(exc)[:300]); torch.cuda.empty_cache()
            res['runs'][f'{Path(d).name}/{run.name}'] = r
            print('EVAL', run.name, {k: (round(v['worst_percent'], 3), round(v['median_percent'], 3), round(v['timing']['median_ms'], 2))
                                     for k, v in r.items() if isinstance(v, dict) and 'worst_percent' in v}, flush=True)
            del net; torch.cuda.empty_cache()
    out.write_text(json.dumps(res, indent=1) + '\n')
    print('EVALUATION DONE', out, flush=True)


if __name__ == '__main__':
    main()
