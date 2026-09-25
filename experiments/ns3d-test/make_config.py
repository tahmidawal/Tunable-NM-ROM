"""Write configs/test{n}.json (the frozen test-run settings) from the development records.

Usage: make_config.py <mesh>
Every NM-ROM arm and CNAB2 setting is taken from the development job `a2_h{n}` of the
ns3d-shift-head lane (the source of the paper's development Table-1 rows), with its 16
per-case development error rows embedded as the reproduction reference. No number here is
chosen on test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = {32: 'experiments/ns3d-shift-head/runs/a2_h32/output/summary.json',
       64: 'experiments/ns3d-shift-head/runs/a2_h64/output/summary.json'}
SPANS = (64, 48, 32, 16, 8)
SETTINGS = ((0.02, 3), (0.04, 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh', type=int, choices=sorted(REF))
    n = ap.parse_args().mesh
    ref_path = ROOT / REF[n]
    s = json.loads(ref_path.read_text())
    arms = [dict(name='nmrom_accurate_head_k8', kind='head', k=8, dt=0.02, iters=3, role='accurate',
                 reference=('frontier', 'k8_q0_dt0.02_it3'))]
    for dt, it in SETTINGS:
        for Rp in SPANS:
            if Rp == 16 and (dt, it) == (0.04, 2):
                name, role = 'nmrom_fast_span16_dt0.04_it2', 'fast (development Table 1)'
            elif Rp == 16:
                name, role = 'nmrom_fast_span16_dt0.02_it3', 'fast (96^3 test setting)'
            else:
                name, role = f'span{Rp}_dt{dt}_it{it}', 'span ladder'
            arms.append(dict(name=name, kind='span', rank=Rp, dt=dt, iters=it, role=role,
                             reference=('span', f'span{Rp}_dt{dt}_it{it}')))
    ref_errors, ref_values = {}, {}
    for a in arms:
        sec, key = a.pop('reference')
        ref_errors[a['name']] = key
        ref_values[key] = s[sec][key]['errors']
    steps = [200, 100, 80, 70, 60, 50, 40, 20, 10]
    for st in steps:
        ref_errors[f'cnab2_s{st}'] = f'cnab2_s{st}'
        ref_values[f'cnab2_s{st}'] = s['cnab2'][str(st)]['errors']
    frozen_dir = f'experiments/ns3d-operators/frozen/h{n}'
    fsha = {name: sha(ROOT / frozen_dir / name) for name in ('head_k8.npz', 'rotation.npz', 'bank_probe.npz')}
    for name in fsha:  # the frozen copies must be the development job's own outputs
        src = ROOT / f'experiments/ns3d-shift-head/runs/a2_h{n}/output/{name}'
        if sha(src) != fsha[name]:
            raise RuntimeError(f'{frozen_dir}/{name} differs from a2_h{n}')
    cfg = dict(
        name=f'test{n}', n=n, horizon=0.2, dt_truth=0.001, rank=64, modes=292, check_modes=8, gram_block=64,
        train_seed=202609201, train_cases=128, head_train_cases=512,
        dev_seed=202609202, dev_cases=16,
        test_seed=202609221, test_cases=32,
        closed_seeds=[[202609203, 32], [202609211, 32]],
        damping=1e-6, ic_iters=12, arms=arms, cnab2_steps=steps,
        gated_arms=['nmrom_accurate_head_k8', 'nmrom_fast_span16_dt0.04_it2', 'nmrom_fast_span16_dt0.02_it3'],
        frozen_dir=frozen_dir, frozen_sha256=fsha,
        development_job=s['job_id'], reference_summary=REF[n], reference_summary_sha256=sha(ref_path),
        reference_errors=ref_errors, reference_values=ref_values,
        full_cases=[0, 1] if n == 32 else [],
        sample_seed=20260925 + n, sample_points=8192, min_free_gb=40, field_cap_gb=4.0,
        timing_rounds=3, burn_calls=2, timing_seed=925 + n,
        smoke=dict(dev_cases=2, test_seed=7, test_cases=2, cnab2_steps=[200, 50, 40], timing_rounds=1,
                   burn_calls=1, min_free_gb=5))
    out = HERE / 'configs' / f'test{n}.json'
    out.write_text(json.dumps(cfg, indent=1) + '\n')
    print(out, sha(out))


if __name__ == '__main__':
    main()
