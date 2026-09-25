"""Write configs/t2test{n}.json: the ns3d-operators panel config of mesh n (its pre-registered,
validation-only operator selection and NM-ROM settings, unchanged), re-pointed at the 32
held-out TEST cases (seed 202609221), with the reproduction gate taken from the ns3d-test
lane's test job at the same mesh (per-case test errors of head k=8, span R'=16 dt 0.02/3
sweeps and every CNAB2 setting).

Usage: make_config.py <mesh> --ops-runs DIR --ns3d-test-summary FILE
  --ops-runs           the ns3d-operators worktree's runs/ (holds tr{n}_<arm>/output/best.pt)
  --ns3d-test-summary  worktrees/2026-09-25-ns3d-test/.../runs/t{n}a/output/summary.json
Nothing here reads a test-cohort panel output of this lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NS = '/cluster/tufts/paralab/tawal01/t2ntest_20260925'
TEST_SEED, TEST_CASES = 202609221, 32
# panel arm name -> ns3d-test result name (same settings, same cohort, earlier job)
REF_MAP = {'nmrom_accurate_head_k8': 'nmrom_accurate_head_k8',
           'nmrom_fast_span16': 'nmrom_fast_span16_dt0.02_it3'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh', type=int, choices=(32, 64))
    ap.add_argument('--ops-runs', type=Path, required=True)
    ap.add_argument('--ns3d-test-summary', type=Path, required=True)
    args = ap.parse_args()
    n = args.mesh
    base_path = ROOT / 'experiments' / 'ns3d-operators' / 'configs' / f'panel{n}.json'
    cfg = json.loads(base_path.read_text())
    # --- NM-ROM settings must be the Table 2 ones (span R'=16, dt 0.02, 3 sweeps; head k=8)
    assert (cfg['span_rank'], cfg['rom_dt'], cfg['rom_iters']) == (16, 0.02, 3), cfg
    # --- operators: same entries (selection unchanged); checkpoint re-pointed and hash-checked locally
    for op in cfg['operators']:
        local = args.ops_runs / f"tr{n}_{op['arm']}" / 'output' / 'best.pt'
        digest = sha(local)
        if digest != op['sha256']:
            raise SystemExit(f"{op['arm']}: local checkpoint {digest} != training record {op['sha256']}")
        op['checkpoint'] = f"checkpoints/tr{n}_{op['arm']}.pt"  # relative to the job dir (panel cwd)
        op['source_checkpoint'] = str(local)
    # --- reproduction reference = ns3d-test test job on the same cohort
    ts = json.loads(args.ns3d_test_summary.read_text())
    tcfg = ts['config']
    assert int(tcfg['n']) == n and int(tcfg['test_seed']) == TEST_SEED and int(tcfg['test_cases']) == TEST_CASES
    assert ts['status'] == 'final', ts['status']
    assert tcfg['frozen_sha256'] == cfg['frozen_sha256'], 'ns3d-test used different frozen NM-ROM files'
    ref_map, ref_values = {}, {}
    for arm, tname in REF_MAP.items():
        ref_map[arm] = tname
        ref_values[tname] = ts['results'][tname]['errors']
    for st in cfg['cnab2_steps']:
        nm = f'cnab2_s{st}'
        r = ts['results'].get(nm)
        if r is not None and r['finite']:
            ref_map[nm] = nm
            ref_values[nm] = r['errors']
    cfg.update(
        eval_seed=TEST_SEED, eval_cases=TEST_CASES, full_cases=[0, 1],
        reference_summary=str(args.ns3d_test_summary.resolve().relative_to(ROOT.parent)),
        reference_summary_job=ts['job_id'], reference_summary_sha256=sha(args.ns3d_test_summary),
        reference_errors=ref_map, reference_values=ref_values,
        base_config=f'experiments/ns3d-operators/configs/panel{n}.json', base_config_sha256=sha(base_path),
        lane='t2-ns3d-test')
    # smoke (--smoke): seed 7, never the test cohort; no reproduction reference exists for it
    cfg['smoke'] = dict(eval_seed=7, eval_cases=2, cnab2_steps=[200, 50, 40], timing_rounds=1, burn_calls=1,
                        reference_errors={})
    out = HERE / 'configs' / f't2test{n}.json'
    out.write_text(json.dumps(cfg, indent=1) + '\n')
    print(out, sha(out), {o['arm']: o['selected'] for o in cfg['operators']})


if __name__ == '__main__':
    main()
