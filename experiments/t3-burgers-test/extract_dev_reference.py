"""Freeze the development-cohort (dev6) Table-3 values and the parent jobs' basis records into inputs/dev-reference.json.

Reads the parent lane's committed records (burgers-compare-hires @ 07c0e3e8): checks/p1024-summary.json (job 4204019),
checks/p2048e-summary.json (job 4218390) and artifacts/<attempt>/result.json.gz (POD eigenvalues / mode hash, snapshot
hash, quadratic-manifold r=16 fit record). Run once, before any test job; the output is committed with DESIGN.md.

    python extract_dev_reference.py <path to the parent lane directory>
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ARMS = dict(nmrom_accurate='bank384_M1536_lat64_g1em06', nmrom_head='q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2',
            pod_lspg16='pod16_M64_dense', qman16='qman16_quad_M64')
CELL_FOM = 'lean_nt3e-3_l3e-3_dt005'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    lane = Path(sys.argv[1])
    out = {}
    for L, att in ((1024, 'p1024'), (2048, 'p2048e')):
        sp = lane / 'checks' / f'{att}-summary.json'
        rp = lane / 'artifacts' / att / 'result.json.gz'
        s = json.loads(sp.read_text())
        r = json.loads(gzip.open(rp).read())
        arms = {a['name']: a for a in s['arms']}
        fom = arms[CELL_FOM]
        acc = arms[ARMS['nmrom_accurate']]
        cand = [a for a in s['arms'] if a['family'] == 'fom' and a.get('gpu_ms') and
                a['worst_evolved_percent'] <= acc['worst_evolved_percent']]
        rule_fom = min(cand, key=lambda a: a['gpu_ms'])
        assert rule_fom['name'] == CELL_FOM, rule_fom['name']
        q16 = next(x for x in r['quadratic_manifold'] if x['rank'] == 16)
        out[str(L)] = dict(
            attempt=att, job_id=s['job_id'], gpu=s['gpu'], commit=s['commit'], cohort=s['cohort'],
            summary_file=str(sp.relative_to(lane.parent.parent)), summary_sha256=sha(sp),
            result_file=str(rp.relative_to(lane.parent.parent)), result_sha256=sha(rp),
            cell_fom=dict(name=CELL_FOM, gpu_ms=fom['gpu_ms'], worst_evolved_percent=fom['worst_evolved_percent']),
            rows={k: dict(arm=v, worst_evolved_percent=arms[v]['worst_evolved_percent'],
                          median_evolved_percent=arms[v]['median_evolved_percent'], gpu_ms=arms[v]['gpu_ms'],
                          speedup_vs_cell_fom=fom['gpu_ms'] / arms[v]['gpu_ms']) for k, v in ARMS.items()},
            snapshots=dict(snapshot_sha256=r['snapshots']['snapshot_sha256'], shape=[r['snapshots']['snapshots']]),
            pod=dict(kmax=r['pod']['kmax'], modes_sha256=r['pod']['modes_sha256'], eigenvalues_top16=r['pod']['eigenvalues'][:16]),
            qman16={k: q16[k] for k in ('ridge', 'heldout_relative', 'bank_sha256', 'weight_frobenius_norm',
                                        'snapshot_relative_linear_only', 'snapshot_relative_with_quadratic', 'split')})
    Path(__file__).with_name('inputs').mkdir(exist_ok=True)
    Path(__file__).with_name('inputs').joinpath('dev-reference.json').write_text(json.dumps(out, indent=1) + '\n')
    for L, v in out.items():
        print(L, v['job_id'], v['gpu'], {k: (round(x['worst_evolved_percent'], 4), round(x['speedup_vs_cell_fom'], 3)) for k, x in v['rows'].items()})


if __name__ == '__main__':
    main()
