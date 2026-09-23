"""Aggregate the audited per-job summaries into checks/lane-summary.json (Table-1 rows before/after, held-out, 6.3 X)."""
import hashlib
import json
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]


def main():
    out = dict(lane='exp/2026-09-23-burgers2d-speed', meshes={})
    for L in (256, 512, 1024):
        pb, ph = LANE / 'checks' / f'b{L}-summary.json', LANE / 'checks' / f'h{L}-summary.json'
        b, h = json.loads(pb.read_text()), json.loads(ph.read_text())
        sb, sh = b['selection'], h['selection']
        row = lambda a: {k: a.get(k) for k in ('timed_arm', 'mode', 'worst_evolved_percent', 'median_evolved_percent',
                                              'median_gpu_ms', 'other_mode_ms', 'certificate', 'rho_max_cert',
                                              'rho_max_confirmation', 'total_iterations_per_case', 'own_fom',
                                              'own_speedup')}
        out['meshes'][L] = dict(
            dev6=dict(job=b['job_id'], gpu=b['gpu'], host=b['host'], accepted=b['accepted'], failed_gates=b['failed_gates'],
                      summary_sha256=hashlib.sha256(pb.read_bytes()).hexdigest(), fast_bar_percent=sb['fast_bar_percent'],
                      accurate=row(sb['accurate']), fast=row(sb['fast']), table1=sb['table1'],
                      before_parent_code_same_job={n: {k: v for k, v in x.items() if k != 'engineered_same_knob'}
                                                   for n, x in sb['parent_settings_this_job'].items()}),
            hold64=dict(job=h['job_id'], gpu=h['gpu'], host=h['host'], accepted=h['accepted'], failed_gates=h['failed_gates'],
                        summary_sha256=hashlib.sha256(ph.read_bytes()).hexdigest(), accurate=sh['accurate'],
                        fast=sh['fast'], table1=sh['table1'], parent_settings_this_job=sh['parent_settings_this_job']))
        if sb.get('general_path_section_6_3'):
            out['section_6_3_general_path'] = dict(sb['general_path_section_6_3'], mesh=L, job=b['job_id'])
    p = LANE / 'checks' / 'lane-summary.json'
    p.write_text(json.dumps(out, indent=1) + '\n')
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
