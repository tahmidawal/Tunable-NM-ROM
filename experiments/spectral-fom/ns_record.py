"""Record (no rerun) the NS 3D spectral comparison from the ns3d-shift-head lane: its FOM CNAB2 is already Fourier
pseudo-spectral (2/3 dealiasing, CN viscous step, AB2 advection).  Uses the lane's own table code
(`write_tables.fom_rows / rom_rows / comparator / most_accurate`, commit 708c70fe) on its pulled summary.json files,
so every number here is the lane's own, read from JSON.  Writes lane-ref/ns3d.json.

Rows: accurate = head k = k_selected at the frozen ladder (dt, sweeps); fast = bank span R' = 8 (the lane's frozen
fast setting).  Spectral FOM columns: (a) the fastest stable CNAB2 setting at least as accurate as the arm (the lane's
comparator); (b) the most accurate non-reference CNAB2 setting.  Ratio = CNAB2 ms / ROM ms (< 1: CNAB2 faster).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

NS = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-ns3d-shift-head/experiments/ns3d-shift-head')
spec = importlib.util.spec_from_file_location('wt', NS / 'write_tables.py')
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
JOBS = [('a2_h32', 'development'), ('a2_h64', 'development'), ('a3_h96', 'development'), ('b2_heldout96', 'held-out')]
frozen = json.loads((NS / 'frozen' / 'frozen_settings.json').read_text())
out = dict(source_lane='exp/2026-09-23-ns3d-shift-head', source_commit='708c70fe',
           frozen_settings_sha256=hashlib.sha256((NS / 'frozen' / 'frozen_settings.json').read_bytes()).hexdigest(),
           note='CNAB2 is the lane FOM and is Fourier pseudo-spectral; the lane itself labels it not rule-compliant as an iterative FOM. Not rerun here.',
           rows=[])
for job, cohort in JOBS:
    s, v, d = W.load(job)
    cfg = s['config']
    foms = W.fom_rows(s)
    roms = [r for r in W.rom_rows(s) if r['dt'] == frozen['dt'] and r['iters'] == frozen['iters']]
    acc = [r for r in roms if r['kind'] == 'head' and r['k'] == frozen['k']]
    fast = [r for r in roms if r['kind'] == 'span' and r['rank'] == 8]
    assert len(acc) == 1 and len(fast) == 1, (job, len(acc), len(fast))
    best = W.most_accurate(foms, 'CNAB2')
    for role, r in (('rom_accurate', acc[0]), ('rom_fast', fast[0])):
        c = W.comparator(foms, 'CNAB2', r['worst'])
        out['rows'].append(dict(job=job, job_id=s['job_id'], cohort=cohort, mesh=cfg['n'], gpu=s.get('gpu'),
                                commit=s.get('source_commit'), role=role, arm=r['label'], worst=r['worst'], median=r['median'],
                                rom_ms=r['ms'],
                                cnab2_matched=None if c is None else dict(label=c['label'], worst=c['worst'], ms=c['ms']),
                                ratio_matched=None if c is None else c['ms'] / r['ms'],
                                cnab2_most_accurate=dict(label=best['label'], worst=best['worst'], ms=best['ms']),
                                ratio_most_accurate=best['ms'] / r['ms'],
                                timing_gates=dict(fast_block=s['timing'].get('neighbour_gate_fast_block_passed'),
                                                  after_cooldown=s['timing'].get('neighbour_gate_after_cooldown_passed'),
                                                  passed=s['timing'].get('neighbour_gate_passed')),
                                summary=str((d / 'summary.json').relative_to(NS.parents[1])),
                                summary_sha256=W.sha(d / 'summary.json')))
Path('lane-ref/ns3d.json').write_text(json.dumps(out, indent=1) + '\n')
for r in out['rows']:
    print(r['job'], r['mesh'], r['role'], f"{100*r['worst']:.3f}%", f"{r['rom_ms']:.3f}ms",
          r['cnab2_matched'] and (r['cnab2_matched']['label'], f"{r['cnab2_matched']['ms']:.3f}"), f"{r['ratio_matched']:.2f}" if r['ratio_matched'] else None,
          r['cnab2_most_accurate']['label'], f"{r['ratio_most_accurate']:.2f}", r['timing_gates'])
