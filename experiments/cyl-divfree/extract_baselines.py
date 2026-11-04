"""Extract run-09 and POD-4 development metrics from their saved JSONs (read-only) into BASELINES.json.

Sources (other worktrees, read only, pinned by SHA256):
  run 09 : ns-obstacle-rom artifacts/neural-pod-guided-diagnostic09/metadata/results/diagnostic/status_007.json
  POD-4  : ns-obstacle-2d checks/neural-scanned-coordinate-diagnostic-plot-data/METADATA.json
           (actual_pod_recovery_extension: same weak-LS solver, mixed Galerkin, PCHIP)
"""
from pathlib import Path
import hashlib, json
WT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees')
RUN09 = WT/'2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom/artifacts/neural-pod-guided-diagnostic09/metadata/results/diagnostic/status_007.json'
POD = WT/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d/checks/neural-scanned-coordinate-diagnostic-plot-data/METADATA.json'
KEYS = ('velocity_relative', 'pressure_scaled', 'drag_relative', 'transverse_absolute', 'divergence', 'mass',
        'boundary_trace', 'velocity_gradient_relative', 'accuracy_targets_passed')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


out = {'sources': {'run09_status_007': {'path': str(RUN09), 'sha256': sha(RUN09)},
                   'pod_recovery_metadata': {'path': str(POD), 'sha256': sha(POD)}}, 'run09': {}, 'run09_floor': {}, 'POD4': {}}
r = json.loads(RUN09.read_text())['record']
assert r['model_sha256'].startswith('645d030088ed')
for c in r['cases']:
    out['run09'][str(c['Re'])] = dict({k: c['scores'][k] for k in KEYS}, solver_status=c['solver_status'], target_checks=c['scores']['target_checks'])
    out['run09_floor'][str(c['Re'])] = {k: c['representation_diagnostic'][k] for k in KEYS}
for c in json.loads(POD.read_text())['actual_pod_recovery_extension']['cases']:
    if c['rank'] != 4: continue
    out['POD4'].setdefault(c['method'], {})[str(c['Re'])] = dict({k: c['recorded_full_rule_scores'][k] for k in KEYS}, status=c['work']['status'])
Path(__file__).with_name('BASELINES.json').write_text(json.dumps(out, indent=1)+'\n')
print(json.dumps(out, indent=1)[:3000])
