"""Extract immutable old invocation states, never modify a historical result."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
out = {}
for n, attempt in [(256, 'plin256'), (1024, 'plin1024b')]:
    path = HERE / 'runs' / attempt / 'archive/output/result.json'
    source = json.loads(path.read_text())
    rows = []
    for x in source['invocations']:
        if x['rep'] != 0 or x.get('model') not in ('new_K32', 'incumbent') or 'latent' not in x:
            continue
        if not x['name'].startswith('q') or '_ccrule' in x['name']:
            continue
        rows.append({k: x.get(k) for k in ['name', 'model', 'case', 'q', 'latent',
                    'correction_coefficients', 'same_grid_error', 'artifact', 'field_sha256']})
    out[str(n)] = {k: source[k] for k in ['config', 'cohort', 'checkpoints', 'directions', 'reconstruction']}
    out[str(n)].update(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       result_path=str(path.relative_to(HERE)),
                       original_source_commit=source['commit'], job_id=source['job_id'],
                       solved_states=rows)
(HERE / 'references/oracle-history.json').write_text(json.dumps(out, indent=2) + '\n')
cfg = dict(attempt='plorc01', history='references/oracle-history.json', intervals=[256, 1024],
           recon_starts=8, recon_budget=400, stationarity_tolerance=1e-6,
           original_protocol='Original frozen checkpoint, cohort and q sets; no timed invocation changes.',
           extra_starts='Retained same-q solved latent from each test rule and previous-q best latent; original eight-start result retained separately.',
           endpoint='Direct free-bank QR projection, with zero nonlinear iterations.',
           acceptance={'metric_direct_field_atol': 2e-7, 'gradient_atol': 2e-7,
                       'q0_original_atol': 2e-7, 'old_floor_atol': 2e-7,
                       'best_found_solved_bracket_atol': 2e-7})
(HERE / 'config-oracle.json').write_text(json.dumps(cfg, indent=2) + '\n')
print('Retained historical states:', sum(len(x['solved_states']) for x in out.values()))
