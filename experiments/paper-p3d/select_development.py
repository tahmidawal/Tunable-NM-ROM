"""Select frozen model bytes on native development error; never mix timing rows."""
import argparse
import hashlib
import json
from pathlib import Path

LANE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def select(attempts):
    rom=[];operators={};sources={}
    for attempt in attempts:
        folder=LANE/'runs'/attempt
        assert json.loads((folder/'COLLECTED.json').read_text())['checksums_verified']
        assert json.loads((folder/'audit-local.json').read_text())['passed']
        assert json.loads((folder/'retention-audit.json').read_text())['passed']
        out=folder/'archive/out';record=json.loads((out/'result.json').read_text());summary=json.loads((out/'summary.json').read_text())
        assert record['complete'] and not record['final_cohort_opened']
        sources[attempt]=dict(result_sha256=digest(out/'result.json'),source_commit=record['source_commit'],job_id=record['job_id'])
        rows={row['method']:row for row in summary['rows'] if row['intervals']==32}
        row=rows['nmrom_K16_q0_dense'];assert row['nonfinite_cases']==0 and row['nonstationary_cases']==0
        rom.append(dict(attempt=attempt,error=row['same_grid_error_worst'],
            checkpoint_identity=[digest(out/'bank.pkl'),digest(out/'head_K16.pkl')]))
        for entry in record['config']['operators']:
            row=rows[entry['name']];assert row['nonfinite_cases']==0
            operators.setdefault(entry['spec']['kind'],[]).append(dict(attempt=attempt,name=entry['name'],
                error=row['same_grid_error_worst'],entry=entry,checkpoint_identity=digest(out/'operators'/entry['name']/'best.pkl')))
    def best(candidates):
        winner=min(candidates,key=lambda row:row['error'])
        same=[row for row in candidates if row['checkpoint_identity']==winner['checkpoint_identity']]
        assert max(row['error'] for row in same)-min(row['error'] for row in same)<1e-12
        return same[-1] # latest verified real panel of identical selected model bytes
    return dict(schema='poisson3d-native-development-selection-v1',sources=sources,candidate_roms=rom,
        candidate_operators=operators,selected_rom=best(rom),selected_operators={kind:best(rows) for kind,rows in operators.items()},
        metric='worst native N32 development relative L2 field error',final_cohort_opened=False,
        timing_policy='Selection compares accuracy only. Timing ratios require the subsequent real selected-bundle panel.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempts',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(select(a.attempts),indent=2)+'\n')
