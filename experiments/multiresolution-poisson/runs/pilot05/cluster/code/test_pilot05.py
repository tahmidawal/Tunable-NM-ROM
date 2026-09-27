"""Small GPU driver smoke; artifacts stay outside the scientific run directories."""
import json,os,sys,tempfile
from pathlib import Path
from speed_core import source_params,sha,np
import pilot05

cell=Path(__file__).resolve().parent
cfg=json.loads((cell/'config05.json').read_text())
cfg.update(intervals=[32],requested_modes=64,coarse_intervals=16,observation_intervals=16,
    reference_intervals=[64,128,256],existing_development_count=1,fresh_development_count=1,
    repetitions=1,warmup=0,burn_seconds=0.,expected_timed_invocations=20,expected_stationary_invocations=16)
draws=np.concatenate((source_params(cfg['existing_development_seed'],1),source_params(cfg['fresh_development_seed'],1)))
cfg['parameter_sha256']=sha(draws)
original=cell.parents[0]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
selected=cell/'runs/pilot04/checkpoints/original_relative.pkl'
if Path('in/model.pkl').exists():original=Path('in/model.pkl');selected=Path('in/selected.pkl')
os.environ.setdefault('COMMIT','local-driver-smoke');os.environ.setdefault('SLURM_JOB_ID','local-smoke')
with tempfile.TemporaryDirectory(prefix='poisson-speed-driver-') as temp:
    root=Path(temp);(root/'config.json').write_text(json.dumps(cfg))
    sys.argv=['pilot05','--config',str(root/'config.json'),'--checkpoint',str(original),
        '--selected-checkpoint',str(selected),'--out',str(root/'out')]
    pilot05.main();d=json.loads((root/'out/result.json').read_text())
    assert d['complete'] and len(d['rows'])==20 and len(d['stationary_rows'])==16
    assert all(x['passed'] for x in d['projection_parity'])
    assert all(x['coefficient_passed'] for x in d['coefficient_checks'])
    assert all(r['absolute_tau_threshold']==r['tau']*r['initial_residual'] for r in d['rows']+d['stationary_rows'] if r['model'])
    print('DRIVER-SMOKE-PASSED',len(d['rows']),len(d['stationary_rows']),flush=True)
