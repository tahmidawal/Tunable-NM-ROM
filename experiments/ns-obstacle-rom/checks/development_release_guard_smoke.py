"""Pure resource/path regression; no native work or admission is generated."""
from pathlib import Path
import copy,json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from development_handoff_launch2d import validate_release_contract
old=json.loads((root/'staging/development-materialize01/HANDOFF-RELEASE.json').read_text())
try:validate_release_contract(old)
except ValueError:pass
else:raise AssertionError('original incomplete release accepted')
proposed=copy.deepcopy(old);job=old['job_directory'].replace('materialize01','materialize02')
proposed.update(job_directory=job,output_directory=job+'/results/population',other_active_disk_reservation_bytes=2415919104,remaining_development_reservation_seconds=3360)
validate_release_contract(proposed)
for key in ['authorized_stage','job_directory','output_directory','other_active_disk_reservation_bytes','allocation_seconds','outer_timeout_seconds','outer_kill_grace_seconds','debit_from_existing_development_reservation_seconds','remaining_development_reservation_seconds','fom_reservation_unchanged_seconds']:
    bad=copy.deepcopy(proposed);del bad[key]
    try:validate_release_contract(bad)
    except ValueError:pass
    else:raise AssertionError(key)
for value in [None,-1,True,'2415919104',0]:
    bad=copy.deepcopy(proposed);bad['other_active_disk_reservation_bytes']=value
    try:validate_release_contract(bad)
    except ValueError:pass
    else:raise AssertionError(value)
assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
print('PASS: prior failure rejects; complete resource contract accepts; missing/type/path/budget guards reject before native imports')
