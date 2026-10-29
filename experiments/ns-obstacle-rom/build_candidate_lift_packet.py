"""Freeze existing Re20 input bytes and minimal source package; no submission."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).parent
REF=ROOT.parents[2]/'2026-10-27-ns-obstacle-2d'/'experiments/ns-obstacle-2d'
SOURCE_COMMIT='a7c58adca'
CASES=[('p3g3h1','g2-ladder02','dfg_h1'),('p3g3h2','g2-ladder02','dfg_h2'),('p3g3h3','g2-h3','dfg_h3'),('p4g4h1','g2-p4trend','dfg_h1'),('p4g4h2','g2-p4trend','dfg_h2')]
NAMES=('config.json','mesh.vol.bin','stokes.sol','result.json','fe_layout.json','mesh_identity.json','reload_checks.json','EXPORT-MANIFEST.json')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 out=ROOT/'staging/candidate-lift01';out.mkdir(exist_ok=True);rows=[];cases=[]
 mirror=subprocess.check_output(['git','-c','gc.auto=0','rev-parse','origin/codeonly/exp/2026-10-27-ns-obstacle-2d'],text=True).strip()
 for identifier,archive,case in CASES:
  folder=REF/'artifacts'/archive/'results'/case;files={};layout=json.loads((folder/'fe_layout.json').read_text())['layout'];n=layout['velocity_ndof']+layout['pressure_ndof']
  for name in NAMES:
   path=folder/name;relative=str(path.relative_to(REF));raw=subprocess.check_output(['git','-c','gc.auto=0','show',SOURCE_COMMIT+':experiments/ns-obstacle-2d/'+relative])
   digest=sha(path)
   if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('retained byte differs from pinned source history')
   mirrored=subprocess.check_output(['git','-c','gc.auto=0','show',mirror+':experiments/ns-obstacle-2d/'+relative])
   if hashlib.sha256(mirrored).hexdigest()!=digest:raise ValueError('retained byte differs from pinned mirror history')
   files[name]=digest;rows.append({'source':relative,'destination':'cases/'+identifier+'/'+name,'sha256':digest,'bytes':len(raw)})
  cases.append({'id':identifier,'directory':'cases/'+identifier,'files':files,'mixed_dofs':n})
 old=ROOT/'staging/native-factor05/payload'
 for name in ('dependencies.py','DEPENDENCIES.json'):(out/name).write_bytes((old/name).read_bytes())
 (out/'REPAIR.json').write_text(json.dumps({'fresh_attempt':'candidate-lift01','purpose':'retained Re20 candidate scaled lift only; no solves/admission'})+'\n')
 script=(old/'native.sh').read_text().replace('ctol_nsobsrom_factor','ctol_nsobsrom_candlift').replace('00:20:00','00:04:00').replace('native-factor05','candidate-lift01').replace('results/interface/result.json','results/control/RESULT.json').replace('"$PY" native_factor_control.py --case inputcase --release CONTROL-RELEASE.json --out results/interface','"$PY" candidate_lift_control.py --plan CASES.json --release CONTROL-RELEASE.json --out results/control')
 script=script.replace('JOB_PHASE=source_verification','export ROM_ALLOCATION_START_UNIX_SECONDS=$("$PY" -c "import time; print(time.time())")\n"$PY" -c "import shutil; assert shutil.disk_usage(\'.\').free >= 12*2**30, \'ten-GiB reserve plus two-GiB exact control staging envelope\'"\nJOB_PHASE=source_verification',1)
 (out/'native.sh').write_text(script)
 bound=sum(105*c['mixed_dofs'] for c in cases)+sum(r['bytes']for r in rows if r['destination'].endswith('mesh.vol.bin'))+32*2**20
 plan={'schema':'ns-obstacle-retained-candidate-lift-control-plan-v1','cases':cases,'source_reference_commit':SOURCE_COMMIT,
  'reused_fine_control':{'directory':str(REF/'artifacts/g2-crossp02/results/immutable_lift'),'control_sha256':sha(REF/'artifacts/g2-crossp02/results/immutable_lift/control.json')},
  'allocation_debit_seconds':240,'remaining_dev_FOM_seconds':2160,'original_dev_FOM_seconds':2400,
  'runtime_estimate_not_measurement':{'setup_allowance_seconds':60,'five_sequential_matrix_application_controls_seconds':150,'finalization_seconds':30,'no_factorization_or_Newton':True},
  'resources':{'input_file_bytes':sum(r['bytes']for r in rows),'uncompressed_scientific_output_bound_bytes':bound,'remote_additional_envelope_bytes':2*2**30,'minimum_disk_reserve_bytes':10*2**30,'process_guard_bytes':16*2**30},
  'scientific_changes':False,'eligibility_created':False,'execution_released':False}
 (out/'CASES.json').write_text(json.dumps(plan,indent=2)+'\n')
 source=[ROOT/'candidate_lift_control.py',ROOT/'fom_native/SOURCE.json',*(ROOT/'fom_native'/name for name in json.loads((ROOT/'fom_native/SOURCE.json').read_text())['files'])]
 runtime=[{'source':str(p.relative_to(ROOT)),'destination':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}for p in source]
 for name in ('native.sh','dependencies.py','DEPENDENCIES.json','REPAIR.json','CASES.json'):
  p=out/name;runtime.append({'source':str(p.relative_to(ROOT)),'destination':name,'sha256':sha(p),'bytes':p.stat().st_size})
 inventory={'schema':'ns-obstacle-candidate-lift-input-inventory-v1','runtime':runtime,'producer':rows,'producer_git_commit':SOURCE_COMMIT,'producer_mirror_commit':mirror,'producer_prefix':'experiments/ns-obstacle-2d','all_producer_bytes_verified_against_pinned_history':True}
 (out/'INPUT-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
 print(json.dumps({'runtime_files':len(runtime),'producer_files':len(rows),'inventory_sha256':sha(out/'INPUT-INVENTORY.json'),'output_bound_bytes':bound,'input_bytes':sum(r['bytes']for r in rows)},indent=2))
if __name__=='__main__':main()
