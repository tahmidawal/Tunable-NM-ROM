"""Synthetic no-data test of prospective selection and both final freezes."""
import copy,json,sys,tempfile
from pathlib import Path
import prepare_selected_final as S
import final_freeze as F

lane=Path(__file__).resolve().parent
primary=json.loads((lane/'developmentA.json').read_text());robust=json.loads((lane/'developmentB.json').read_text())
with tempfile.TemporaryDirectory(dir=lane/'smokes') as directory:
    fake=Path(directory);run=fake/'runs/fixture';source=run/'archive/out';source.mkdir(parents=True)
    old_s,old_f,old_args=S.__file__,F.__file__,sys.argv
    S.__file__=str(fake/'prepare_selected_final.py');F.__file__=str(fake/'final_freeze.py');sys.argv=['fixture','--attempt','fixture']
    try:
        rows=[];meshes=[]
        for n in primary['evaluation_intervals']:
            meta={}
            def row(name,err,ms,kind):
                meta[name]=dict(kind=kind)
                rows.append(dict(intervals=n,method=name,same_grid_current_evolved_worst=err,device_ms_median=ms,nonfinite_cases=0,cases_with_nonstationary_solves=0))
            for k in primary['latent_dimensions']:
                for q in primary['q_ladder']:
                    row(f'nmrom_K{k}_q{q}_dense',(.10 if k==16 else .015)/(1+q),10.,'nonlinear_rom')
            for base in primary['frozen_operators']:
                row(base,.003 if n==32 else .02,2.,'neural_operator')
                if n!=32:row(base+'_native_grid_interpolated',.002,3.,'neural_operator')
            row('fom_cn_cg_dt0.025_rtol1e-06',.0001,5.,'full_order')
            row('fom_cn_cg_dt0.05_rtol1e-04',.005,3.,'full_order')
            row('fom_cn_cg_dt0.1_rtol1e-02',.01,4.,'full_order')
            row('dst_exact',0.,1.,'full_order');row('linear_bank_exact',.002,1.,'linear_control')
            for rank in [32,64,96,128]:row(f'pod{rank}_exact',.001,1.,'linear_control')
            meshes.append(dict(intervals=n,methods=meta))
        operators=[dict(name=base,spec=dict(kind=kind)) for base,kind in zip(primary['frozen_operators'],['fno3d','unet3d','deeponet3d','transolver3d'])]
        for dest,cfg in [(source,primary),(source/'seedB',robust)]:
            dest.mkdir(exist_ok=True)
            record=dict(config=cfg,complete=True,final_cohort_opened=False,meshes=meshes,operators=operators,invocations=[{}],source_commit='synthetic_fixture',job_id='none')
            (dest/'result.json').write_text(json.dumps(record))
            for name in ['bank.pkl','head_K16.pkl','head_K32.pkl','pod_N32.pkl','pod_N64.pkl']+[f'operators/{base}/adapter.pkl' for base in primary['frozen_operators']]:
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic protocol fixture: not model weights')
        (source/'summary.json').write_text(json.dumps(dict(rows=rows)))
        for name in ['audit-local.json','audit-states-primary.json','audit-panel-primary.json','audit-field-seedB.json','audit-states-seedB.json','audit-panel-seedB.json','audit-frozen-replay-primary.json','audit-frozen-replay-seedB.json']:
            (run/name).write_text(json.dumps(dict(passed=True,synthetic_fixture=True)))
        S.main()
        selected=json.loads((fake/'final-selection.json').read_text());a=json.loads((fake/'finalA.json').read_text());b=json.loads((fake/'finalB.json').read_text())
        assert selected['selected_head']==32
        assert all('fom_cn_cg_dt0.1_rtol1e-02' not in names for names in selected['cg_methods_by_mesh'].values())
        assert all(name.endswith('_native_grid_interpolated') for name in selected['operator_methods_by_mesh']['64'])
        assert b['primary_freeze_sha256']==F.digest(fake/'final-primary-freeze.json')
        assert a['reserved_final_count']==b['reserved_final_count']==64
        assert not selected['final_data_accessed']
    finally:S.__file__,F.__file__,sys.argv=old_s,old_f,old_args
(lane/'smokes/selection-protocol.json').write_text(json.dumps(dict(passed=True,synthetic_fixture_only=True,final_parameters_or_fields_generated=False,head_not_selected_by_span=True,dominated_cg_rejected=True,one_transfer_per_operator=True,companion_bound_to_primary_freeze=True,final_case_count_unchanged=True),indent=2)+'\n')
print('SELECTION_PROTOCOL_SMOKE_PASS')
