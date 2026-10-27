"""Add prospective A2 reporting layers without changing the archived pilot verdict."""
from pathlib import Path
import json
import numpy as np
from analyse import safe_ratio,sha


def main():
    here=Path(__file__).resolve().parent
    p=here/'runs/EXPORT.json';export=json.loads(p.read_text())
    raw=Path(export['result_path']);assert sha(raw)==export['result_sha256']
    j=json.loads(raw.read_text());ids=j['config']['cases'];layers={}
    for arm in ('k8','k16'):
        fields={str(i):j['cases'][str(i)]['field_distance'][f'{arm}_to_none'] for i in ids}
        field_mask={i:bool(np.isfinite(v) and v<=.00025) for i,v in fields.items()}
        gradient_masks={};full_bad_masks={};numeric_finite=[]
        for i in ids:
            m=j['cases'][str(i)][arm]['whole_residual']
            gf=np.asarray(m['g_full']);ge=np.asarray(m['g_sampled'])
            den=np.asarray(m['jacobian_norm_full'])*np.asarray(m['residual_norm_full'])
            dg=safe_ratio(np.linalg.norm(ge-gf,axis=1),den)
            gamma=safe_ratio(np.linalg.norm(gf,axis=1),den)
            numeric_finite.append(bool(np.isfinite(gf).all() and np.isfinite(ge).all() and
                                       np.isfinite(den).all() and np.isfinite(dg).all() and np.isfinite(gamma).all()))
            gradient_masks[str(i)]=(np.isfinite(dg) & (dg<=.001)).tolist()
            full_bad_masks[str(i)]=((~np.isfinite(gamma)) | (gamma>.001)).tolist()
        common=bool(j['Q1'][arm]['q1'] and j['Q1']['none']['q1'] and all(field_mask.values()))
        gradient_pass=bool(all(numeric_finite) and all(all(v) for v in gradient_masks.values()))
        bad=sum(sum(v) for v in full_bad_masks.values());steps=sum(len(v) for v in full_bad_masks.values())
        full_pass=bool(all(numeric_finite) and bad<=.01*steps)
        layers[arm]=dict(own_objective_Q1=j['Q1'][arm],full_residual_Q1=j['Q1']['none'],
            shared_field_bar=.00025,shared_field_to_none=fields,shared_field_case_mask=field_mask,
            common_layer_pass=common,extra_gradient_bar=.001,extra_gradient_step_mask=gradient_masks,
            extra_gradient_pass=gradient_pass,extra_full_stationarity_bar=.001,
            extra_full_stationarity_bad_step_mask=full_bad_masks,extra_full_stationarity_bad_steps=bad,
            extra_full_stationarity_steps=steps,extra_full_stationarity_fraction_limit=.01,
            extra_full_stationarity_pass=full_pass,finite_recomputed_numerically=all(numeric_finite))
    export['pilot_admission_layers']=dict(source_result_sha256=sha(raw),arms=layers,
        original_all_three_field_chain_pass=j['qualification']['field_chain'],
        original_strict_joint_verdict_unchanged=j['qualification']['passes_observed_population'],
        interpretation='Both archived supports fail common fields independently of the additional POD-only gradient gate.')
    amendment=here/'AMENDMENT-A2.md'
    export['reporting_amendment']=dict(path=str(amendment),sha256=sha(amendment),independent_design_gate='READY',
        common_profile_is_preregistered_HR_qualification=False,
        extra_POD_gradient_and_bank_rho_are_equivalent=False)
    helper=here/'pod_subjects.py'
    export['stageF_adapter']=dict(path=str(helper),sha256=sha(helper),independent_code_control_gate='READY',
        scope='five fixed no-HR integration schedules; new full64 Q1 required for each selected fine setting')
    p.write_text(json.dumps(export,indent=2)+'\n')


if __name__=='__main__':main()
