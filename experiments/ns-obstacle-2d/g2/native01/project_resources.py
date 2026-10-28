"""Arithmetic-only projections from measured smoke07; never allocate projected arrays."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ROM=ROOT.parents[2]/'2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom'
GIB=2**30


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    matrix=json.loads((ROOT/'g2/CASE-MATRIX.DRAFT.json').read_text())
    baseline=matrix['baseline_G1_counts']
    rows=[]
    # Registered maximal pressure/velocity prefixes, test count, training count.
    d=2;rank=8;tests=64;training=9;vcandidates=8+96;qcandidates=1+8+32
    moment_rows=1+d+training*(d+d*d+tests)
    for case in matrix['cases']:
        projection=case['projection'];n=projection['volume_points'];nd=projection['dofs']
        # FE adapter keeps vv/vg and qq, constructs a weighted concatenation, and
        # calls dense SVD/QR on selected columns. The sum is an envelope, not a
        # simultaneous-liveness or LAPACK-workspace theorem.
        arrays={
            'candidate_velocity_values':n*d*vcandidates*8,
            'candidate_velocity_gradients':n*d*d*vcandidates*8,
            'candidate_pressure_values':n*qcandidates*8,
            'weighted_candidate_concat':n*(d+d*d)*vcandidates*8,
            'weighted_gradient_temporary':n*d*d*vcandidates*8,
            'selected_QR_or_SVD_matrix':n*(d+d*d)*tests*8,
            'selected_QR_output':n*(d+d*d)*tests*8,
            'fixed_velocity_test_values_and_gradients':n*(d+d*d)*tests*8,
            'training_velocity_pressure_values_and_gradients':n*(d+d*d+1+d)*training*8,
            'max_prefix_trial_values_and_gradients':n*(d+d*d)*(rank+rank)*8,
            'moment_matrix':n*moment_rows*8,
            'scaled_moment_matrix':n*moment_rows*8,
            'moment_scaling_abs_product_temporaries':2*n*moment_rows*8,
            'maximum_NNLS_selected_matrix':moment_rows*2560*8,
            'nine_mixed_coefficient_vectors':nd*training*8,
        }
        basis_names=list(arrays)[:10]
        fit_names=['moment_matrix','scaled_moment_matrix','moment_scaling_abs_product_temporaries','maximum_NNLS_selected_matrix']
        basis_envelope=sum(arrays[key] for key in basis_names)+arrays['nine_mixed_coefficient_vectors']
        fit_envelope=sum(arrays[key] for key in fit_names)+arrays['fixed_velocity_test_values_and_gradients']+arrays['max_prefix_trial_values_and_gradients']
        violations=list(projection['cap_violations'])
        if arrays['candidate_velocity_gradients']>GIB:violations.append('adapter_single_gradient_array_1GiB')
        if arrays['moment_matrix']>GIB:violations.append('quadrature_moment_matrix_1GiB')
        if projection['factor_estimate_plus_coarse_peak_bytes']-baseline['peak_rss_bytes']>8*GIB:
            violations.append('adapter_factor_estimate_8GiB_if_mixed_nnz_proxy')
        # Conservative 2x workspace contingency plus measured process baseline.
        estimate=2*max(basis_envelope,fit_envelope)+baseline['peak_rss_bytes']
        if estimate>16*GIB:violations.append('adapter_process_RSS_16GiB_projected_envelope')
        rows.append({'case':case['case'],'projected_volume_points':n,'projected_mixed_dofs':nd,
                     'explicit_float64_array_bytes':arrays,'basis_phase_array_envelope_bytes':basis_envelope,
                     'quadrature_fit_phase_array_envelope_bytes':fit_envelope,
                     'workspace_contingency_multiplier':2,'projected_working_set_envelope_bytes':estimate,
                     'existing_cap_violations':violations})
    available=shutil.disk_usage(ROOT).free
    # Full native source/dependency archive plus local archive/extraction/tracked
    # chunks/Git objects. This deliberately exceeds the measured smoke07 footprint.
    control_disk_envelope=3*GIB
    record={'status':'source_arithmetic_only_no_cap_changes_no_ladder_release',
            'baseline':baseline,'matrix_sha256':sha(ROOT/'g2/CASE-MATRIX.DRAFT.json'),
            'downstream_source_commit':subprocess.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'],cwd=ROM,text=True).strip(),
            'downstream_source_sha256':{name:sha(ROM/name) for name in ['fe_adapter.py','spaces.py','quadrature.py','bank.py']},
            'shape_assumptions':{'dimension':d,'training_states':training,'max_velocity_prefix':rank,'max_pressure_prefix':rank,
                                 'velocity_test_candidates':vcandidates,'pressure_test_candidates':qcandidates,
                                 'fixed_momentum_tests':tests,'moment_rows':moment_rows},
            'first_native_control':{'case':'new coarse DFG geometry with smoke07 geometry/orders/full quadrature',
                'measured_elements':baseline['elements'],'measured_dofs':baseline['dofs'],
                'measured_volume_points':baseline['volume_points'],'measured_boundary_points':567,
                'measured_FOM_peak_rss_bytes':baseline['peak_rss_bytes'],'measured_whole_batch_MaxRSS_KiB':1182176,
                'prospective_parent_plus_fresh_child_working_set_envelope_bytes':4*GIB,
                'guard_or_allocation':{'gpus':1,'CPUs':16,'memory_GiB':64,'wall_seconds':1200,
                    'FOM_memory_stop_GiB':48,'volume_point_cap':100000},
                'actual_new_G2_peak_memory_unmeasured':True,'fit_existing_caps':True,
                'retention_disk_envelope_bytes':control_disk_envelope},
            'storage':{'local_free_bytes_at_projection':available,'reserve_bytes':10*GIB,
                'control_envelope_plus_reserve_fits':control_disk_envelope+10*GIB<=available,
                'campaign_unique_data_cap_bytes':60*GIB,
                'ladder_cumulative_archive_and_Git_storage_projection_pending_native_measurement':True},
            'cases':rows,
            'limitations':['h^-2 and polynomial-order scaling are projections, not measured refined meshes.',
                'The adapter eigen/factor nnz differs from mixed Newton nnz; mixed nnz is a conservative proxy, not an exact prediction.',
                'Dense basis SVD/QR workspace, Python object overhead, allocator/JAX copies and training autodiff memory need native measurement.',
                'The 2x workspace envelope is a planning contingency, not a guaranteed upper bound.',
                'At most one method/prefix fitting phase should be live; no simultaneous-bank-and-POD assumption.',
                'Current quarter-h ladder violates point and downstream moment/array caps; no cap increase or quadrature weakening is authorized.',
                'First native control exercises producer/reload only, not downstream adapter/training.',
                'A bounded measured adapter control and exact cumulative storage projection precede any resource-only cap amendment.']}
    (ROOT/'g2/native01/RESOURCE-PROJECTION.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
