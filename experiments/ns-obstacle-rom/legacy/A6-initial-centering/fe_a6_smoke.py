"""Synthetic A6 source checks only: no NGSolve import, FE problem, or training."""
from pathlib import Path
import json, sys, time
import numpy as np
from scipy.sparse import csc_matrix, diags
from fe_pod_a6 import (coefficient_pod_a6, validate_actual_pod, mass_pod, metric_qr,
                       check_cluster_cuts, orient_clusters, first_numerical_max,
                       expand_components, check_component_blocks)
from fe_controls_a6 import boundary_control, gradient_control
from fe_adapter import bounded_factor, bounded_eigenmodes, adapter_digest
from fe_algebra import coefficient_pod, verify_reconstructed_modes


def rejected(function, message=None):
    try: function()
    except (ValueError, ArithmeticError) as exc:
        if message is not None and message not in str(exc): raise
        return True
    raise AssertionError('required rejection did not occur')


def run():
    start = time.perf_counter(); rng = np.random.default_rng(27102027)
    n = 40; w = rng.uniform(.3, 1., n)
    vm = rng.normal(size=(n, 2, 18)); vc = rng.normal(size=(18, 9))
    pm = rng.normal(size=(n, 12)); pm[:, 0] = 1.; pc = rng.normal(size=(12, 9))
    one = np.zeros(12); one[0] = 1.
    flat = vm.reshape(2*n, 18); mv = flat.T@(np.repeat(w, 2)[:, None]*flat)
    mp = pm.T@(w[:, None]*pm)
    u = np.einsum('ndk,ks->snd', vm, vc); p = (pm@pc).T
    def check(cu, pu, vp, pp, vmap=vm, pmap=pm, massv=mv, massp=mp, onep=one):
        pod = coefficient_pod_a6(cu, vp, pu, pp, onep, w, massv, massp)
        au = np.einsum('ndk,kr->ndr', vmap, pod['velocity_coefficients'])
        ap = pmap@pod['pressure_coefficients']
        _, vu = validate_actual_pod(au, vp, w, pod['metadata']['velocity'])
        gp, vpp = validate_actual_pod(ap, pp, w, pod['metadata']['pressure'], True)
        assert np.allclose(gp, ap.T@(w[:, None]*ap))
        return pod, {'velocity': vu, 'pressure': vpp}
    pod, healthy = check(vc, pc, u, p)
    q, r, qr = metric_qr(vc, mv, 9, np.linalg.svd((u.reshape(9,-1)*np.repeat(np.sqrt(w),2)).T, compute_uv=False)[0])
    defect = vc[:, qr['permutation']]-q@r
    direct_backward = np.sqrt(np.sum(defect*(mv@defect)))/qr['reference_scale']
    assert abs(direct_backward-qr['normalized_backward_error']) < 1e-15
    assert sorted(qr['permutation']) == list(range(9)) and not qr['pivoting_is_rank_cutoff']
    # Same resolved near-cutoff inputs: preserve original rank while replacing the quotient.
    left, _ = np.linalg.qr(rng.normal(size=(18, 2))); right, _ = np.linalg.qr(rng.normal(size=(9, 2)))
    tiny_vc = (left*np.array([1., 4e-12]))@right.T
    om, _ = np.linalg.qr(rng.normal(size=(2*n, 18)))
    om = (om/np.repeat(np.sqrt(w),2)[:,None]).reshape(n,2,18)
    tiny_u = np.einsum('ndk,ks->snd', om, tiny_vc)
    tiny_mv = om.reshape(2*n,18).T@(np.repeat(w,2)[:,None]*om.reshape(2*n,18))
    tiny_pod, tiny = check(tiny_vc, pc, tiny_u, p, vmap=om, massv=tiny_mv)
    legacy = coefficient_pod(tiny_vc, tiny_u, pc, p, one, w)
    legacy_u = np.einsum('ndk,kr->ndr',om,legacy['velocity_coefficients'])
    legacy_rejected = rejected(lambda: verify_reconstructed_modes(legacy_u,legacy['velocity_values'],w,'velocity'))
    assert tiny_pod['metadata']['velocity']['numerical_rank'] == 2
    pmap = rng.normal(size=(n,2)); pmap -= (w@pmap/w.sum())[None,:]
    pmap, _ = np.linalg.qr(np.sqrt(w[:,None])*pmap)
    pmap = np.c_[np.ones(n),pmap/np.sqrt(w[:,None])]
    tiny_pc = np.r_[np.zeros((1,9)),np.diag([1.,4e-12])@right.T]
    tiny_pressure = (pmap@tiny_pc).T; onep = np.array([1.,0.,0.]); massp=pmap.T@(w[:,None]*pmap)
    tiny_ppod, tiny_p = check(vc,tiny_pc,u,tiny_pressure,pmap=pmap,massp=massp,onep=onep)
    assert tiny_ppod['metadata']['pressure']['numerical_rank'] == 2
    # Zero centered pressure is constant-only; zero velocity has no eligible prefix.
    pc0=one[:,None]*np.arange(1.,10.)[None,:]
    constant_pod, constant_check=check(vc,pc0,u,(pm@pc0).T)
    assert constant_pod['pressure_coefficients'].shape[1] == 1
    zero_velocity_rejected=rejected(lambda: coefficient_pod_a6(vc*0,u*0,pc,p,one,w,mv,mp),'no registered velocity prefix')
    # Scaling cannot alter the normalized velocity cluster policy/orientation.
    amplitude=[]
    for scale in (1e-5,1e5):
        scaled, sm=mass_pod(vc*scale,u*scale,w,mv,8,(2,4,8))
        amplitude.append(float(np.linalg.norm(scaled-pod['velocity_coefficients'])))
        assert amplitude[-1]<1e-10
    cluster_rejections={
        'registered_prefix':rejected(lambda: check_cluster_cuts([1.,.5,.5,.1],[2]),'splits'),
        'rank_boundary':rejected(lambda: check_cluster_cuts([1.,1.001e-12,.999e-12,0.],[2]),'splits'),
        'retained_maximum':rejected(lambda: check_cluster_cuts([9.,8.,7.,6.,5.,4.,3.,2.,2.],[8]),'splits'),
        'pressure_tail':rejected(lambda: check_cluster_cuts([1.,1.,.1],[1]),'splits')}
    # A near-equal pivot score is a numerical tie; the smallest declared index wins.
    assert first_numerical_max([1.+8*np.finfo(float).eps,1.],[8,2])==1
    rotation,_=np.linalg.qr(rng.normal(size=(2,2)))
    canonical, _=orient_clusters(np.eye(2),[(0,2)])
    rotated, _=orient_clusters(rotation,[(0,2)])
    assert np.allclose(canonical,rotated,atol=1e-14)
    # Explicit component expansion works with a deliberately scrambled coefficient layout.
    ns=8; maps=rng.permutation(2*ns).reshape(2,ns)
    scalar_mass=diags(np.linspace(1,2,ns),format='csc'); scalar_h1=diags(np.arange(1,ns+1)*scalar_mass.diagonal(),format='csc')
    vector_mass=np.zeros((2*ns,2*ns)); vector_h1=vector_mass.copy()
    for ids in maps:
        vector_mass[np.ix_(ids,ids)]=scalar_mass.toarray(); vector_h1[np.ix_(ids,ids)]=scalar_h1.toarray()
    blocks=check_component_blocks(csc_matrix(vector_h1),csc_matrix(vector_mass),scalar_h1,scalar_mass,maps)
    scalar_modes=np.eye(ns)[:,:4]/np.sqrt(scalar_mass.diagonal())[:,None]
    expanded=expand_components(scalar_modes,maps,2*ns)
    assert np.linalg.matrix_rank(expanded)==8
    assert np.allclose(expanded.T@vector_mass@expanded,np.eye(8))
    corrupted=vector_h1.copy(); corrupted[maps[0,0],maps[1,0]]=1e-4
    cross_rejected=rejected(lambda: check_component_blocks(csc_matrix(corrupted),csc_matrix(vector_mass),scalar_h1,scalar_mass,maps))
    m=diags(np.linspace(1.,2.,40),format='csc'); h=diags(np.arange(1,41)*m.diagonal(),format='csc')
    fac,_=bounded_factor(h); _, eig=bounded_eigenmodes(h,m,fac.solve,6)
    assert eig['gap_checked'] and not eig['scalar_spectral_count_verified']
    # Independent signs/closure and off-diagonal convention, with failing variants.
    theta=(np.arange(32)+.5)*2*np.pi/32; radius=.05; center=np.array([.2,.2])
    radial=np.c_[np.cos(theta),np.sin(theta)]
    points=np.r_[np.array([[0,.205],[2.2,.205],[1.1,0],[1.1,.41]]),center+radius*radial]
    normals=np.r_[np.array([[-1.,0],[1.,0],[0,-1.],[0,1.]]),-radial]
    labels=np.array(['inlet','outlet','wall_bottom','wall_top']+['obstacle']*32)
    weights=np.r_[[.41,.41,2.2,2.2],np.full(32,2*np.pi*radius/32)]
    normals_check=boundary_control(points,weights,normals,labels,center)
    flipped=normals.copy(); flipped[4:]*=-1
    normal_sign_rejected=rejected(lambda: boundary_control(points,weights,flipped,labels,center))
    broken_w=weights.copy(); broken_w[4]*=1.1
    closure_rejected=rejected(lambda: boundary_control(points,broken_w,normals,labels,center))
    value=np.c_[points[:,1],np.zeros(len(points))]; gradient=np.zeros((len(points),2,2)); gradient[:,0,1]=1
    grad=gradient_control(points,value,gradient,0)
    transpose_rejected=rejected(lambda: gradient_control(points,value,gradient.transpose(0,2,1),0))
    assert 'ngsolve' not in sys.modules
    return {'scope':'A6 synthetic coefficient/matrix/analytic arrays only; no FE execution',
        'passed':True,'physical_admission':False,'NGSolve_imported':False,'adapter_source_sha256':adapter_digest(),
        'healthy':healthy,'QR_permutation_check':qr,'near_cutoff_velocity':tiny,
        'near_cutoff_pressure':tiny_p,'legacy_quotient_velocity_rejected':legacy_rejected,
        'constant_pressure_count':1,'zero_velocity_rejected':zero_velocity_rejected,
        'amplitude_scaled_mode_differences':amplitude,'cluster_cuts_rejected':cluster_rejections,
        'component_blocks':blocks,'explicit_component_rank':int(np.linalg.matrix_rank(expanded)),
        'cross_block_rejected':cross_rejected,'scalar_eigen':eig,'normal_control':normals_check,
        'normal_sign_rejected':normal_sign_rejected,'closure_rejected':closure_rejected,
        'gradient_control':grad,'gradient_transpose_rejected':transpose_rejected,'seconds':time.perf_counter()-start}


if __name__=='__main__': print(json.dumps(run(),indent=2,allow_nan=False))
