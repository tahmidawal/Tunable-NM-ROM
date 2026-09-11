"""Additive linear enrichment retaining the accepted nonlinear K32 head."""
import numpy as np
import pilot as base
from pilot import jax,jnp
from fresh_models import head_apply,head_geometry,tree_to_npz


def build(endpoints,initializers,out):
    original=endpoints['trained_phase'];capacity=endpoints['trained_phase40']
    p=original['p'];linear32,center=initializers['trained_phase'];linear40,center40=initializers['trained_phase40']
    np.testing.assert_allclose(center,center40,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(linear32,linear40[:,:32],atol=1e-10,rtol=1e-10)
    basis=jnp.asarray(linear40[:,32:40])
    enriched={**p,'linear':jnp.concatenate((p['linear'],basis),axis=1),
              'l1':{**p['l1'],'w':jnp.concatenate((p['l1']['w'],jnp.zeros((8,p['l1']['w'].shape[1]))),axis=0)}}
    np.testing.assert_allclose(original['codes'],capacity['codes'][:,:32],atol=1e-9,rtol=1e-9)
    codes=jnp.concatenate((original['codes'],capacity['codes'][:,32:40]),axis=1)
    endpoint=dict(p=enriched,frozen=original['frozen'],codes=codes)
    rng=np.random.default_rng(691217);errors=[];ranks=[];arrays={}
    for ii in np.linspace(0,len(codes)-1,6,dtype=int):
        z=original['codes'][ii];w=jnp.asarray(rng.normal(size=32));zz=jnp.concatenate((z,jnp.zeros(8)));ww=jnp.concatenate((w,jnp.zeros(8)))
        a,b,jac,curve=head_geometry(p,original['frozen'],z,w,'mlp')
        aa,bb,jj,cc=head_geometry(enriched,original['frozen'],zz,ww,'mlp')
        errors.append(max(float(jnp.max(abs(a-aa))),float(jnp.max(abs(b-bb))),float(jnp.max(abs(jac-jj[:,:32]))),float(jnp.max(abs(curve-cc))),float(jnp.max(abs(jj[:,32:]-basis)))))
        singular=np.linalg.svd(np.asarray(jj),compute_uv=False);ranks.append(float(singular[-1]/singular[0]))
    assert max(errors)<1e-10 and min(ranks)>1e-8
    tree_to_npz(out/'head_trained_nested40.npz',endpoint)
    np.savez_compressed(out/'nested_basis.npz',basis=np.asarray(basis),linear32=linear32,linear40=linear40,center=center,
                        inclusion_errors=np.asarray(errors),sampled_rank_ratios=np.asarray(ranks))
    record=dict(method='trained_nested40',internal_configuration_dimension=40,internal_phase_dimension=80,weak_equations=64,
        construction='h32(z)+B8*y; preserveall nonlinearK32parameters, append8zero firstlayerrows and8fixedlinearcolumns.',
        basis_units='Original standardizedtraining-PCAlinearcolumns32:40 inorthonormalbankcoefficients; newyare dimensionless standardizedscores.',
        max_value_tangent_jacobian_curvature_inclusion_error=max(errors),minimum_sampled_rank_ratio=min(ranks),
        original_checkpoint_parameters_preserved=True,retrained=False,basis_sha256=base.array_sha(basis),checkpoint_sha256=base.sha(out/'head_trained_nested40.npz'))
    return endpoint,(np.concatenate((linear32,np.asarray(basis)),axis=1),center),record
