"""Matched physical coefficient data for a wide head on the frozen global control."""
from common import *
from build_extended_span import read


def prepare(source,train,dev,out):
    rec=json.loads((source/'result.json').read_text());assert rec['status']=='complete_stable_global_controls'
    assert digest(source/'stable_global_basis.npz')==rec['basis_sha256']
    basis=jnp.asarray(np.load(source/'stable_global_basis.npz')['coefficients'][:960]);Q=basis.reshape(960,-1)*.5
    A,tn,tl,tp,ts=read(train);B,vn,vl,vp,vs=read(dev)
    assert tl==[v['case'] for v in rec['training_sources']] and vl==rec['dev_labels']
    assert [v['sha256'] for v in ts]==[v['sha256'] for v in rec['training_sources']]
    assert [v['sha256'] for v in vs]==[v['sha256'] for v in rec['dev_sources']]
    Ct=A@Q.T*.5;Cd=B@Q.T*.5
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'pod_basis.npz',coefficients=np.asarray(basis))
    np.savez_compressed(out/'reduced_data.npz',train_coefficients=np.asarray(Ct),dev_coefficients=np.asarray(Cd),
        train_norm=np.asarray(tn),dev_norm=np.asarray(vn),train_missing=np.asarray(jnp.maximum(tn**2-jnp.sum(Ct*Ct,axis=1),0)),
        dev_missing=np.asarray(jnp.maximum(vn**2-jnp.sum(Cd*Cd,axis=1),0)),train_parameters=np.asarray(tp),dev_parameters=np.asarray(vp))
    save(out/'result.json',dict(status='complete_stable_global_head_data',basis_sha256=digest(out/'pod_basis.npz'),
        reduced_data_sha256=digest(out/'reduced_data.npz'),construction=rec['construction'],source_basis_sha256=rec['basis_sha256'],
        train_labels=tl,dev_labels=vl,training_sources=ts,dev_sources=vs,final_test_opened=False))
