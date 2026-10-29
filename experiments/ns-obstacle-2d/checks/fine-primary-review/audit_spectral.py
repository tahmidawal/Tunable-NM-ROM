"""Independent saved sparse-array algebra only; no FE assembly or solve."""
import hashlib,json,sys,resource
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import norm as snorm
p=Path(sys.argv[1])/'results/preparation/prepared';m=json.loads((p/'manifest.json').read_text());bound=m['coefficient_files'];inputs={}
def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def verified(n):
 f=p/n;h=sha(f);assert h==bound[n];inputs[n]=h;return f
def arr(n):
 with np.load(verified('replay_'+n+'.npz'),allow_pickle=False) as z:return z['values']
def digest(*a):
 h=hashlib.sha256()
 for v in a:
  v=np.ascontiguousarray(v);h.update(str((v.shape,v.dtype.str)).encode());h.update(v.tobytes())
 return h.hexdigest()
def sparse_hash(a):
 a=a.tocsc(copy=True);a.sum_duplicates();a.sort_indices();return digest(np.asarray(a.shape,dtype=np.int64),a.indptr,a.indices,a.data)
mat={n:load_npz(verified('replay_metric_'+n+'.npz')).tocsr() for n in ('velocity_h1','velocity_mass','pressure_h1','pressure_mass','velocity_scalar_h1','velocity_scalar_mass')}
assert all(np.isfinite(a.data).all() for a in mat.values())
with np.load(verified('coefficients.npz'),allow_pickle=False) as z:c={k:z[k] for k in z.files}
sids=arr('scalar_free_ids');vids=arr('velocity_free_ids');maps=arr('component_maps');sm=arr('velocity_scalar_modes');qm=arr('pressure_eigenmodes');rhs=arr('supremizer_rhs')
assert np.array_equal(np.sort(maps.ravel()),np.arange(mat['velocity_h1'].shape[0]))
assert np.array_equal(np.sort(maps[:,sids].ravel()),vids)
hs=mat['velocity_scalar_h1'][sids][:,sids];ms=mat['velocity_scalar_mass'][sids][:,sids]
eigen={}
for name,a,b,x in [('eigen_velocity',hs,ms,sm[sids]),('eigen_scalar',mat['pressure_h1'],mat['pressure_mass'],qm)]:
 meta=m[name];assert sparse_hash(a)==meta['h1_sha256'] and sparse_hash(b)==meta['mass_sha256']
 lam=np.array(meta['eigenvalues_H1']);ax=a@x;bx=b@x
 err=np.linalg.norm(ax-bx*lam,axis=0)/np.maximum(np.linalg.norm(ax,axis=0)+abs(lam)*np.linalg.norm(bx,axis=0),1e-300)
 gram=x.T@bx;gap=float(np.linalg.norm(gram-np.eye(len(lam)),2))
 assert max(err)<1e-9 and gap<1e-10
 assert np.max(abs(err-np.array(meta['oriented_residuals'])))<1e-13
 eigen[name]={'residuals':err.tolist(),'mass_gram_operator_error':gap,'mass_gram':gram.tolist(),'h1_sha256':sparse_hash(a),'mass_sha256':sparse_hash(b),'look_ahead_not_recomputed':True}
blocks={}
for label,v,s in [('H1',mat['velocity_h1'],mat['velocity_scalar_h1']),('L2',mat['velocity_mass'],mat['velocity_scalar_mass'])]:
 for i in range(2):
  denom=snorm(v[maps[i]][:,maps[i]])
  for j in range(2):
   b=v[maps[i]][:,maps[j]];e=float(snorm(b-s if i==j else b)/denom);assert e<1e-12;blocks[f'{label}_{i}_{j}']=e
sup=c['supremizers'];hv=mat['velocity_h1'][vids][:,vids];res=float(np.linalg.norm(hv@sup[vids]-rhs)/max(np.linalg.norm(rhs),1e-300));assert res<1e-10
assert abs(res-m['supremizer_residual'])<1e-13
mask=np.ones(len(sup),bool);mask[vids]=False;assert np.max(abs(sup[mask]),initial=0)==0
cv=c['momentum_tests'].T@(mat['velocity_h1']@c['momentum_tests']);cq=c['continuity_tests'].T@(mat['pressure_mass']@c['continuity_tests']);cp=c['pressure'].T@(mat['pressure_mass']@c['pressure'])
gaps={}
for name,g,size in [('momentum_H1_gram',cv,64),('continuity_L2_gram',cq,16),('pressure_L2_gram',cp,8)]:
 gap=float(np.max(abs(g-np.eye(size))));assert gap<1e-9
 assert np.max(abs(g-np.array(m[name])))<1e-10;gaps[name]=gap
# Replay the actual velocity test candidate selection/transform without solving.
selection=m['momentum_selection'];indices=np.array(selection['candidate_indices']);transform=np.array(selection['transform']);ev=np.zeros((len(sup),2*sm.shape[1]))
for j in range(sm.shape[1]):
 for component in range(2):ev[maps[component],2*j+component]=sm[:,j]
reconstructed=np.c_[sup,ev][:,indices]@transform;selection_gap=float(np.max(abs(reconstructed-c['momentum_tests'])));assert selection_gap<1e-10
print(json.dumps({'schema':'ns-obstacle-independent-fine-spectral-review-v1','passed':True,'job_id':'6633574','scope':'saved sparse matrices and retained modes/coefficient algebra; no new assembly/factorization/eigensolve','prepared_manifest_sha256':sha(p/'manifest.json'),'inputs':inputs,'eigen':eigen,'component_blocks':blocks,'supremizer_residual':res,'metric_gram_max_absolute_errors':gaps,'momentum_H1_gram':cv.tolist(),'continuity_L2_gram':cq.tolist(),'pressure_L2_gram':cp.tolist(),'momentum_candidate_reconstruction_max_absolute_error':selection_gap,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
