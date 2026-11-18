"""Physical Gram cache and train-only shared local POD, retaining all parent nodes."""
import argparse
from common import *
import data,local_polynomial as LP,tensor_metric as TM
from training_io import checkpoint,summary

def cohort(directories):
    rows=[]
    for folder in directories.split(','):
        folder=Path(folder);assert str(folder).startswith(os.environ['STUDY_NAMESPACE']+'/')
        record=json.loads((folder/'result.json').read_text())
        for r in record['cases']:
            if r['case'].startswith('fixture'):continue
            assert r['provisional_ladder_pass'];rows.append((folder,r))
    assert len({r['case'] for _,r in rows})==len(rows)
    return rows

def expand_matrix(U,nodes,root):
    rank=U.shape[1];T=jnp.zeros((root+nodes*U.shape[0],root+nodes*rank));T=T.at[:root,:root].set(jnp.eye(root))
    for j in range(nodes):T=T.at[root+j*U.shape[0]:root+(j+1)*U.shape[0],root+j*rank:root+(j+1)*rank].set(U)
    return T

@jax.jit
def reduced_fit(gram,rhs,norm2,transform):
    R=transform.T@gram@transform;b=rhs@transform
    c=jnp.linalg.solve(R+1e-10*jnp.eye(R.shape[0]),b.T).T
    err=jnp.sqrt(jnp.maximum(jnp.einsum('bi,ij,bj->b',c,R,c)-2*jnp.sum(c*b,axis=1)+norm2,0)/norm2)
    return c,err

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--span',required=True)
    ap.add_argument('--train',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--root',type=int,default=16)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);result=dict(meta=preflight(),status='building',root=a.root,
        local_side=6,nodes=52,depth=4,policy='swept',cases={'train':[],'dev':[]},normalization='current-time reference L2',final_test_opened=False)
    base=np.load(Path(a.span)/'pod_basis.npz')['coefficients'][:a.root]
    np.savez_compressed(out/'global_basis.npz',coefficients=base)
    x,w=data.grid(128,True);x=jnp.asarray(x);w=jnp.asarray(w)
    metric=TM.make_geometry(6);rule=TM.rule();rule2=TM.rule(order=32)
    result['integration']='64 aligned panels per axis, 24 Gauss nodes per panel; tensor Gram/reference moments'
    normalized_patches=[];train_coefs=[];devrows=[]
    for group,dirs in [('train',a.train),('dev',a.dev)]:
        for folder,r in cohort(dirs):
            path=folder/r['levels'][-1]['file'];assert digest(path)==r['levels'][-1]['sha256'];bb=np.load(path)['coefficients']
            norm2=jnp.sum(jnp.asarray(bb)**2,axis=(1,2))*.25
            cells=LP.tree(x,w,r['parameters'],52,4,'swept')
            gram,tr,mx,my=metric(jnp.asarray(base),jnp.asarray(cells),*rule)
            rhs=TM.rhs(jnp.asarray(base),jnp.asarray(bb),mx,my)
            c,err=TM.fit(gram,rhs,norm2)
            if not result['cases'][group]:
                gram2,_,mx2,my2=metric(jnp.asarray(base),jnp.asarray(cells),*rule2)
                rhs2=TM.rhs(jnp.asarray(base),jnp.asarray(bb),mx2,my2)
                assert float(jnp.max(jnp.abs(gram2-gram)))<1e-10 and float(jnp.max(jnp.abs(rhs2-rhs)))<1e-10
            filename=r['case']+'.npz';np.savez_compressed(out/filename,gram=np.asarray(gram),rhs=np.asarray(rhs),norm2=np.asarray(norm2),
                teacher=np.asarray(c),cells=np.asarray(cells),transforms=np.asarray(tr),parameters=np.asarray(r['parameters']))
            row=dict(case=r['case'],file=filename,sha256=digest(out/filename),reference_file=str(path),reference_sha256=digest(path),
                     direct_error=np.asarray(err),variables=gram.shape[0],coefficient_norm_over_reference=np.asarray(jnp.linalg.norm(c,axis=1)/jnp.sqrt(norm2)))
            result['cases'][group].append(row);save(out/'result.json',result)
            if group=='train':
                local=np.asarray(c[:,a.root:]).reshape(11,52,36)
                normalized_patches.append((local/np.sqrt(np.asarray(norm2))[:,None,None]).reshape(-1,36))
                train_coefs.append(dict(case=r['case'],coefficients=np.asarray(c),norm=np.sqrt(np.asarray(norm2)),cells=cells))
            print(group,r['case'],float(jnp.max(err)),flush=True)
    patches=jnp.asarray(np.concatenate(normalized_patches));_,s,vh=jnp.linalg.svd(patches,full_matrices=False)
    result['local_POD_singular_values']=np.asarray(s);result['local_POD_sha256']=checkpoint(out/'local_POD.pkl',dict(rotation=vh.T,training_coefficients=train_coefs,root=a.root))
    result['direct_controls']={}
    for rank in (4,8,16,24,36):
        transform=expand_matrix(vh[:rank].T,52,a.root);errs=[];saved=[]
        for row in result['cases']['dev']:
            v=np.load(out/row['file']);c,e=reduced_fit(jnp.asarray(v['gram']),jnp.asarray(v['rhs']),jnp.asarray(v['norm2']),transform)
            errs.append(np.asarray(e));saved.append(dict(case=row['case'],codes=np.asarray(c)))
        result['direct_controls'][str(rank)]=dict(projection=summary(np.asarray(errs)),active_variables=a.root+52*rank)
        checkpoint(out/f'local_direct{rank}_fits.pkl',saved);save(out/'result.json',result)
        print('shared_POD',rank,float(np.max(errs)),flush=True)
    result['status']='complete_local_physical_cache_and_POD';save(out/'result.json',result)
if __name__=='__main__':main()
