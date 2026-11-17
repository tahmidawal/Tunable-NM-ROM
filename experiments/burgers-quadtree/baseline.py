from campaign import *
import argparse
import qcore as Q
from qstudy import Mesh
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
rep=dict(meta=meta(),contract='historical reproduction: 257x257 six fields, prepared input; not primary timing',rows=[])
mdl=Q.Model();mesh=Mesh(mdl,256,['fast','acc','head']);phs=Q.cohort('dev6');np.save(o/'parameters.npy',phs)
Z=mdl.Zold[::max(1,len(mdl.Zold)//8192)];H=np.asarray(jax.vmap(lambda z:Q.sc.head(mdl.params,z))(jnp.asarray(Z)))
for s in ('head','fast','acc'):
 st=Q.SETTINGS[s];r=st['Rp'];op=mesh.ops[st['M']];rule='q0scaled' if s=='head' else 'lat64';ij,w=Q.mesh_rule(rule,256)
 data=dict(mesh.base(s),**Q.mesh_data(mdl,r,ij,w,256,op['kx'],op['ky']))
 if s=='head':
  pf=mdl.head_params(r);cold=Q.build_cold_head(mdl,pf,Z);tab=Q.head_tables(pf,mdl.K,data,cold)
  fq,_=Q.make_head_query('mesh',pf,mdl.K,256,.005,mdl.trust_head());call=lambda u,nu:fq(u,nu,data,cold,tab)
 else:
  trust,_=mdl.trust_linear(r);cold=Q.build_cold_linear(mdl,r,H@mdl.Lrot[:r].T)
  fq,_=Q.make_linear_query('mesh',r,256,.005,trust);call=lambda u,nu:fq(u,nu,data,cold)
 for i,ph in enumerate(phs):
  u=jnp.asarray(E.initial(256,ph));v,coldsec=timer(lambda:call(u,ph[4]));times=[]
  for k in range(3):
   E.burn(.4);v,sec=timer(lambda:call(u,ph[4]));times.append(sec)
   np.savez_compressed(o/f'{s}_{i}_rep{k}.npz',**v)
  ref=R.make_solver(2,256,adv='upwind')(u[1:-1,1:-1],ph[4],.005,'be',1e-10,1e-10)
  rf=np.pad(np.asarray(ref['out']).reshape(6,255,255),((0,0),(1,1),(1,1)))
  np.savez_compressed(o/f'{s}_{i}_discrete_reference.npz',**ref)
  err=np.linalg.norm((np.asarray(v['fields'])-rf).reshape(6,-1),axis=1)/np.linalg.norm(u)
  rep['rows'].append(dict(setting=s,case=i,cold_seconds=coldsec,warm_seconds=times,discrete_errors=err,reasons=np.asarray(v['reason']),stationarity=np.asarray(v['gn'])))
  save(o/'result.json',rep);print(s,i,err.max(),flush=True)
rep['complete']=True;save(o/'result.json',rep)
