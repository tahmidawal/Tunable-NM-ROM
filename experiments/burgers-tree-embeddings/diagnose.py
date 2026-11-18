"""Bounded representation diagnostics, including deliberately optimistic span controls."""
import argparse
from common import *
import data,decoder as dec,geometry as geo,fit,local_study as LS
from training_io import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--dev',required=True);ap.add_argument('--global-model',required=True);ap.add_argument('--local-model',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    D=load_reference(a.dev,'dev');T=load_reference(a.train,'train')
    mdl=read_checkpoint(Path(a.local_model)/'local_model.pkl');model=mdl['global_model'];pl=mdl['local']
    gf=read_checkpoint(Path(a.global_model)/'global_head16_selected.pkl');z=jnp.asarray(gf['dev_codes'])
    x,w,Y=D['xy'],D['w'],D['Y'];G0=LS.global_matrix(model,x);parent=model['heads']['16']
    result=dict(meta=preflight(),status='running',comparisons=[],optimistic_span_diagnostics=[],fit_repair=[],
                note='64x64 Gauss development screening; no PDE/runtime qualification; final tests sealed')
    # Same parent and local checkpoint, actual active variable budgets.
    for rank in (4,8,16):
        for cap in (32,64,128):
            if 16+4*rank>cap:continue
            rows,saved=LS.evaluate_pair(model,pl,None,D,z,rank,25,3,cap)
            result['comparisons'] += rows;checkpoint(out/f'fits_r{rank}_cap{cap}.pkl',saved)
            save(out/'result.json',result)
            print('direct',rank,cap,[(r['allocation'],r['projection']['worst']) for r in rows],flush=True)
    # A lower bound that releases the parent to ALL128 coefficients and each child
    # to ALL16 coefficients. It is not an eligible128-variable method.
    for kind in ('neural16','polynomial16'):
        errs=[];conditions=[]
        for i,p in enumerate(D['parameters']):
            cells=geo.allocated(x,np.asarray(data.initial(x,p))**2,w,128,4)
            B=dec.local_matrix(pl,x,cells,16) if kind=='neural16' else LS.polynomial_matrix(x,cells,16)
            G=jnp.concatenate([G0,B],axis=1);C,pred,s=fit.linear_fit(G,Y[i*11:(i+1)*11],w)
            errs.append(np.asarray(errors(pred,Y[i*11:(i+1)*11],w)));conditions.append(np.asarray(s))
        result['optimistic_span_diagnostics'].append(dict(kind=kind,projection=summary(np.asarray(errs)),
                   free_linear_coefficients=128+28*16,eligible=False,singular_values=conditions))
        save(out/'result.json',result);print('span',kind,float(np.max(errs)),flush=True)
    # Bounded optimization audit on the worst four prescribed cases, with125 LM steps.
    best=next(r for r in result['comparisons'] if r['rank']==4 and r['cap']==128 and r['allocation']=='prescribed')
    worst=np.argsort(np.max(np.asarray(best['projection']['per_snapshot']),axis=1))[-4:]
    for i in worst:
        cells=best['trees'][i];s=slice(i*11,(i+1)*11)
        f=LS.project_tree(model,pl,None,x,w,Y[s],z[s],cells,4,125,3)
        result['fit_repair'].append(dict(case=D['labels'][i],before=best['projection']['per_snapshot'][i],
                   after=np.asarray(f['error']),stationarity=np.asarray(f['stationarity'])))
        save(out/'result.json',result)
    # Train-code stationarity and optimization gap on a registered small subset.
    training=read_checkpoint(Path(a.local_model)/'direct_training_codes_selected.pkl')['codes4']
    trainroot=jnp.asarray(gf['train_codes']);trpred=dec.head(parent,trainroot)@LS.global_matrix(model,T['xy']).T
    _,_,trees=LS.padded_trees(T,trpred,4,2026111851)
    gap=[]
    for i in (0,1,2,3,16,32,48,63):
        cs=trees[i][2];s=slice(i*11,(i+1)*11);coeff=lambda a:dec.coefficients(parent,None,a,len(cs),4)
        GG=jnp.concatenate([LS.global_matrix(model,T['xy']),dec.local_matrix(pl,T['xy'],cs,4)],axis=1)
        pred=jax.vmap(coeff)(jnp.asarray(training[2,s,:16+4*len(cs)]))@GG.T
        fresh=LS.project_tree(model,pl,None,T['xy'],T['w'],T['Y'][s],trainroot[s],cs,4,125,3)
        gap.append(dict(case=T['labels'][i],training_codes=summary(errors(pred,T['Y'][s],T['w'])),refitted=summary(fresh['error'])))
    result['training_code_gap']=gap
    # Save a self-contained same-snapshot decomposition for plotting (quadrature grid).
    i=int(worst[-1]);t=int(np.argmax(best['projection']['per_snapshot'][i]));s=slice(i*11,(i+1)*11)
    f=LS.project_tree(model,pl,None,x,w,Y[s],z[s],best['trees'][i],4,125,3)
    root=G0@dec.head(parent,f['codes'][t,:16]);frozenroot=G0@dec.head(parent,z[s][t])
    np.savez_compressed(out/'witness.npz',x=np.unique(np.asarray(x)[:,0]),reference=Y[s][t].reshape(64,64),
          root_before=frozenroot.reshape(64,64),joint_root=root.reshape(64,64),
          children=(f['pred'][t]-root).reshape(64,64),reconstruction=f['pred'][t].reshape(64,64),
          time=data.TIMES[t],case=D['labels'][i],parameters=D['parameters'][i],cells=best['trees'][i],
          normalized_error=f['error'][t])
    result.update(status='complete_diagnostic_only',witness_case=D['labels'][i],witness_time=data.TIMES[t])
    save(out/'result.json',result)
if __name__=='__main__':main()
