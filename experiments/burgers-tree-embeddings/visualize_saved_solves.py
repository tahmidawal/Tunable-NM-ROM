"""Decode retained trajectories only; no integration, refitting or model selection."""
import argparse, zipfile
from common import *
from training_io import read_checkpoint
import data, packed_fields as PF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE']);arc=ns/'archive'
    source=arc/'b4base/output';record=json.loads((source/'result.json').read_text())
    basisfile=arc/'b1expanded/output/pod_basis.npz'
    globalfile=arc/'b1stableglobal/output/stable_global_basis.npz'
    localfile=arc/'b1cache/output/local_POD.pkl'
    assert digest(basisfile)==record['global_basis_sha256']
    assert digest(globalfile)==record['global_control_sha256']
    assert digest(localfile)==record['local_rotation_sha256']
    root=np.load(basisfile)['coefficients'][:128]
    globalbasis=np.load(globalfile)['coefficients'][:960]
    rotation=read_checkpoint(localfile)['rotation'][:,:24]
    axis=np.linspace(0,1,4096);times=np.arange(6)*.05
    output=jax.jit(PF.center);synth=jax.jit(data.synth_grid)
    results=dict(meta=metadata,source_result_sha256=digest(source/'result.json'),
        sources={str(p):digest(p) for p in [basisfile,globalfile,localfile]},
        field_shape=[4096,4096],dtype='float64',times=times,
        selection='dev000 representative pulse; dev013 worst B4 direct-tree evolved error',
        final_test_opened=False,new_pde_solves=0,cases=[])
    for label in ['dev000','dev013']:
        folder=out/label;folder.mkdir();row=next(c for c in record['cases'] if c['case']==label)
        ref=Path(row['reference_coefficients_path']);assert digest(ref)==row['reference_coefficients_sha256']
        values=np.load(ref);assert np.allclose(values['times'][::2],times,atol=1e-14,rtol=0)
        b=values['coefficients'][::2]
        sources={str(ref):digest(ref)};metrics={};maximum=0.;error_maximum=0.
        for j,t in enumerate(times):
            u=np.asarray(synth(jnp.asarray(b[j]),jnp.asarray(axis)))
            assert u.shape==(4096,4096) and u.dtype==np.float64 and np.isfinite(u).all()
            maximum=max(maximum,float(u.max()));np.save(folder/f'reference_t{j}.npy',u)
        for method,basis in [('tree_direct24',root),('global_stable960',globalbasis)]:
            statefile=source/f'{label}_{method}_states.pkl';sources[str(statefile)]=digest(statefile)
            state=read_checkpoint(statefile);assert len(state['coefficients'])==6
            cache={'packed':PF.prepare(basis,state['cells'],state['transforms'],rotation,axis[None],axis[None])}
            errors=[];linf=[];final_change=None
            for j,t in enumerate(times):
                u=np.asarray(output(jnp.asarray(state['coefficients'][j]),cache))
                truth=np.load(folder/f'reference_t{j}.npy');delta=u-truth
                e=float(np.linalg.norm(delta)/np.linalg.norm(truth))
                assert abs(e-row['methods'][method]['physical_error'][j])<1e-7
                assert np.isfinite(u).all();np.save(folder/f'{method}_t{j}.npy',u)
                errors.append(e);linf.append(float(np.max(np.abs(delta))))
                maximum=max(maximum,float(u.max()));error_maximum=max(error_maximum,linf[-1])
                if j==5:
                    original=source/f'{label}_{method}_4096.npz';sources[str(original)]=digest(original)
                    old=np.load(original)['prediction']
                    final_change=float(np.linalg.norm(u-old)/np.linalg.norm(truth));assert final_change<1e-10
            metrics[method]=dict(relative_L2=errors,max_absolute_error=linf,
                final_retained_grid_relative_change=final_change,
                reference_allowance=row['methods'][method]['reference_allowance_max'])
        fig,axes=plt.subplots(5,6,figsize=(19,14),layout='constrained')
        rows=[('reference','Reference',False),('tree_direct24','Our direct tree',False),
              ('global_stable960','Global control',False),('tree_direct24','Tree − reference',True),
              ('global_stable960','Global − reference',True)]
        for i,(name,title,is_error) in enumerate(rows):
            cmap=plt.get_cmap('RdBu_r' if is_error else 'viridis')
            norm=Normalize(-error_maximum if is_error else 0,error_maximum if is_error else maximum)
            for j,t in enumerate(times):
                u=np.load(folder/f'{name}_t{j}.npy')
                if is_error:u=u-np.load(folder/f'reference_t{j}.npy')
                # Exact one-image-pixel per grid node; transpose because arrays index x,y.
                pixels=cmap(norm(u.T[::-1,:]),bytes=True)
                stem=f'{name}_{"signed_error" if is_error else "field"}_t{j}'
                Image.fromarray(pixels).save(folder/f'{stem}_4096.png',compress_level=1)
                ax=axes[i,j]
                im=ax.imshow(u.T,origin='lower',extent=(0,1,0,1),cmap=cmap,norm=norm,
                             interpolation='nearest',rasterized=True)
                ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
                if i==0:ax.set_title(f't = {t:.2f}')
                if j==0:ax.set_ylabel(title+'\ny')
                if i==4:ax.set_xlabel('x')
                if is_error:ax.set_title(f'L2 = {metrics[name]["relative_L2"][j]*100:.4f}%',fontsize=10)
            fig.colorbar(im,ax=axes[i,:].tolist(),shrink=.85,label='u − reference' if is_error else 'u')
        fig.suptitle(f'{label}: saved PDE trajectories on actual 4096 × 4096 grids\n'
            'Overview scaled for display; individual PNGs retain every grid pixel. Shared scales across times.',fontsize=15)
        fig.savefig(out/f'{label}-evolution.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for method,color,labeltext in [('tree_direct24','#2166ac','Our direct tree'),('global_stable960','#d97706','Global control')]:
            m=metrics[method]
            axes[0].plot(times,100*np.asarray(m['relative_L2']),'o-',color=color,label=labeltext)
            axes[1].plot(times,m['max_absolute_error'],'o-',color=color,label=labeltext)
        axes[0].axhline(1,color='#b2182b',ls='--',label='1% physical limit')
        axes[0].set(ylabel='Current-reference relative L2 error (%)',xlabel='Time',ylim=(0,1.05))
        axes[1].set(ylabel='Maximum pointwise absolute error',xlabel='Time')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{label}: errors from full 4096 × 4096 arrays (t = 0 is the initial fit)')
        fig.savefig(out/f'{label}-errors.png',dpi=180);plt.close(fig)
        results['cases'].append(dict(case=label,parameters=row['parameters'],sources=sources,metrics=metrics,
            field_color_limits=[0,maximum],signed_error_color_limits=[-error_maximum,error_maximum]))
        save(out/'result.json',results);print('rendered',label,metrics,flush=True)
    with zipfile.ZipFile(out/'full-resolution-images.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.glob('*/*_4096.png')):z.write(p,p.relative_to(out))
    results['status']='complete_saved_trajectory_visualization';save(out/'result.json',results)


if __name__=='__main__':main()
