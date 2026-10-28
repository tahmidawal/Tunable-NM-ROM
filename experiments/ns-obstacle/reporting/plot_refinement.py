"""Generate provisional refinement diagnostics from retained results, never admission."""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();sources=[]
    def read(path):
        raw=path.read_bytes();sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()});return json.loads(raw)
    benchmark=read(args.input/'benchmark.json');rows=[]
    for level in range(3):
        d=read(args.input/f'results/dfg_h{level}/result.json');m=read(args.input/f'results/manufactured_h{level}/result.json');f=d['metrics']['forces'];s=f['physical_symmetric_order12']['coefficient'];g=f['gradient_order12']['coefficient']
        rows.append(dict(level=level,cells=d['geometry']['element_count'],divergence=d['metrics']['divergence']['value'],drag_error=abs(s[0]-benchmark['drag_coefficient'])/abs(benchmark['drag_coefficient']),lift_error=abs(s[1]-benchmark['lift_coefficient']),pressure_drop_error=abs(f['pressure_probes']['difference']-benchmark['pressure_drop'])/abs(benchmark['pressure_drop']),physical_gradient_lift_gap=abs(s[1]-g[1]),manufactured_velocity=m['metrics']['velocity_error']['value'],manufactured_pressure=m['metrics']['pressure_error']['value']))
    contract=read(Path(__file__).resolve().parents[1]/'campaign.json')['reference_gates']
    panels=[('divergence',contract['divergence_scaled'],'Scaled divergence'),('lift_error',contract['dfg_lift_absolute'],'Physical lift error vs DFG'),('physical_gradient_lift_gap',contract['physical_gradient_transverse_gap_absolute'],'Physical / gradient lift gap'),('pressure_drop_error',contract['dfg_pressure_drop_relative'],'Pressure-drop relative error'),('manufactured_velocity',contract['analytic_velocity_relative'],'Manufactured velocity error'),('manufactured_pressure',contract['analytic_pressure_scaled'],'Manufactured pressure error')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for ax,(key,threshold,title) in zip(axes.flat,panels):
        ax.semilogy([r['level'] for r in rows],[r[key] for r in rows],'-o',color='#1b6c91',label='Saved value')
        ax.axhline(threshold,color='#b84634',ls='--',label='Acceptance threshold')
        ax.set(title=title,xlabel='Mesh level',xticks=[0,1,2],xticklabels=['h0','h1','h2']);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('2D refinement: improvement, but reference not yet qualified',fontsize=15)
    fig.supxlabel('Provisional diagnostics from ladder02. Common-field comparison failed; no ROM accuracy or speed result.',fontsize=10)
    args.out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(args.out/f'refinement.{suffix}',dpi=160)
    plt.close(fig)
    (args.out/'values.json').write_text(json.dumps({'scope':'provisional; no physical reference admission','rows':rows,'thresholds':contract},indent=2)+'\n')
    (args.out/'PROVENANCE.json').write_text(json.dumps({'inputs':sources,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
