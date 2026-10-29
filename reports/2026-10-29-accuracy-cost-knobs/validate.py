"""Audit the atlas independently of the plotting implementation; no GPU or network."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import math
import statistics
import sys
import importlib.util
import io
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    rows=json.loads((HERE/'POINTS.json').read_text())
    plots=json.loads((HERE/'PLOTS.json').read_text())
    keys=[(r['dimension'],r['panel'],r['id']) for r in rows]
    assert len(keys)==len(set(keys)), 'duplicate configuration keys'
    bykey=dict(zip(keys,rows)); linked=0;repetitions=0
    for r in rows:
        cs=r['cases'];ee=[v['e'] for v in cs.values()]
        assert all(math.isfinite(x) and x>=0 for x in ee)
        assert abs(statistics.median(ee)-r['error_median'])<1e-14
        assert max(ee)==r['error_worst']
        assert sum(e>.01 for e in ee)==r['error_over_1pct']
        assert len(ee)==r['n_cases']
        if r['dimension']==2:assert set(cs)=={f'dev6|{i}' for i in range(6)}
        if r['dimension']==3:assert r['K'] is None
        if r['source']=='historical3d-work':
            assert r['milliseconds'] is None and r['cost_kind']=='iterations'
            assert sorted(map(int,cs))==list(range(64))
        for cid,c in cs.items():
            if c.get('e_t') is not None:assert abs(max(c['e_t'][1:])-c['e'])<1e-12
            tt=c.get('time_repetitions_ms',[])
            if c.get('milliseconds') is not None:
                assert tt and all(math.isfinite(t) and t>0 for t in tt)
                assert abs(statistics.median(tt)-c['milliseconds'])<1e-10
                linked+=1;repetitions+=len(tt)
        mm=[c.get('milliseconds') for c in cs.values()]
        if r['milliseconds'] is not None:
            assert all(t is not None for t in mm)
            assert abs(statistics.median(mm)-r['milliseconds'])<1e-10
            assert r['timing_usable']
    source_npz=0
    for p in (HERE/'data').glob('*timed.json.gz'):
        d=json.loads(gzip.decompress(p.read_bytes()));raw=d['result']
        assert raw['backend']=='gpu' and raw['complete']
        for lb,s in d['subjects'].items():
            source_npz+=1
            if p.name.startswith('jcp-'):assert s['expected_manifest_match']
            assert len(s['npz_sha256'])==64
            for cid,v in s['scores'].items():
                assert v['reference_eligible'] and len(v['reference_sha256'])==64
                assert abs(max(v['e_t'][1:])-v['e'])<1e-12
    for p in plots:
        assert (HERE/'figs'/f"{p['slug']}.png").stat().st_size>5000
        assert (HERE/'figs'/f"{p['slug']}.svg").stat().st_size>5000
        pp=[bykey[tuple(k)] for k in p['point_keys']]
        assert all(r['dimension']==p['dimension'] for r in pp)
        assert p['knob']!='K' or p['dimension']==2
    coverage=json.loads((HERE/'COVERAGE.json').read_text())
    assert next(v for v in coverage if v['dimension']==3 and v['knob']=='K')['matched_groups']==0
    assert all(next(v for v in coverage if v['dimension']==2 and v['knob']==k)['matched_groups']>0
               for k in ['r','K','M','m','dt','mesh'])
    assert all(next(v for v in coverage if v['dimension']==3 and v['knob']==k)['matched_groups']>0
               for k in ['r','M','m','dt','mesh'])
    for name,expected in json.loads((HERE/'MANIFEST.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    timed3d_checks={}
    data3=HERE/'data/timed3d-screen.json.gz'
    if data3.exists():
        doc=json.loads(gzip.decompress(data3.read_bytes()))
        coefs=np.load(HERE/'data/timed3d-coefficients.npz')
        reference=np.load(HERE/'data/timed3d-reference-fields.npz')['fields']
        for bank in ['M2','W1024']:
            products=np.load(HERE/'data'/f'timed3d-products-{bank}.npz')
            gram=products['gram'];cross=products['cross'].reshape(4,6,-1)
            n0=np.array([v['judge']['n0'] for v in doc['reference_metadata']])
            for key,s in doc['subjects'].items():
                if s['bank']!=bank:continue
                a=coefs[key];r=a.shape[-1]
                e2=np.einsum('cti,ij,ctj->ct',a,gram[:r,:r],a)-2*np.sum(a*cross[:,:,:r],axis=-1)+products['ref_norm_squared']
                et=np.sqrt(np.maximum(e2,0))/n0[:,None]
                assert all(np.allclose(et[int(cid)],v['e_t'],atol=1e-12,rtol=1e-10) for cid,v in s['scores'].items())
            if '--raw-spot-check' in sys.argv:
                spec=importlib.util.spec_from_file_location('bank3d_independent',HERE/'data/independent_bank3d.py')
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                path=HERE/'data'/f'timed3d-model-{bank}.npz'
                z=np.load(path if path.exists() else io.BytesIO(b''.join(p.read_bytes() for p in sorted(path.parent.glob(path.name+'.part*')))))
                count=sum(k.startswith('w') for k in z.files)
                p={'freq':z['freq'],'scale':float(z['scale']),'net':[(z[f'w{i}'],z[f'b{i}']) for i in range(count)]}
                control=doc['direct_checks'][bank]['subject'];a=coefs[control][0];r=a.shape[-1]
                axis=np.arange(1,64)/64
                x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
                errors=np.zeros(6)
                for start in range(0,len(x),8192):
                    end=min(start+8192,len(x));g=mod.bank_np(p,z['T'][:,:r],x[start:end],chunk=8192)
                    delta=a@g.T-reference[0,:,start:end];errors+=np.sum(delta*delta,axis=1)
                et=np.sqrt(errors)/n0[0]
                expected=np.array(doc['subjects'][control]['scores']['0']['e_t'])
                deviation=float(max(abs(et-expected)));assert deviation<1e-10
                timed3d_checks[bank]={'direct_lattice_decode_vs_gram_max_difference':deviation,'passed':True}
    raw_checks=0
    if '--raw-spot-check' in sys.argv:
        root=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
        sys.path.insert(0,str(root/'worktrees/2026-10-06-jcp-references/experiments/jcp-references'))
        import refstore
        for name in ['dims-timed','p1-timed']:
            d=json.loads(gzip.decompress((HERE/'data'/f'{name}.json.gz').read_bytes()))
            for lb,s in list(d['subjects'].items())[:2]:
                with np.load(s['npz_path'],allow_pickle=False) as z:
                    score=refstore.score(z['f257'][0],2,'dev6',0,'ST')
                assert abs(score['e']-s['scores']['dev6|0']['e'])<1e-12
                assert np.allclose(score['e_t'],s['scores']['dev6|0']['e_t'][1:],rtol=1e-12,atol=1e-14)
                raw_checks+=1
    result={'status':'PASS','configurations':len(rows),'figures':len(plots),
       'independent_reference_loader_raw_spot_checks':raw_checks,
       'independent_3d_field_decode_checks':timed3d_checks,
       'configuration_counts':dict(Counter(str(r['dimension'])+'D' for r in rows)),
       'timing_linked_case_records':linked,'retained_admitted_timing_repetitions':repetitions,
       'source_scoring_witnesses':source_npz,
       'checks':['unique configuration keys','fixed population and error aggregation',
        'same-output timed score and reference identity retention','timing repetition aggregation',
        'no runtime inferred from first-call or cross-job historical 3D data',
        'no 3D solver K label misrepresented as latent dimension','knob coverage and image existence',
        'all retained artifact checksums'],
       'limits':['Recorded historical qualification audits are reused; numerical solvers are not rerun.',
        'Reference uncertainty is conditional; no sealed cohort or new validation is accessed.',
        '3D nonlinear-head K and complete matched tolerance timing ladders remain absent.']}
    (HERE/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
