"""Guarded reference/scoring adapter after approved one-time E0 publication."""
import functools
import io
import json
import os
from pathlib import Path

import numpy as np
from open_sealed import ALLOW, COHORTS, REGISTRY, STORE, sha, verify_approval, verify_file

class Opened:
    def __init__(self,source,config):
        binding=verify_approval(source)
        fixed=json.loads((source/'F1.json').read_bytes())['files']
        config_path=Path(config).resolve().relative_to(source.resolve()).as_posix()
        if config_path not in fixed:
            raise ValueError('Unfrozen study configuration')
        verify_file(source/config_path,fixed[config_path])
        self.registry=Path(os.environ['JCP_OPEN_REGISTRY'])
        if self.registry.resolve()!=REGISTRY:
            raise ValueError('Foreign opening record')
        complete=json.loads((self.registry/'COMPLETE.json').read_bytes())
        self.record=json.loads(verify_file(self.registry/'published.json',complete['published_sha256']))
        intent=json.loads((self.registry/'INTENT.json').read_bytes())
        if (complete.get('file_count')!=len(ALLOW) or intent!=binding or any(complete.get(k)!=v or self.record.get(k)!=v for k,v in binding.items())
                or set(self.record['files'])!=ALLOW):
            raise ValueError('Opening/publication/selection binding mismatch')
        self.status_sha=json.loads((source/'F1.json').read_bytes())['reference_status_sha256']
        self.store=self.registry/'payload'

    def seed(self,dim):
        return json.loads(verify_file(self.store/'inrange-seeds.json',self.record['seed_sha256']))[f'{dim}d/{COHORTS[dim][0]}']

    def bytes(self,rel):
        if rel not in ALLOW:
            raise ValueError('Case outside the in-range population')
        return verify_file(self.store/rel,self.record['files'][rel])

    @functools.lru_cache(maxsize=384)
    def meta(self,dim,cohort,case,kind):
        rel=f'{dim}d/{cohort}/{kind}/c{case:03d}.json'
        m=json.loads(self.bytes(rel))
        if (m['dim'],m['cohort'],m['case'],m['kind'])!=(dim,cohort,case,kind):
            raise ValueError('Reference metadata identity mismatch')
        n0=m['judge']['n0']
        eps=m['judge']['eps_ref']
        if not np.isfinite(n0) or n0<=0 or (m['outcome']=='accepted' and
                (eps is None or not np.isfinite(eps) or eps<0)):
            raise ValueError('Original reference norm or accepted uncertainty is invalid')
        return m

    def load(self,dim,cohort,case,kind='ST',field='finest'):
        m=self.meta(dim,cohort,case,kind)
        if field!='finest':
            raise ValueError('Only the registered finest reference is scored')
        with np.load(io.BytesIO(self.bytes(f'{dim}d/{cohort}/{kind}/c{case:03d}.npz')),allow_pickle=False) as arr:
            return np.asarray(arr[m['final'][2]]),m

    def refstore(self,rs):
        if rs.store_version()!=STORE:
            raise ValueError('Reference version mismatch')
        verify_file(rs._meta_dir()/'STATUS.json',self.status_sha)
        status=rs._status()
        def ref_status(d,c,i,k='ST'):
            meta=self.meta(d,c,i,k)
            ds=status['dimension_status'][str(d)]
            return dict(accepted=meta['outcome']=='accepted',outcome=meta['outcome'],eps_ref=meta['judge']['eps_ref'],
                        dimension_status=ds,crosscheck=None,eligible=meta['outcome']=='accepted' and ds=='validated')
        def score(field,d,c,i,k='ST'):
            f,meta=self.load(d,c,i,k)
            if f.shape!=field.shape:
                raise ValueError('Incorrect scoring lattice')
            err,et=rs.V.rel_err(field,f,meta['judge']['n0'])
            return dict(e=err,e_t=et,**ref_status(d,c,i,k))
        from types import SimpleNamespace
        return SimpleNamespace(load=self.load,reference_status=ref_status,store_version=rs.store_version,score=score)

    def regenerated(self,d,solver):
        # Original E0 uses a row-wise generator in 2D, unlike the historical test64 generator.
        import importlib.util
        refdir=solver.V/'jcp-references'
        spec=importlib.util.spec_from_file_location('e0cohorts',refdir/'cohorts.py')
        ch=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ch)
        co,n=COHORTS[d]
        seed=self.seed(d)
        if d==2:
            rows=ch.descriptors_2d(co,seed)
            digest=sha(np.ascontiguousarray(rows).tobytes())
        else:
            rows=ch.table_3d(co,seed)
            digest=rows['sha256']
        commit_by_batch={}
        max_draw=max_peak=0.
        for i in range(n):
            a=self.meta(d,co,i,'ST')
            b=self.meta(d,co,i,'S')
            if a['desc']!=b['desc'] or a['cohort_sha256']!=b['cohort_sha256']:
                raise ValueError('Original S/ST identities or descriptors disagree')
            batch=i//32
            original=a['cohort_sha256']
            if commit_by_batch.setdefault(batch,original)!=original:
                raise ValueError('Inconsistent original cohort within one archived generation job')
            m=a['desc']
            def relative(value, reference):
                value,reference=np.asarray(value),np.asarray(reference)
                if value.shape!=reference.shape or not np.isfinite(value).all() or not np.isfinite(reference).all():
                    raise ValueError('Incompatible original and regenerated descriptors')
                return float(np.max(np.abs(value-reference)/np.maximum(np.abs(reference),1e-300)))
            if d==2:
                canonical=np.asarray([m[k] for k in ('cx','cy','w','a','nu')])
                if not np.array_equal(rows[i,:4],canonical[:4]):
                    raise ValueError('2D uniform draws mismatch')
                max_draw=max(max_draw,relative(rows[i,4],canonical[4]))
                rows[i]=canonical
            else:
                nb=int(m['B'])
                if rows['B'][i]!=nb:
                    raise ValueError('3D number of blobs mismatch')
                for k in ('nu','A'):
                    max_draw=max(max_draw,relative(rows[k][i],m[k]))
                    rows[k][i]=m[k]
                for k in ('c','w','rho'):
                    max_draw=max(max_draw,relative(rows[k][i,:nb],m[k]))
                    rows[k][i,:nb]=m[k]
                max_peak=max(max_peak,relative(rows['s_star'][i],m['s_star']))
                rows['s_star'][i]=m['s_star']
        if max_draw>1e-14 or max_peak>1e-12:
            raise ValueError('Regenerated descriptors exceed original A5 portability bounds')
        if d==2:
            canonical_hash=sha(np.ascontiguousarray(rows).tobytes())
            if len(set(commit_by_batch.values()))==1 and canonical_hash!=next(iter(commit_by_batch.values())):
                raise ValueError('Fully reconstructed 2D cohort mismatches original committed SHA')
        else:
            import hashlib
            h=hashlib.sha256()
            for k in ('B','c','w','rho','A','nu','s_star'):
                h.update(np.ascontiguousarray(rows[k]).tobytes())
            canonical_hash=rows['sha256']=h.hexdigest()
        self.descriptor_receipt={'original_cohort_sha256_by_batch':commit_by_batch,
                          'regenerated_sha256': digest, 'used_sha256':canonical_hash,
                          'max_relative_draw_difference':max_draw,'max_relative_peak_difference':max_peak}
        print(json.dumps(self.descriptor_receipt), flush=True)
        return rows
