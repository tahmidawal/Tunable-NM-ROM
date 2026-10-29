"""Prospective exact all-point moment fit with disk-backed blocks.

This is not selected by the production pilot until its capacity amendment is reviewed.
The full moment objective, scaling, support policy and tolerances match quadrature.py.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from spaces import positive_weights
from quadrature import output_moments
from dataclasses import replace


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_blocks(rule, coefficients, points, directory, block_points=16384,
                 max_block_bytes=2**28, max_storage_bytes=4*2**30):
    """Persist every original point in order; enforce caps before allocation."""
    directory = Path(directory)
    if directory.exists(): raise FileExistsError('fresh moment directory required')
    coefficients = np.asarray(coefficients); points = np.asarray(points)
    n, d = rule.lift.shape
    rows = 1+d+len(coefficients)*(d+d*d+rule.tests.shape[-1])
    if not 1 <= block_points <= n: raise ValueError('invalid block point count')
    if rows*min(block_points,n)*8 > max_block_bytes: raise MemoryError('moment block cap')
    if rows*n*8 > max_storage_bytes: raise MemoryError('total moment storage cap')
    if points.shape != (n,d): raise ValueError('full original point set required')
    started = time.perf_counter(); directory.mkdir(parents=True, exist_ok=False)
    records = []
    point_fields = ('weights','lift','lift_gradient','pressure_lift_per_nu','velocity',
        'velocity_gradient','pressure','tests','test_gradient','scalar_tests')
    for start in range(0,n,block_points):
        end = min(n,start+block_points)
        local = replace(rule, **{name:getattr(rule,name)[start:end] for name in point_fields})
        matrix = output_moments(local, coefficients, points[start:end], max_bytes=max_block_bytes)
        if matrix.dtype != np.dtype('float64') or not np.isfinite(matrix).all():
            raise ValueError('finite original-f64 moments required')
        path = directory/f'block-{start:09d}.npy'; np.save(path,matrix,allow_pickle=False)
        records.append({'file':path.name,'start':start,'end':end,'sha256':_hash(path),
                        'shape':list(matrix.shape),'bytes':path.stat().st_size})
        del matrix, local
    manifest = {'schema':'ns-obstacle-output-moment-blocks-v1','rows':rows,'points':n,
        'block_points':block_points,'blocks':records,'all_points_in_original_order':True,
        'no_subsampling':True,'matrix_bytes':rows*n*8,'max_block_bytes':max_block_bytes,
        'max_storage_bytes':max_storage_bytes,'generation_seconds':time.perf_counter()-started,
        'source_sha256':_hash(Path(__file__)),
        'moment_source_sha256':_hash(Path(__file__).with_name('quadrature.py')),
        'physical_admission':False}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return MomentBlocks(directory)


class MomentBlocks:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.rows=self.manifest['rows']; self.points=self.manifest['points']
        expected=0
        for record in self.manifest['blocks']:
            if record['start']!=expected or not record['start']<record['end']<=self.points:
                raise ValueError('missing, repeated or reordered block points')
            if _hash(self.directory/record['file'])!=record['sha256']:
                raise ValueError('moment block hash mismatch')
            a=np.load(self.directory/record['file'],mmap_mode='r',allow_pickle=False)
            if a.dtype!=np.dtype('float64') or a.shape!=(self.rows,record['end']-record['start']):
                raise ValueError('moment block schema mismatch')
            expected=record['end']; del a
        if expected!=self.points: raise ValueError('incomplete point set')

    def iter_blocks(self):
        for record in self.manifest['blocks']:
            a=np.load(self.directory/record['file'],mmap_mode='r',allow_pickle=False)
            yield record['start'],record['end'],a
            del a

    def column(self,index,scale):
        for start,end,a in self.iter_blocks():
            if start<=index<end: return np.array(a[:,index-start]/scale,copy=True)
        raise IndexError(index)


def fit_positive_blocks(blocks,full_weights,max_support,relative_tolerance=1e-8,kkt_tolerance=1e-8):
    """Exact full dictionary, bounded resident block; no sampled correlation search."""
    from scipy.optimize import nnls
    started=time.perf_counter(); w=positive_weights(full_weights,blocks.points)
    if not 1<=max_support<=blocks.points: raise ValueError('invalid support cap')
    scale=np.zeros(blocks.rows)
    for start,end,a in blocks.iter_blocks():
        if not np.isfinite(a).all(): raise ValueError('nonfinite moments')
        scale+=np.sum(np.abs(a)*w[start:end],axis=1)
    scale=np.maximum(scale,1e-12); target=np.zeros(blocks.rows); norm=np.empty(blocks.points)
    for start,end,a in blocks.iter_blocks():
        matrix=a/scale[:,None]
        target+=matrix@w[start:end]; norm[start:end]=np.linalg.norm(matrix,axis=0)
        del matrix
    target_norm=max(float(np.linalg.norm(target)),1e-12)
    selected=[]; columns=[]; weights=np.empty(0); history=[]; residual=target.copy();reason='support_cap'
    for count in range(max_support):
        iteration_started=time.perf_counter()
        score=np.empty(blocks.points)
        for start,end,a in blocks.iter_blocks():
            matrix=a/scale[:,None]; corr=matrix.T@residual
            score[start:end]=np.divide(corr,norm[start:end],out=np.full_like(corr,-np.inf),where=norm[start:end]>0)
            del matrix
        score[selected]=-np.inf; choice=int(np.argmax(score))
        if not np.isfinite(score[choice]) or score[choice]<=0:
            reason='no_positive_correlation';break
        selected.append(choice);columns.append(blocks.column(choice,scale));active=np.column_stack(columns)
        try: weights,_=nnls(active,target,maxiter=max(100,10*len(selected)))
        except RuntimeError:
            reason='nnls_iteration_limit';break
        residual=target-active@weights; relative=float(np.linalg.norm(residual)/target_norm)
        history.append({'support':len(selected),'relative_moment_residual':relative,
                        'iteration_seconds':time.perf_counter()-iteration_started})
        if relative<=relative_tolerance:
            reason='fit_tolerance';break
    common={'selected':selected,'history':history,'max_support':max_support,
        'relative_tolerance':relative_tolerance,'kkt_tolerance':kkt_tolerance,
        'row_scaling':scale.tolist(),'training_scope':'decoder output/gradient moments only',
        'moment_manifest_sha256':(blocks.manifest_sha256 if hasattr(blocks,'manifest_sha256') else _hash(blocks.directory/'manifest.json')),
        'moment_manifest':blocks.manifest,
        'normalized_target':target.copy(),
        'selected_normalized_columns':np.column_stack(columns) if columns else np.empty((blocks.rows,0)),
        'all_points_searched':blocks.points,'elapsed_seconds':time.perf_counter()-started,
        'summation_policy':'increasing contiguous point blocks; floating-point sums may differ from dense order'}
    if len(weights)!=len(selected):
        return dict(common,passed=False,reason=reason,relative_moment_residual=None,kkt=None)
    active=np.column_stack(columns) if columns else np.empty((blocks.rows,0))
    gradient=-active.T@residual; positive=weights>0
    stationarity=float(np.max(np.abs(gradient[positive]))) if np.any(positive) else float('inf')
    dual=float(max(0.,-np.min(gradient[~positive]))) if np.any(~positive) else 0.
    complement=float(np.max(np.abs(weights*gradient))) if len(weights) else float('inf')
    normalizer=max(float(np.linalg.norm(active,2))*target_norm,1e-12) if len(selected) else 1.
    kkt=max(stationarity,dual,complement)/normalizer
    relative=float(np.linalg.norm(residual)/target_norm)
    return dict(common,passed=bool(reason=='fit_tolerance' and kkt<=kkt_tolerance),reason=reason,
        selected_weights=weights.tolist(),support=np.asarray(selected)[positive].tolist(),
        weights=weights[positive].tolist(),relative_moment_residual=relative,kkt=kkt,
        kkt_scope='restricted selected-support NNLS; no full-dictionary optimality claim')


class MemoryMomentBlocks:
    """Derived all-point workspace, hash-retained without redundant disk matrices."""
    def __init__(self,rule,coefficients,points,max_block_bytes=2**28,
                 max_matrix_bytes=8*2**30,max_process_bytes=16*2**30):
        import resource
        self.points=len(rule.weights);self.arrays=[];records=[];cursor=0;rows=None
        started=time.perf_counter()
        for start,end,local in rule.iter_rules():
            if start!=cursor or not start<end<=self.points:raise ValueError('complete ordered moment points required')
            if rows is None:
                d=local.lift.shape[1];rows=1+d+len(coefficients)*(d+d*d+local.tests.shape[-1])
                matrix_bytes=rows*self.points*8
                # Account for original blocks, scaled/absolute/block-building work,
                # full correlation/norm vectors and the largest selected NNLS matrix.
                workspace=4*max_block_bytes+4*self.points*8+4*rows*2560*8
                current=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                if matrix_bytes>max_matrix_bytes or current+matrix_bytes+workspace>max_process_bytes:
                    raise MemoryError('full derived moment workspace exceeds reviewed capacity')
                self.capacity={'matrix_bytes':matrix_bytes,'workspace_envelope_bytes':workspace,
                    'preconstruction_peak_rss_bytes':current,'process_cap_bytes':max_process_bytes,
                    'matrix_cap_bytes':max_matrix_bytes,'block_cap_bytes':max_block_bytes}
            matrix=output_moments(local,coefficients,points[start:end],max_bytes=max_block_bytes)
            if matrix.shape!=(rows,end-start) or not np.isfinite(matrix).all():raise ValueError('invalid full moment block')
            matrix.setflags(write=False);self.arrays.append((start,end,matrix))
            records.append({'start':start,'end':end,'shape':list(matrix.shape),'dtype':str(matrix.dtype),
                            'sha256':hashlib.sha256(matrix.tobytes(order='C')).hexdigest()})
            cursor=end
        if cursor!=self.points or rows is None:raise ValueError('incomplete moment matrix')
        self.rows=rows
        self.manifest={'schema':'ns-obstacle-derived-memory-moments-v1','points':self.points,'rows':rows,
            'blocks':records,'all_original_points':True,'storage':'derived RAM workspace; no unique source discarded',
            'coefficient_shape':list(np.asarray(coefficients).shape),
            'coefficient_sha256':hashlib.sha256(np.asarray(coefficients).tobytes()).hexdigest(),
            'source_sha256':_hash(Path(__file__)),
            'moment_source_sha256':_hash(Path(__file__).with_name('quadrature.py')),
            'numpy_version':np.__version__,'capacity':self.capacity,
            'generation_seconds':time.perf_counter()-started,'physical_admission':False}
        self.manifest_sha256=hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()
    def iter_blocks(self):
        yield from self.arrays
    def column(self,index,scale):
        for start,end,a in self.arrays:
            if start<=index<end:return np.array(a[:,index-start]/scale,copy=True)
        raise IndexError(index)
