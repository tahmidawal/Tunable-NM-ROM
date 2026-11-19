"""Reusable fixed-decoder tables covering every cell through a declared depth.

No reference fields or development-selected geometries enter this cache. Offline
construction/storage/loading must be reported; query selection and gathering are
online work. Scientific GPU qualification is separate from this implementation.
"""
from common import *
import geometry as geo, tensor_metric as TM, discrete_weak as W, packed_fields as PF
import compact_fields as CF


def make_reducer(root):
    @jax.jit
    def reduce(gram, rotation):
        channels, rank = rotation.shape
        nodes = (len(gram)-root)//channels
        cross = jnp.einsum('gpr,ra->gpa',gram[:root,root:].reshape(root,nodes,channels),rotation).reshape(root,-1)
        local = jnp.einsum('ra,prqs,sb->paqb',rotation,
                          gram[root:,root:].reshape(nodes,channels,nodes,channels),rotation).reshape(nodes*rank,nodes*rank)
        return jnp.block([[gram[:root,:root],cross],[cross.T,local]])
    return reduce


def build(basis, rotation, L, Q, tests, depth=4, output_points=4096, rule=None,field_kernel='packed'):
    assert 1 <= depth <= 4 and L % Q == 0 and (Q-1)**2 >= 4*tests
    assert field_kernel in ('packed','compact')
    cells = geo.uniform(depth);geo.valid(cells)
    metric = TM.make_geometry(6)
    physical, transforms, _, _ = metric(jnp.asarray(basis),jnp.asarray(cells),*(TM.rule() if rule is None else rule))
    gram = make_reducer(len(basis))(physical,jnp.asarray(rotation))
    weak, _ = W.prepare(basis,cells,transforms,rotation,L,tests,gram)
    # Drop the compact full-grid axes: only the tested matrices and fixed-shape
    # stencil/output tables are used by this candidate implementation.
    cache = {key:weak[key] for key in ['A','indices','lam','gram']}
    axis = np.linspace(0,1,output_points)[None]
    if field_kernel=='packed':
        cache['packed'] = PF.prepare_stencil(basis,cells,transforms,rotation,L,Q)
        output = PF.prepare(basis,cells,transforms,rotation,axis,axis)
    else:
        cache['compact'] = CF.prepare_stencil(basis,cells,transforms,rotation,L,Q,depth)
        output = CF.prepare(basis,cells,transforms,rotation,axis,axis,depth)
    jax.block_until_ready((cache,output))
    return dict(cells=np.asarray(cells),root=len(basis),rank=np.asarray(rotation).shape[1],
                depth=depth,L=L,Q=Q,tests=tests,output_points=output_points,field_kernel=field_kernel,
                arrays=dict(cache=cache,output=output,transforms=jnp.asarray(transforms)),
                scope='every geometrically admissible cell through the declared depth; no solution fields')


@jax.jit
def gather(arrays, nodes, columns):
    rank=(len(columns)-len(arrays['cache']['packed']['basis']))//len(nodes) if len(nodes) else arrays['cache']['packed']['transform'].shape[-1]
    def fields(p):
        return dict(basis=p['basis'],sine_x=p['sine_x'],sine_y=p['sine_y'],
                    X=p['X'][:,:,nodes],Y=p['Y'][:,:,nodes],transform=p['transform'][nodes,:,:rank])
    ca = arrays['cache']
    result = dict(A=ca['A'][:,columns],indices=ca['indices'],lam=ca['lam'],
                  gram=ca['gram'][columns[:,None],columns[None,:]],packed=fields(ca['packed']))
    return dict(cache=result,output_cache={'packed':fields(arrays['output'])},transforms=arrays['transforms'][nodes])


@jax.jit
def gather_compact(arrays,nodes,columns,selectors):
    rank=(len(columns)-len(arrays['cache']['compact']['basis']))//len(nodes) if len(nodes) else arrays['cache']['compact']['transform'].shape[-1]
    def fields(p):
        result={k:p[k] for k in ['basis','sine_x','sine_y']}
        result['transform']=p['transform'][nodes,:,:rank]
        buckets=[]
        for source,selection in zip(p['buckets'],selectors):
            rows=source['inverse'][nodes[selection['slots']]]
            buckets.append(dict(X=source['X'][:,rows],Y=source['Y'][:,rows],
                ix=source['ix'][rows],iy=source['iy'][rows],
                slot_indices=selection['slots'],slot_mask=selection['mask']))
        result['buckets']=tuple(buckets)
        return result
    ca=arrays['cache']
    result=dict(A=ca['A'][:,columns],indices=ca['indices'],lam=ca['lam'],
        gram=ca['gram'][columns[:,None],columns[None,:]],compact=fields(ca['compact']))
    return dict(cache=result,output_cache={'compact':fields(arrays['output'])},transforms=arrays['transforms'][nodes])


def select(catalogue, cells,rank=None):
    """Host policy/lookup and device gathering are both charged to the query."""
    geo.valid(cells)
    lookup={tuple(v):i for i,v in enumerate(catalogue['cells'])}
    nodes=np.asarray([lookup[tuple(c)] for c in cells],dtype=np.int64)
    root,full_rank=catalogue['root'],catalogue['rank']
    rank=full_rank if rank is None else rank
    assert 0<rank<=full_rank
    columns=np.concatenate([np.arange(root),root+(nodes[:,None]*full_rank+np.arange(rank)).ravel()])
    if catalogue.get('field_kernel','packed')=='packed':
        result=gather(catalogue['arrays'],jnp.asarray(nodes),jnp.asarray(columns))
    else:
        selectors=[]
        if len(cells):
            for d in range(1,catalogue['depth']+1):
                count=min(4**d,len(cells));active=[i for i,c in enumerate(cells) if c[0]==d]
                slots=np.zeros(count,dtype=np.int32);slots[:len(active)]=active
                selectors.append(dict(slots=jnp.asarray(slots),mask=jnp.asarray(np.arange(count)<len(active),dtype=jnp.float64)))
        result=gather_compact(catalogue['arrays'],jnp.asarray(nodes),jnp.asarray(columns),tuple(selectors))
    jax.block_until_ready(result)
    return result
