"""Equivalent fixed-checkpoint output cache; all preparation must be charged."""
from common import *
from neural import packed_residual as P, residual_tree as R


class OutputCache:
    def __init__(self, params, cells, rows=64, size=4096):
        assert size%rows==0
        self.params=params;self.cells=np.asarray(cells);self.rows=rows;self.size=size
        self.axis=np.linspace(0.,1.,size);self.features={};self.kernels={};self.tiles=None
        self.compilation_seconds=0.;self.logical_cache_bytes=0
        self.feature=jax.jit(lambda f,c,x:R.feature_matrix(f,c[None],x)[:,0,:])
        self.apply=jax.jit(P.apply)

    def prepare_tile(self, lo):
        # Inspect host coordinates before their one-way upload, never np.asarray(device_xy).
        host=np.stack(np.meshgrid(self.axis[lo:lo+self.rows],self.axis,indexing='ij'),-1).reshape(-1,2)
        ids_list=[];blocks=[]
        for cell in self.cells:
            h=2.**(-cell[0]);center=(cell[1:]+.5)*h
            ids=np.flatnonzero(np.all(np.abs((host-center)/h)<.75,axis=1))
            x=jnp.asarray(host[ids]);c=jnp.asarray(cell)
            if len(ids):
                if len(ids) not in self.features:
                    begin=time.perf_counter();self.features[len(ids)]=self.feature.lower(self.params['feature'],c,x).compile()
                    self.compilation_seconds+=time.perf_counter()-begin
                block=self.features[len(ids)](self.params['feature'],c,x)
            else:block=jnp.zeros((0,self.params['scale'].shape[0]),dtype=jnp.float64)
            ids_list.append(jnp.asarray(ids,dtype=jnp.int32));blocks.append(block)
        return jnp.asarray(host),dict(indices=tuple(ids_list),blocks=tuple(blocks))

    def build(self):
        assert self.tiles is None
        self.tiles=[self.prepare_tile(lo) for lo in range(0,self.size,self.rows)]
        jax.block_until_ready(self.tiles)
        self.logical_cache_bytes=sum(v.size*v.dtype.itemsize for v in jax.tree.leaves(self.tiles))
        return self.logical_cache_bytes

    def reconstruct(self, code, reuse=False):
        assert not reuse or self.tiles is not None
        chunks=[]
        for k,lo in enumerate(range(0,self.size,self.rows)):
            points,packed=self.tiles[k] if reuse else self.prepare_tile(lo)
            key=tuple(len(v) for v in packed['indices'])
            if key not in self.kernels:
                begin=time.perf_counter();self.kernels[key]=self.apply.lower(self.params,code,points,packed).compile()
                self.compilation_seconds+=time.perf_counter()-begin
            chunks.append(self.kernels[key](self.params,code,points,packed).reshape(self.rows,self.size))
        output=jnp.concatenate(chunks,axis=0);jax.block_until_ready(output)
        return output
