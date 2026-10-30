"""Frozen tanh bank: NumPy values/analytic first derivatives and portable NPZ.

Output values have dimensionless velocity units. Gradients are with respect to physical
coordinates; multiply by D before dimensionless weak assembly. No pressure is implicit.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from geometry import Geometry


@dataclass
class FrozenBank:
    geometry: Geometry
    layers: list[tuple[np.ndarray, np.ndarray]]
    transform: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w,b in self.layers]
        self.transform = np.asarray(self.transform, dtype=np.float64)
        previous = self.geometry.dimension
        for w,b in self.layers:
            if w.ndim != 2 or b.shape != (w.shape[1],) or w.shape[0] != previous:
                raise ValueError("inconsistent network layers")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError("nonfinite network")
            previous = w.shape[1]
        if previous % self.geometry.dimension:
            raise ValueError("output must have dimension*rank channels")
        if self.transform.shape[0] != previous // self.geometry.dimension:
            raise ValueError("ordering shape mismatch")
        if not np.isfinite(self.transform).all():
            raise ValueError("nonfinite ordering")

    @property
    def rank(self):
        return self.transform.shape[1]

    def evaluate(self, points, prefix=None):
        """values[n,component,rank], gradients[n,component,coordinate,rank]."""
        x = np.asarray(points, dtype=np.float64)
        offset, scale = self.geometry.coordinate_scaling()
        h = (x-offset)/scale
        dh = np.broadcast_to(np.diag(1/scale), (len(x), self.geometry.dimension, self.geometry.dimension)).copy()
        for index,(w,bias) in enumerate(self.layers):
            h = h@w + bias
            dh = np.einsum("nhd,hk->nkd", dh, w)
            if index < len(self.layers)-1:
                h = np.tanh(h)
                dh *= (1-h*h)[:, :, None]
        raw_rank = h.shape[1]//self.geometry.dimension
        f = h.reshape(len(x),self.geometry.dimension,raw_rank)
        df = dh.reshape(len(x),self.geometry.dimension,raw_rank,self.geometry.dimension).transpose(0,1,3,2)
        b,db = self.geometry.envelope(x)
        values = b[:,None,None]*f
        gradients = b[:,None,None,None]*df + db[:,None,:,None]*f[:,:,None,:]
        rank = self.rank if prefix is None else int(prefix)
        if not 1 <= rank <= self.rank:
            raise ValueError("invalid frozen prefix")
        tr = self.transform[:,:rank]
        return values@tr, gradients@tr

    def values_blocked(self, points, block_points=4096):
        """All original point values, without retaining full-grid derivative temporaries."""
        x=np.asarray(points,dtype=np.float64)
        if not 1<=block_points<=16384:raise ValueError("unreviewed bank evaluation block size")
        shape=(len(x),self.geometry.dimension,self.rank)
        if np.prod(shape,dtype=np.int64)*8>2**30:raise MemoryError("bank values exceed single-array cap")
        values=np.empty(shape,dtype=np.float64)
        for start in range(0,len(x),block_points):
            end=min(start+block_points,len(x))
            values[start:end]=self.evaluate(x[start:end])[0]
        return values

    def save(self, path):
        meta = dict(self.metadata, schema="ns-obstacle-frozen-bank-v1", geometry=self.geometry.as_dict(),
                    layers=len(self.layers), value_units="u/U", gradient_coordinates="physical",
                    derivative_axes="point,component,coordinate,column", activation="tanh",
                    divergence_free=False, ordering="frozen-full-bank-before-prefix")
        arrays = {"transform":self.transform,"metadata_json":np.array(json.dumps(meta,sort_keys=True))}
        for i,(w,b) in enumerate(self.layers):
            arrays[f"W{i}"]=w
            arrays[f"b{i}"]=b
        np.savez_compressed(path,**arrays)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data["metadata_json"]))
            if meta.get("schema") != "ns-obstacle-frozen-bank-v1" or meta.get("activation") != "tanh":
                raise ValueError("unsupported bank format")
            geom=dict(meta["geometry"])
            geom["center"]=tuple(geom["center"])
            layers=[(data[f"W{i}"],data[f"b{i}"]) for i in range(meta["layers"])]
            return cls(Geometry(**geom),layers,data["transform"],meta)


def initialize(geometry,rank=8,width=64,depth=2,seed=27102027):
    rng=np.random.default_rng(seed)
    sizes=[geometry.dimension]+[width]*depth+[geometry.dimension*rank]
    layers=[(rng.normal(size=(a,b))/np.sqrt(a),np.zeros(b,dtype=np.float64)) for a,b in zip(sizes[:-1],sizes[1:])]
    return FrozenBank(geometry,layers,np.eye(rank),{"seed":seed,"trained":False,"width":width,"depth":depth})
