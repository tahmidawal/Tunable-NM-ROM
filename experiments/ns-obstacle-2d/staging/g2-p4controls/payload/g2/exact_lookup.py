"""Certify exact physical coordinates; repair only failed native inverse lookup.

No physical point or acceptance threshold is altered. Fallback searches the native
triangle and two vertex-adjacency rings, solving the actual curved map in reference
coordinates. An unlocated point remains a failure, never a subset/extrapolated value.
"""
import numpy as np

class ExactPointLocator:
    def __init__(self,mesh,ng,D):
        self.mesh,self.ng,self.D=mesh,ng,float(D)
        self.elements={int(e.nr):e for e in mesh.Elements(ng.VOL)}
        self.vertices={k:{int(v.nr) for v in e.vertices} for k,e in self.elements.items()}
        self.by_vertex={}
        for k,vertices in self.vertices.items():
            for v in vertices:self.by_vertex.setdefault(v,set()).add(k)
        self.coordinate=ng.CF((ng.x,ng.y))
        self.eps=64*np.finfo(np.float64).eps
        self.tolerance=1e-10*self.D

    def _certificate(self,number,ref,mapped,xy):
        bary=np.array([ref[0],ref[1],1-ref[0]-ref[1]])
        valid=(np.isfinite(ref).all() and np.isfinite(mapped).all() and abs(ref[2])<=self.eps
               and np.min(bary)>=-self.eps and np.linalg.norm(mapped-xy)<=self.tolerance)
        return {'element_number':number,'reference_coordinates':ref.tolist(),
                'barycentric_coordinates':bary.tolist(),'mapped_coordinates':mapped.tolist(),
                'physical_error':float(np.linalg.norm(mapped-xy)),'valid':bool(valid)}

    def locate(self,xy):
        xy=np.asarray(xy,dtype=np.float64)
        native=self.mesh(*[float(v) for v in xy],tol=1e-12)
        nr=int(native.nr)
        if nr<0 or native.vb!=self.ng.VOL or nr not in self.elements:
            raise ValueError('native locator has no volume triangle; no unbounded fallback')
        if str(self.elements[nr].type)!='ET.TRIG':raise ValueError('triangle-only exact lookup')
        ref=np.asarray(native.pnt,dtype=np.float64)
        mapped=np.array(self.coordinate(native),dtype=np.float64,copy=True)
        original=self._certificate(nr,ref,mapped,xy)
        if original['valid']:return native,None,original,None
        candidates={nr};front={nr}
        for _ in range(2):
            vertices=set().union(*(self.vertices[k] for k in front))
            neighbors=set().union(*(self.by_vertex[v] for v in vertices))
            front=neighbors-candidates;candidates|=neighbors
        record={'requested_point':xy.tolist(),'original_lookup':original,'candidates':[],
                'physical_point_changed':False,'thresholds_changed':False}
        for candidate in [nr]+sorted(candidates-{nr}):
            element=self.elements[candidate]
            if str(element.type)!='ET.TRIG':continue
            tr=self.mesh.GetTrafo(element)
            xi=ref[:2].copy() if candidate==nr else np.array([1/3,1/3],dtype=np.float64)
            last=None
            try:
                for iteration in range(12):
                    ir=self.ng.IntegrationRule([tuple(map(float,xi))],[1.])
                    mip=tr(ir[0]);point=np.array(self.coordinate(mip),dtype=np.float64,copy=True)
                    last=self._certificate(candidate,np.r_[xi,0.],point,xy)
                    if np.linalg.norm(point-xy)<=self.tolerance*.01:break
                    jac=np.array(mip.jacobi,dtype=np.float64,copy=True)
                    if jac.shape!=(2,2) or not np.isfinite(jac).all():raise ValueError('invalid curved-map Jacobian')
                    xi-=np.linalg.solve(jac,point-xy)
                    if not np.isfinite(xi).all() or np.max(np.abs(xi))>4:raise ValueError('bounded inverse iteration left candidate neighborhood')
                last['iterations']=iteration+1;record['candidates'].append(last)
                if last['valid']:
                    record['selected']=last
                    # Keep native transformation and integration rule alive with the mapped point.
                    return mip,(tr,ir),last,record
            except Exception as exc:record['candidates'].append({'element_number':candidate,'error':repr(exc),'last':last})
        error=ValueError('exact-coordinate adjacent-element lookup failed; no point change or extrapolation')
        error.lookup_evidence=record
        raise error
