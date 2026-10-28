"""Source-only output factory for the independently owned 2D FOM query.

Returns the same dimensionless output dictionaries as Model2D.query. Geometry/rule
verification is offline; field/force evaluation and output allocation are online.
Native equivalence and performance qualification are required before timing claims.
"""
from pathlib import Path
import numpy as np
from oriented_normals import fluid_outward_normal
from scalars import scalar_value as scalar


def make_output_factory(ng,expected_volume,expected_boundary,normal_evidence_path):
    """Expected rules must come from the exact admitted immutable mesh/export."""
    def factory(mesh,V,Q,X,cfg):
        if mesh.dim!=2 or cfg['kind']!='dfg' or cfg['outlet']!='gradient':
            raise ValueError('fixed 2D DFG query geometry required')
        D,U=cfg['D'],cfg['U'];order=cfg['diagnostic_order']
        normal,certificate=fluid_outward_normal(mesh,cfg['center'],order=order,
                                                evidence_path=Path(normal_evidence_path))
        if certificate.get('passed') is not True:raise ValueError('normal certificate failed')
        def mapped(boundary=False):
            for element in mesh.Elements(ng.BND if boundary else ng.VOL):
                transform=mesh.GetTrafo(element)
                for ip in ng.IntegrationRule(element.type,order):
                    # Native mapped point consumed before transformation leaves scope.
                    yield element,ip,transform(ip)
        points=[];weights=[]
        for element,ip,mip in mapped():
            points.append(np.array(mip.point,copy=True));weights.append(scalar(ip.weight,"reference weight")*scalar(mip.measure,"mapped measure"))
        points=np.asarray(points);weights=np.asarray(weights)
        for name,actual in [('points',points),('weights',weights)]:
            expected=np.asarray(expected_volume[name])
            if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):
                raise ValueError('query full rule differs from admitted '+name)
        labels=[];boundary_points=[];boundary_weights=[];normals=[]
        for element,ip,mip in mapped(True):
            labels.append(element.mat);boundary_points.append(np.array(mip.point,copy=True))
            boundary_weights.append(scalar(ip.weight,"boundary weight")*scalar(mip.measure,"boundary measure"));normals.append(np.array(normal(mip),copy=True))
        labels=np.asarray(labels);boundary_points=np.asarray(boundary_points)
        boundary_weights=np.asarray(boundary_weights);normals=np.asarray(normals)
        if labels.tolist()!=[r['label'] for r in expected_boundary]:raise ValueError('boundary labels differ')
        for actual,key in [(boundary_points,'point'),(boundary_weights,'weight'),(normals,'normal')]:
            expected=np.asarray([r[key] for r in expected_boundary])
            if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):
                raise ValueError('query boundary rule differs from admitted '+key)
        poles=np.array([[.15,.2],[.25,.2]],dtype=np.float64)
        for point in poles:
            if mesh(*point,tol=1e-12).nr<0:raise ValueError('exact pressure pole outside FE geometry; no nudge')
        def fields(u,p,nu):
            velocity=np.empty((len(points),2));pressure=np.empty(len(points))
            for index,(_,_,mip) in enumerate(mapped()):
                velocity[index]=np.asarray(u(mip))/U;pressure[index]=scalar(p(mip),'FOM output pressure')/U**2
            return {'velocity':velocity,'pressure':pressure}
        def probes(u,p,nu):
            velocity=np.empty((len(poles),2));pressure=np.empty(len(poles))
            for index,point in enumerate(poles):
                mip=mesh(*point,tol=1e-12)
                if mip.nr<0:raise ValueError('exact pressure pole outside FE geometry; no nudge')
                velocity[index]=np.asarray(u(mip))/U;pressure[index]=scalar(p(mip),'FOM pole pressure')/U**2
            return {'velocity':velocity,'pressure':pressure}
        def forces(u,p,nu):
            if not np.isfinite(nu) or nu<=0:raise ValueError('actual positive viscosity required')
            derivative=ng.BoundaryFromVolumeCF(ng.Grad(u));force=np.zeros(2)
            for index,(element,ip,mip) in enumerate(mapped(True)):
                if labels[index]!='obstacle':continue
                gradient=np.asarray(derivative(mip)).reshape(2,2)
                stress=nu*(gradient+gradient.T)-scalar(p(mip),'FOM force pressure')*np.eye(2)
                force+=boundary_weights[index]*(stress@(-normals[index]))
            return {'force_coefficient':2*force/(U**2*D)}
        return {'full_velocity_pressure_fields':fields,'fixed_probes':probes,'integrated_forces':forces}
    return factory
