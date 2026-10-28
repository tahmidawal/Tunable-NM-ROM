"""Pure-array geometry bookkeeping; no native FE import or execution."""
from __future__ import annotations
import numpy as np

# Pinned ngsolve v6.2.2608 fem/elementtopology.cpp, not inferred from a mesh.
TET_VERTICES = np.array(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(0.,0.,0.)))
TET_FACES = ((3,1,2),(3,2,0),(3,0,1),(0,2,1))
TET_NORMALS = np.array(((-1.,0.,0.),(0.,-1.,0.),(0.,0.,-1.),(1.,1.,1.)))
LABELS = ('inlet','outlet','wall','sphere')
BINS = ('all','negative','zero','positive','near_zero')
TERMS = ('measure','normal_x','normal_y','normal_z','position_dot_normal')
VOLUME_DTYPE = np.dtype([('element','i4'),('point_id','i2'),('reference','f8',(3,)),
    ('point','f8',(3,)),('jacobian','f8',(3,3)),('reference_weight','f8'),
    ('determinant','f8'),('determinant_from_J','f8'),('scaled_determinant','f8'),
    ('physical_weight','f8'),('reference_weight_sign','i1'),('determinant_sign','i1')])
FACET_DTYPE = np.dtype(VOLUME_DTYPE.descr + [('face','i4'),('local_face','i1'),
    ('label','i1'),('triangle_reference','f8',(2,)),('measure','f8'),
    ('normal','f8',(3,)),('cofactor_normal','f8',(3,)),('native_normal_error','f8')])
BOUNDARY_DTYPE = np.dtype([('element','i4'),('point_id','i2'),('face','i4'),
    ('label','i1'),('reference','f8',(2,)),('point','f8',(3,)),
    ('jacobian','f8',(3,2)),('normal','f8',(3,)),('reference_weight','f8'),
    ('measure','f8'),('physical_weight','f8'),('reference_weight_sign','i1')])


def determinant(j):
    return (j[...,0,0]*(j[...,1,1]*j[...,2,2]-j[...,1,2]*j[...,2,1])
            -j[...,0,1]*(j[...,1,0]*j[...,2,2]-j[...,1,2]*j[...,2,0])
            +j[...,0,2]*(j[...,1,0]*j[...,2,1]-j[...,1,1]*j[...,2,0]))


def scaled_determinant(j, det):
    denominator = np.prod(np.linalg.norm(j,axis=-2),axis=-1)
    result = np.zeros_like(det)
    np.divide(det,denominator,out=result,where=denominator>0)
    return result, denominator


def masks(det, denominator, epsilon):
    return np.stack((np.ones_like(det,dtype=bool),det<0,det==0,det>0,
                     np.abs(det)<=epsilon*denominator))


def cofactor_normal(j, local_face, det):
    cof = np.stack((np.cross(j[..., :,1],j[..., :,2]),
                    np.cross(j[..., :,2],j[..., :,0]),
                    np.cross(j[..., :,0],j[..., :,1])),axis=-1)
    conormal = cof @ TET_NORMALS[local_face]
    measure = np.linalg.norm(conormal,axis=-1)
    normal = np.full_like(conormal,np.nan)
    np.divide(conormal*np.where(det>0,1.,-1.)[...,None],measure[...,None],
              out=normal,where=measure[...,None]>0)
    return conormal, measure, normal


def facet_reference(local_face, global_vertices, triangle_points, sorted_vertices):
    indices = list(TET_FACES[local_face])
    if sorted_vertices:
        indices.sort(key=lambda i:global_vertices[i])
    bary = np.zeros((len(triangle_points),4))
    bary[:,indices[0]] = triangle_points[:,0]
    bary[:,indices[1]] = triangle_points[:,1]
    bary[:,indices[2]] = 1.-triangle_points[:,0]-triangle_points[:,1]
    return bary@TET_VERTICES, bary


def boundary_reference(bary, volume_vertices, boundary_vertices):
    lookup = {v:i for i,v in enumerate(volume_vertices)}
    # Reference triangle vertices are (1,0), (0,1), (0,0).
    return bary[:,[lookup[boundary_vertices[0]],lookup[boundary_vertices[1]]]]


def moments(points, normals, physical_weights, det, denominator, epsilon):
    values = np.column_stack((np.ones(len(points)),normals,np.sum(points*normals,axis=1)))
    return np.einsum('bn,nt,n->bt',masks(det,denominator,epsilon),values,physical_weights)


def projection(ne, nb, old_ne, old_nb, volume_rule=343, face_rule=49, sample_volume=14, sample_face=6):
    """Stored point records, not repeated coefficient evaluation visits."""
    old_observations = old_ne*sample_volume + old_nb*sample_face
    new_observations = ne*sample_volume + nb*sample_face
    parts = {'A_text_parity':old_observations,'B_baseline':new_observations,
        'B_volume':ne*volume_rule,'B_default_facets':4*ne*face_rule,
        'B_sorted_facets_including_common_interior_points':4*ne*face_rule,
        'B_matched_ordinary_boundary_points':2*nb*face_rule,
        'C_binary_parity':new_observations}
    maximum_record_bytes = max(t.itemsize for t in (VOLUME_DTYPE,FACET_DTYPE,BOUNDARY_DTYPE))
    return {'stored_point_records':parts,'combined_stored_point_records':sum(parts.values()),
        'actual_largest_dtype_bytes':maximum_record_bytes,
        'conservative_record_bytes':512,
        'raw_record_bytes_upper_bound':sum(parts.values())*512,
        'native_facet_form_evaluation_visits':len(BINS)*len(TERMS)*(4*ne+nb)*face_rule,
        'native_volume_integral_evaluation_visits':2*ne*volume_rule,
        'ordinary_integral_evaluation_visits':len(TERMS)*nb*(6+49+81),
        'qualification':'native forms repeatedly evaluate the same locations; those visits are separate from the combined stored-point cap'}
