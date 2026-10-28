"""Prospective mapped-point blocks and TSQR selection; not production-wired.

Retains the candidate enumeration and singular-value rank criterion of spaces.py.
No squared-condition Gram eigensystem is used for rank decisions.
"""
import numpy as np
from spaces import RANK_CUTOFF, positive_weights


def evaluated_blocks(adapter,coefficients,vector=True,boundary=False,block_points=4096):
    """Evaluate original mapped points while each native transformation remains alive."""
    from fe_adapter import check_allocation, strict_scalar
    ng=adapter.ng; space=adapter.V if vector else adapter.Q
    c=np.asarray(coefficients)
    if c.ndim==1:c=c[:,None]
    if c.dtype!=np.dtype('float64') or c.ndim!=2 or c.shape[0]!=space.ndof or not np.isfinite(c).all():
        raise ValueError('original-f64 compatible coefficient columns required')
    if not 1<=block_points<=16384:raise ValueError('unreviewed block capacity')
    values_shape=(block_points,2,c.shape[1]) if vector else (block_points,c.shape[1])
    gradients_shape=(block_points,2,2,c.shape[1]) if vector else (block_points,2,c.shape[1])
    check_allocation(values_shape);check_allocation(gradients_shape)
    fields=[];derivatives=[]
    for j in range(c.shape[1]):
        field=ng.GridFunction(space);field.vec.FV().NumPy()[:]=c[:,j]
        derivative=ng.Grad(field)
        if boundary:derivative=ng.BoundaryFromVolumeCF(derivative)
        fields.append(field);derivatives.append(derivative)
    values=np.empty(values_shape);gradients=np.empty(gradients_shape);used=0;start=0
    for mip in adapter.iter_mapped(boundary):
        for j,(field,derivative) in enumerate(zip(fields,derivatives)):
            values[...,j][used]=(np.asarray(field(mip)) if vector else strict_scalar(field(mip),'blocked FE value'))
            gradients[...,j][used]=np.asarray(derivative(mip)).reshape((2,2) if vector else (2,))
        used+=1
        if used==block_points:
            if not np.isfinite(values).all() or not np.isfinite(gradients).all():raise ValueError('nonfinite native block')
            yield start,start+used,values,gradients
            # The consumer must use a block before advancing; new buffers prevent
            # accidental saved-block mutation, at the cost of at most one extra block.
            start+=used;used=0;values=np.empty(values_shape);gradients=np.empty(gradients_shape)
    if used:
        if not np.isfinite(values[:used]).all() or not np.isfinite(gradients[:used]).all():raise ValueError('nonfinite native block')
        yield start,start+used,values[:used],gradients[:used]


def select_test_columns_blocks(blocks,target_count,expected_points,with_gradients):
    """Each block is (start,end,values,weights,dimensionless_gradients_or_None)."""
    triangular=None;cursor=0;candidate_count=None;blocks_used=0
    for start,end,values,weights,gradients in blocks:
        if start!=cursor or not start<end<=expected_points:raise ValueError('complete ordered point blocks required')
        values=np.asarray(values);weights=positive_weights(weights,len(values))
        if values.dtype!=np.dtype('float64') or len(values)!=end-start or not np.isfinite(values).all():raise ValueError('invalid candidate block')
        if candidate_count is None:candidate_count=values.shape[-1]
        if values.shape[-1]!=candidate_count:raise ValueError('candidate enumeration changed')
        value_rows=values.reshape(-1,candidate_count)*np.repeat(np.sqrt(weights),int(np.prod(values.shape[1:-1])))[:,None]
        parts=[value_rows]
        if with_gradients:
            gradients=np.asarray(gradients)
            if gradients.dtype!=np.dtype('float64') or len(gradients)!=len(values) or gradients.shape[-1]!=candidate_count or not np.isfinite(gradients).all():raise ValueError('invalid gradient block')
            parts.append(gradients.reshape(-1,candidate_count)*np.repeat(np.sqrt(weights),int(np.prod(gradients.shape[1:-1])))[:,None])
        elif gradients is not None:raise ValueError('unexpected gradients for L2 selection')
        # Orthogonal reduction preserves singular values. Block row permutation
        # preserves the full H1 metric; all columns participate before selection.
        for rows in parts:
            local=np.linalg.qr(rows,mode='r')
            triangular=(local if triangular is None else np.linalg.qr(np.vstack((triangular,local)),mode='r'))
        cursor=end;blocks_used+=1
    if cursor!=expected_points or triangular is None:raise ValueError('incomplete candidate rule')
    selected=[];rejected=[]
    for j in range(candidate_count):
        singular=np.linalg.svd(triangular[:,selected+[j]],compute_uv=False)
        if not len(singular) or singular[0]==0 or singular[-1]<=RANK_CUTOFF*singular[0]:
            rejected.append(j);continue
        selected.append(j)
        if len(selected)==target_count:break
    if len(selected)!=target_count:raise ValueError('insufficient independent candidates')
    _,r=np.linalg.qr(triangular[:,selected],mode='reduced')
    # Make orientation explicit; legacy dense QR may differ by column signs.
    r=np.where(np.diag(r)<0,-1.,1.)[:,None]*r
    transform=np.linalg.inv(r)
    return {'candidate_indices':selected,'rejected_indices':rejected,'target_count':target_count,
      'transform':transform.tolist(),'rank_cutoff':RANK_CUTOFF,
      'metric':'dimensionless H1' if with_gradients else 'dimensionless L2',
      'factorization':'all-point TSQR then identical ordered singular-value decisions',
      'orientation':'positive diagonal selected R','blocks':blocks_used,'points':cursor,
      'physical_admission':False}
