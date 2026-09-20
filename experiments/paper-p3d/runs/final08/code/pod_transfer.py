"""Known-boundary prolongation of a frozen native POD basis; no fine truth."""
import numpy as np


def prolong(basis,native,n):
    assert n%native==0 and basis.shape[0]==(native-1)**3
    position=np.arange(1,n)*native/n;left=np.floor(position).astype(int);weight=position-left
    matrix=np.zeros((n-1,native-1))
    for row,(index,w) in enumerate(zip(left,weight)):
        if 1<=index<native:matrix[row,index-1]+=1-w
        if 1<=index+1<native:matrix[row,index]+=w
    native_basis=np.asarray(basis).reshape(native-1,native-1,native-1,-1)
    return np.einsum('ia,jb,kc,abcr->ijkr',matrix,matrix,matrix,native_basis,optimize='greedy').reshape((n-1)**3,-1)


def verify():
    from scipy.interpolate import RegularGridInterpolator
    native=8;n=16;rng=np.random.default_rng(920801);basis=rng.normal(size=((native-1)**3,5))
    values=prolong(basis,native,n)
    points=np.stack(np.meshgrid(*([np.arange(1,n)/n]*3),indexing='ij'),axis=-1).reshape(-1,3)
    expected=RegularGridInterpolator(tuple([np.arange(native+1)/native]*3),
        np.pad(basis.reshape(native-1,native-1,native-1,5),((1,1),(1,1),(1,1),(0,0))))(points)
    defect=float(np.max(abs(values-expected)));identity=float(np.max(abs(prolong(basis,native,native)-basis)))
    assert defect<1e-12 and identity==0
    return dict(passed=True,independent_interpolation_maximum_defect=defect,native_identity_defect=identity,
        source='training-grid basis and known zero boundary only; no finer training solutions')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
