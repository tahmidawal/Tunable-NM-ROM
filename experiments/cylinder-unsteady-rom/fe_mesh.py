"""Second rebuttal push, PROTOCOL.md A7: coarser meshes of the same channel for the FE comparator ladder.

Same geometry and boundary names as the admitted mesh, netgen 2D, curve order 4.  Sizes (maxh, obstacle maxh):
c1 (0.014, 0.0035), c2 (0.02, 0.005), c3 (0.028, 0.007), c4 (0.04, 0.01); the admitted mesh m0 is (0.01, 0.0025).
Writes <out>/<name>.vol.bin and <out>/MESHES.json (path, sha256, elements, vertices, Taylor-Hood unknowns).
Usage: fe_mesh.py --admitted inputs/admitted_mesh.vol.bin --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

SIZES = {'c1': (0.014, 0.0035), 'c2': (0.02, 0.005), 'c3': (0.028, 0.007), 'c4': (0.04, 0.01)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def info(ng, path):
    mesh = ng.Mesh(str(path))
    rec = {'path': str(path), 'sha256': sha(path), 'elements': int(mesh.ne), 'vertices': int(mesh.nv),
           'curve_order': int(mesh.GetCurveOrder()), 'boundaries': sorted(set(mesh.GetBoundaries())), 'unknowns': {}}
    for k in (2, 3):
        X = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle') * ng.H1(mesh, order=k - 1)
        rec['unknowns'][f'p{k}'] = int(X.ndof)
    rec['area'] = float(ng.Integrate(1, mesh, order=12))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--admitted', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import ngsolve as ng
    from netgen.geom2d import SplineGeometry
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    res = {'m0': dict(info(ng, Path(a.admitted).resolve()), maxh=0.01, obstacle_h=0.0025)}
    for name, (h, ho) in SIZES.items():
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (2.2, 0.41), bcs=('wall_bottom', 'outlet', 'wall_top', 'inlet'))
        geo.AddCircle((0.2, 0.2), 0.05, leftdomain=0, rightdomain=1, bc='obstacle', maxh=ho)
        mesh = ng.Mesh(geo.GenerateMesh(maxh=h))
        mesh.Curve(4)
        p = out / f'{name}.vol.bin'
        mesh.ngmesh.Save(str(p))
        res[name] = dict(info(ng, p), maxh=h, obstacle_h=ho)
        print(name, res[name]['elements'], res[name]['unknowns'], res[name]['curve_order'], flush=True)
    (out / 'MESHES.json').write_text(json.dumps(res, indent=2) + '\n')


if __name__ == '__main__':
    main()
