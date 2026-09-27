"""Post-timing field serialization; original device outputs are preserved exactly."""
import hashlib
from pathlib import Path
import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def save_reference(out, label, truth, grid, speed):
    path = out/'fields'/f'{label}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reference is written once per case/mesh, never once per method/config.
    np.savez(path, truth_u=np.asarray(truth[0]), truth_v=np.asarray(truth[1]),
             intervals=grid.n, boundary=grid.bx, speed=float(speed))
    return str(path.relative_to(out)), file_sha(path)


def save_fields(out, label, u, v, truth, grid, speed, full, shared_reference=False):
    ut, vt = truth
    path = out/'fields'/f'{label}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if full and shared_reference:
        np.savez(path, u=np.asarray(u), v=np.asarray(v), intervals=grid.n, boundary=grid.bx, speed=float(speed))
        return str(path.relative_to(out)), 'full_grid_with_shared_truth'
    if full:
        np.savez_compressed(path, u=np.asarray(u), v=np.asarray(v), truth_u=np.asarray(ut), truth_v=np.asarray(vt),
                            intervals=grid.n, boundary=grid.bx, speed=float(speed))
        return str(path.relative_to(out)), 'self_contained_full_grid'
    inds = np.unique(np.linspace(0, grid.shape[0]-1, min(65, grid.shape[0]), dtype=int))
    take = lambda f: np.asarray(f)[:, inds[:, None], inds]
    np.savez_compressed(path, u=take(u), v=take(v), truth_u=take(ut), truth_v=take(vt),
                        active_axis_indices=inds, intervals=grid.n, boundary=grid.bx, speed=float(speed))
    return str(path.relative_to(out)), 'observation_grid_only_full_metrics_computed_before_subsampling'
