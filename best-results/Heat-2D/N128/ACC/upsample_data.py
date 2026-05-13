"""
upsample_data.py — build training_data_2d_128.pkl from the N=64 pickle.

Bilinearly upsamples every snapshot (64×64 → 128×128) and saves a pkl
with the same schema that fixed.py (N=128) expects.  Runs in ~2 min on
a login node (no GPU needed).
"""
import pickle
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

SRC = Path('/cluster/tufts/paralab/tawal01/NMROM-Apr8/Heat2D/data/training_data_2d_64.pkl')
DST = Path('/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Autoresearch/Heat-2D/N128/training_data_2d_128.pkl')

def up(arr_flat, n_src=64, n_dst=128):
    """Zoom a flat (N_src^2,) or (n_snap, N_src^2) array to N_dst^2."""
    factor = n_dst / n_src
    if arr_flat.ndim == 1:
        return zoom(arr_flat.reshape(n_src, n_src), factor).flatten().astype(np.float32)
    return np.stack([zoom(row.reshape(n_src, n_src), factor).flatten().astype(np.float32)
                     for row in arr_flat])

print(f'Loading {SRC} ...')
with open(SRC, 'rb') as f:
    src = pickle.load(f)

print('Upsampling U_train ...')
U_train = up(np.asarray(src['U_train'], dtype=np.float32))
print(f'  U_train: {src["U_train"].shape} -> {U_train.shape}')

print('Upsampling U_val ...')
U_val = up(np.asarray(src['U_val'], dtype=np.float32))
print(f'  U_val:   {src["U_val"].shape} -> {U_val.shape}')

print('Upsampling all_snapshots ...')
all_snapshots = [up(np.asarray(t, dtype=np.float32)) for t in src['all_snapshots']]

print('Upsampling val_snapshots ...')
val_snapshots = [up(np.asarray(t, dtype=np.float32)) for t in src['val_snapshots']]

payload = {
    'U_train':        U_train,
    'U_val':          U_val,
    'all_snapshots':  all_snapshots,
    'val_snapshots':  val_snapshots,
    'train_params':   src['train_params'],
    'val_params':     src['val_params'],
    'traj_kappas':    src['traj_kappas'],
    'val_kappas':     src['val_kappas'],
    'traj_starts':    src['traj_starts'],  # integer indices, no upsample needed
    'grid_config':    {'N': 128, 'L': 1.0, 'dx': 1.0/127, 'dt': 5e-3, 'NUM_STEPS': 50},
}

print(f'Saving to {DST} ...')
DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'wb') as f:
    pickle.dump(payload, f, protocol=4)

size_mb = DST.stat().st_size / 1e6
print(f'Done. {DST} ({size_mb:.0f} MB)')
