"""Data adapter for the official CROM offline code (dc3042/CROM_offline_training @ 39333f7): serves our Burgers 2D
snapshot pool in exactly the item format of simulation.SimulationDataset ({'filename','x','q','encoder_input','time'})
and exposes the same DataModule interface (train/test loaders, get_dataFormat, get_dataParams). Only the storage
backend differs: frames come from pool26.fields.npy (held in RAM) instead of one h5 file per frame.

Ordering mirrors util.obtainFilesRecursively: the list of simulations (trajectories) is shuffled with random.seed(0),
frames inside a simulation follow in time order; train_dataloader keeps shuffle=False as released.
Statistics (mean/std of q, mean/std/min/max of x) are the same quantities as computeStandardizeTransformation,
accumulated in float64 chunks instead of one vstack of every frame (which would need ~30 GB more RAM).
"""
import os, random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

L = 256
NTRAIN = 4352            # trajectories P[0:4352] train (same split as lane nfrom-baseline)


def node_coords():
    x = np.arange(L + 1) / L
    return np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2).astype(np.float32)   # row-major = field order


class OursDataset(Dataset):
    def __init__(self, fields, order):
        self.fields, self.order = fields, order          # fields [Ntraj, 26, 66049] float32 in RAM
        self.x = torch.from_numpy(node_coords())

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        tr, lv = self.order[idx]
        q = torch.from_numpy(np.ascontiguousarray(self.fields[tr, lv])).float().view(-1, 1)
        return {'filename': f'sim_seq_{tr:05d}/h5_f_{lv:010d}.h5', 'x': self.x, 'q': q,
                'encoder_input': torch.cat((q, self.x), 1), 'time': torch.tensor(0.01 * lv, dtype=torch.float64)}


class OursDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=16, num_workers=8):
        super().__init__()
        self.data_path, self.batch_size, self.num_workers = data_path, batch_size, num_workers
        mm = np.load(os.path.join(data_path, 'pool26.fields.npy'), mmap_mode='r')
        self.fields = np.ascontiguousarray(mm[:NTRAIN])                       # ~28.7 GB, loaded once before workers fork
        sims = list(range(NTRAIN))
        random.seed(0); random.shuffle(sims)                                 # as util.obtainFilesRecursively
        self.order = [(s, lv) for s in sims for lv in range(self.fields.shape[1])]
        self.sim_dataset = OursDataset(self.fields, self.order)
        self.computeStandardizeTransformation()

    def train_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def computeStandardizeTransformation(self):
        s = s2 = 0.0; n = 0
        for a in range(0, len(self.fields), 64):
            c = self.fields[a:a + 64].astype(np.float64)
            s += c.sum(); s2 += (c * c).sum(); n += c.size
        mean = s / n
        std = np.sqrt(max(s2 / n - mean * mean, 0.0))
        self.mean_q = np.array([mean]); self.std_q = np.array([std if std >= 1e-12 else 1.0])
        x = node_coords().astype(np.float64)
        self.mean_x, self.std_x = x.mean(0), x.std(0)
        self.min_x, self.max_x = x.min(0), x.max(0)

    def get_dataParams(self):
        return {'mean_q': self.mean_q, 'std_q': self.std_q, 'mean_x': self.mean_x, 'std_x': self.std_x, 'min_x': self.min_x, 'max_x': self.max_x}

    def get_dataFormat(self):
        example_input_array = torch.unsqueeze(self.sim_dataset[0]['encoder_input'], 0)
        [_, i_dim] = self.sim_dataset[0]['x'].shape
        [npoints, o_dim] = self.sim_dataset[0]['q'].shape
        return {'i_dim': i_dim, 'o_dim': o_dim, 'npoints': npoints, 'data_path': self.data_path}, example_input_array
