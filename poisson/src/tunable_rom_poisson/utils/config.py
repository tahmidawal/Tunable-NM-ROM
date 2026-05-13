"""YAML-backed experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    # Grid
    N: int = 64
    spatial_dim: int = 2
    # Encoder
    patch_size: int = 8
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 8
    # Decoder
    rank: int = 128
    hidden_dim: int = 256
    # Training
    num_epochs: int = 100_000
    batch_size: int = 32
    peak_lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_frac: float = 0.05
    seed: int = 42
    # Data
    n_train: int = 700
    n_val: int = 140
    data_source: str = "analytical"  # "analytical" or "cg"
    # ROM
    gn_max_iters: int = 8
    gn_rel_tol: float = 1e-3
    n_eq_samples: int = 200
    min_eq_points: int = 2000

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return ExperimentConfig(**cfg_dict)
