"""YAML-backed experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    # Grid
    N: int = 64
    spatial_dim: int = 2
    # Encoder
    patch_size: int = 8
    embed_dim: int = 96
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 64
    # Decoder
    rank: int = 256
    hidden_dim: int = 256
    # Training
    num_epochs: int = 80_000
    batch_size: int = 32
    peak_lr: float = 2e-3
    weight_decay: float = 5e-4
    warmup_frac: float = 0.1
    seed: int = 0
    # Data
    n_train: int = 500
    n_val: int = 50
    # ROM
    gn_max_iters: int = 8
    gn_rel_tol: float = 1e-3
    n_eq_samples: int = 16
    min_eq_points: int = 64
    num_rom_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return ExperimentConfig(**cfg_dict)
