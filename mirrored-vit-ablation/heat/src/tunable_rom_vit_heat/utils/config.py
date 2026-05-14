"""YAML-backed experiment configuration for the Heat ViT-decoder ablation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    # Grid
    N: int = 64
    spatial_dim: int = 3
    # Encoder
    patch_size: int = 8
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 40
    # Decoder (ViT) — None means "mirror encoder"
    dec_patch_size: Optional[int] = None
    dec_embed_dim: Optional[int] = None
    dec_num_heads: Optional[int] = None
    dec_num_layers: Optional[int] = None
    # Training
    num_epochs: int = 100_000
    batch_size: int = 32
    peak_lr: float = 2e-3
    weight_decay: float = 5e-4
    warmup_frac: float = 0.1
    seed: int = 0
    # Data
    n_train: int = 500
    n_val: int = 50
    # ROM
    gn_max_iters: int = 2
    gn_rel_tol: float = 1e-2
    n_eq_samples: int = 32
    min_eq_points: int = 128
    num_rom_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    # Drop CP-specific fields if present in older YAMLs.
    for k in ("rank", "hidden_dim"):
        cfg_dict.pop(k, None)
    return ExperimentConfig(**cfg_dict)
