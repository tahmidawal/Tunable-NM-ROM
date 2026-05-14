"""YAML-backed experiment configuration for the Poisson ViT-decoder ablation."""
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
    latent_dim: int = 16
    # Decoder (ViT) — None means "mirror encoder"
    dec_patch_size: Optional[int] = None
    dec_embed_dim: Optional[int] = None
    dec_num_heads: Optional[int] = None
    dec_num_layers: Optional[int] = None
    # Training
    num_epochs: int = 100_000
    batch_size: int = 32
    peak_lr: float = 1e-3
    weight_decay: float = 2e-3
    warmup_frac: float = 0.05
    seed: int = 42
    # Laplacian-aware loss weight; 0 to disable. Needed for ViT decoder
    # to keep the patch-boundary Laplacian content from corrupting the
    # NM-ROM residual.
    lap_weight: float = 0.0
    # Data
    n_train: int = 700
    n_val: int = 140
    data_source: str = "analytical"  # "analytical" or "cg"
    # ROM
    gn_max_iters: int = 11
    gn_rel_tol: float = 5e-2
    n_eq_samples: int = 200
    min_eq_points: int = 2000

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    for k in ("rank", "hidden_dim"):
        cfg_dict.pop(k, None)
    return ExperimentConfig(**cfg_dict)
