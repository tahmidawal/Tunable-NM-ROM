"""YAML config loader for the exploring-decoders experiment."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    decoder_kind: str = "siren"            # 'siren' or 'xattn'

    # PDE + data.
    N: int = 128
    spatial_dim: int = 2
    n_train: int = 700
    n_val: int = 140
    seed: int = 42

    # Encoder (shared, pinned to paper).
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 16

    # Decoder — SIREN.
    siren_num_layers: int = 5
    siren_hidden_dim: int = 256
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 128

    # Decoder — XATTN.
    d_attn: int = 64
    num_fourier: int = 16
    xattn_num_layers: int = 3
    xattn_hidden_dim: int = 256
    fourier_scale: float = 4.0

    # Decoder — Affine-z specific.
    affine_bias_hidden: int = 128

    # Training.
    num_epochs: int = 50_000
    batch_size: int = 24
    points_per_sample_train: int = 1024
    points_per_sample_val: int = 4096
    on_grid_frac_train: float = 0.5        # half on-grid, half off-mesh per batch
    peak_lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    log_every: int = 500

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hidden_dim(self) -> int:
        """Decoder hidden width selected by decoder_kind."""
        if self.decoder_kind in ("siren", "affine_z"):
            return self.siren_hidden_dim
        return self.xattn_hidden_dim


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(**raw)
