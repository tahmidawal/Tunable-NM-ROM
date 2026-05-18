#!/usr/bin/env python
"""Train the variational + default-field SIREN autoencoder."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

from tunable_rom_decoders.autoencoder_vdf import INRAutoencoderVDF
from tunable_rom_decoders.training_rom_vdf import train_vdf, save_vdf_checkpoint


@dataclass
class VDFConfig:
    # PDE / data.
    N: int = 128
    spatial_dim: int = 2
    seed: int = 42

    # Encoder.
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_enc_layers: int = 4
    latent_dim: int = 16

    # SIREN body.
    siren_num_layers: int = 5
    siren_hidden_dim: int = 384
    omega_0: float = 30.0
    omega: float = 1.0
    modulator_hidden: int = 192

    # Default-field trunk.
    default_hidden_dim: int = 128
    default_num_layers: int = 4
    default_omega_0: float = 15.0

    # Training.
    num_epochs: int = 50000
    batch_size: int = 24
    points_per_sample_train: int = 1024
    points_per_sample_val: int = 4096
    on_grid_frac_train: float = 0.5
    peak_lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500

    # Variational.
    kl_beta: float = 1.0e-3
    kl_warmup_steps: int = 5000

    log_every: int = 500

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decoder_kind"] = "siren_vdf"
        return d


def load_vdf_config(path: Path) -> VDFConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.pop("decoder_kind", None)
    return VDFConfig(**raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    cfg = load_vdf_config(args.config)
    data = np.load(args.data)
    U_train = data["U_train"]
    U_val = data["U_val"]

    model = INRAutoencoderVDF(
        N=cfg.N, spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads, num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        coord_dim=cfg.spatial_dim,
        hidden_dim=cfg.siren_hidden_dim,
        siren_num_layers=cfg.siren_num_layers,
        omega_0=cfg.omega_0,
        omega=cfg.omega,
        modulator_hidden=cfg.modulator_hidden,
        default_hidden_dim=cfg.default_hidden_dim,
        default_num_layers=cfg.default_num_layers,
        default_omega_0=cfg.default_omega_0,
    )

    params, meta = train_vdf(
        model, U_train, U_val,
        N=cfg.N, spatial_dim=cfg.spatial_dim,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        points_per_sample_train=cfg.points_per_sample_train,
        points_per_sample_val=cfg.points_per_sample_val,
        on_grid_frac_train=cfg.on_grid_frac_train,
        peak_lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        kl_beta=cfg.kl_beta,
        kl_warmup_steps=cfg.kl_warmup_steps,
        seed=cfg.seed,
        log_every=cfg.log_every,
    )

    save_vdf_checkpoint(args.out, params, cfg.to_dict(), meta)
    print(f"\nSaved VDF checkpoint to {args.out}")
    print(f"Best val rel-L2: {meta['best_val_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
