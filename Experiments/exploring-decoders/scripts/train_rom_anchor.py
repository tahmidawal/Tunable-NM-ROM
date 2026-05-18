#!/usr/bin/env python
"""Train baseline ModulatedSIREN with strong L2(z) + anchor-at-zero loss."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from tunable_rom_decoders.autoencoder import INRAutoencoder
from tunable_rom_decoders.training_rom_anchor import (
    train_anchor, save_anchor_checkpoint,
)
from tunable_rom_decoders.utils.config import ExperimentConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    lambda_z = float(raw.pop("lambda_z", 1.0e-1))
    lambda_z_warmup = int(raw.pop("lambda_z_warmup_steps", 5000))
    lambda_anchor = float(raw.pop("lambda_anchor", 1.0))
    lambda_anchor_warmup = int(raw.pop("lambda_anchor_warmup_steps", 5000))

    cfg = ExperimentConfig(**raw)
    data = np.load(args.data)
    U_train = data["U_train"]
    U_val = data["U_val"]

    model = INRAutoencoder(
        decoder_kind=cfg.decoder_kind,
        N=cfg.N, spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads, num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        coord_dim=cfg.spatial_dim,
        hidden_dim=cfg.hidden_dim,
        siren_num_layers=cfg.siren_num_layers,
        omega_0=cfg.omega_0, omega=cfg.omega,
        modulator_hidden=cfg.modulator_hidden,
        d_attn=cfg.d_attn, num_fourier=cfg.num_fourier,
        xattn_num_layers=cfg.xattn_num_layers,
        fourier_scale=cfg.fourier_scale,
    )

    params, meta = train_anchor(
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
        lambda_z=lambda_z,
        lambda_z_warmup_steps=lambda_z_warmup,
        lambda_anchor=lambda_anchor,
        lambda_anchor_warmup_steps=lambda_anchor_warmup,
        seed=cfg.seed,
        log_every=cfg.log_every,
    )

    out_cfg = cfg.to_dict()
    out_cfg["lambda_z"] = lambda_z
    out_cfg["lambda_z_warmup_steps"] = lambda_z_warmup
    out_cfg["lambda_anchor"] = lambda_anchor
    out_cfg["lambda_anchor_warmup_steps"] = lambda_anchor_warmup
    save_anchor_checkpoint(args.out, params, out_cfg, meta)
    print(f"\nSaved anchor checkpoint to {args.out}")
    print(f"Best val rel-L2: {meta['best_val_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
