#!/usr/bin/env python
"""Train baseline ModulatedSIREN AE with an L2 penalty on the encoder z.

Re-uses the exploring-decoders config schema (load_config) and only adds
two optional fields: `lambda_z`, `lambda_z_warmup_steps`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.training_rom_zreg import (
    train_zreg, save_zreg_checkpoint,
)
from tunable_rom_speed.utils.config import load_config  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--lambda-z", type=float, default=None,
                   help="Override lambda_z from config.")
    p.add_argument("--lambda-z-warmup-steps", type=int, default=None)
    args = p.parse_args()

    # The config dataclass doesn't know about lambda_z; read those two
    # YAML keys directly, then strip them before calling load_config.
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    lambda_z = (args.lambda_z if args.lambda_z is not None
                else float(raw.pop("lambda_z", 1.0e-3)))
    lambda_z_warmup = (args.lambda_z_warmup_steps if args.lambda_z_warmup_steps is not None
                       else int(raw.pop("lambda_z_warmup_steps", 2000)))
    # Write the cleaned raw to a temp file and load via load_config, OR
    # bypass load_config: build ExperimentConfig directly from raw.
    from tunable_rom_speed.utils.config import ExperimentConfig
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

    params, meta = train_zreg(
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
        seed=cfg.seed,
        log_every=cfg.log_every,
    )

    out_cfg = cfg.to_dict()
    out_cfg["lambda_z"] = lambda_z
    out_cfg["lambda_z_warmup_steps"] = lambda_z_warmup
    save_zreg_checkpoint(args.out, params, out_cfg, meta)
    print(f"\nSaved zreg checkpoint to {args.out}")
    print(f"Best val rel-L2: {meta['best_val_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
