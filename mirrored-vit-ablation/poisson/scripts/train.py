#!/usr/bin/env python
"""Train the symmetric ViT autoencoder for the Poisson equation (ablation)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_vit_poisson.models.autoencoder import ViTViTAutoencoder
from tunable_rom_vit_poisson.utils.config import load_config
from tunable_rom_vit_poisson.utils.training import train_autoencoder, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]

    model = ViTViTAutoencoder(
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        dec_patch_size=cfg.dec_patch_size,
        dec_embed_dim=cfg.dec_embed_dim,
        dec_num_heads=cfg.dec_num_heads,
        dec_num_layers=cfg.dec_num_layers,
    )
    print(f"Training AE: {U_train.shape[0]} train, {U_val.shape[0]} val")
    params, meta = train_autoencoder(
        model,
        U_train,
        U_val,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        peak_lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        warmup_frac=cfg.warmup_frac,
        seed=cfg.seed,
    )
    save_checkpoint(args.out, params, cfg.to_dict(), meta)
    print(f"Saved checkpoint to {args.out} (best val: {meta['best_val']:.4e})")


if __name__ == "__main__":
    main()
