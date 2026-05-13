#!/usr/bin/env python
"""Train the ViT-CP autoencoder for the Heat equation.

Usage:
    python -m scripts.train --config configs/heat2d_n64.yaml \
        --data data/heat2d_n64.npz --out checkpoints/heat2d_n64.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_heat.models.autoencoder import ViTCPAutoencoder
from tunable_rom_heat.utils.config import load_config
from tunable_rom_heat.utils.training import train_autoencoder, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]

    model = ViTCPAutoencoder(
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        rank=cfg.rank,
        hidden_dim=cfg.hidden_dim,
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
