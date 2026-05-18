#!/usr/bin/env python
"""Train one INR autoencoder on CG-discrete data for ROM use."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_rom_decoders.autoencoder import INRAutoencoder
from tunable_rom_decoders.training_rom import (
    train_rom_ready,
    compute_tokens_ref,
    save_rom_checkpoint,
)
from tunable_rom_decoders.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train = data["U_train"]
    U_val = data["U_val"]

    model = INRAutoencoder(
        decoder_kind=cfg.decoder_kind,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        coord_dim=cfg.spatial_dim,
        hidden_dim=cfg.hidden_dim,
        siren_num_layers=cfg.siren_num_layers,
        omega_0=cfg.omega_0,
        omega=cfg.omega,
        modulator_hidden=cfg.modulator_hidden,
        d_attn=cfg.d_attn,
        num_fourier=cfg.num_fourier,
        xattn_num_layers=cfg.xattn_num_layers,
        fourier_scale=cfg.fourier_scale,
    )

    params, meta = train_rom_ready(
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
        seed=cfg.seed,
        log_every=cfg.log_every,
    )

    tokens_ref = None
    if cfg.decoder_kind == "xattn":
        print("[train_rom] computing tokens_ref = mean(encoder.tokens(u))")
        tokens_ref = compute_tokens_ref(model, params, U_train)
        print(f"[train_rom] tokens_ref shape = {tokens_ref.shape}")

    save_rom_checkpoint(args.out, params, cfg.to_dict(), meta, tokens_ref=tokens_ref)
    print(f"\nSaved ROM checkpoint to {args.out}")
    print(f"Best val rel-L2 (grid): {meta['best_val_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
