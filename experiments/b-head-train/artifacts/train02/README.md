# Full raw head-ablation archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including every numerical array, the immutable staged source, logs and manifests.

From this directory, verify and restore into a fresh destination:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored
tar -xzf collection.tar.gz -C restored
cd restored
sha256sum -c OUTPUTS.sha256
sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads.
The whole-archive checksum is recorded in `archive.json`.

## The twelve trained checkpoints live in this archive

Every checkpoint this cell trained is Git-tracked HERE, inside the chunked archive, not as a
loose file: unpacked they are 377 MB and the archive already carries them compressed.
Restore with the commands above; they land in `restored/output/`. `audit.json` beside this
file re-checks every hash below against the emitted file, and the evaluation attempt records
the same hash for the checkpoint it actually loaded.

| checkpoint | arm | K | R | sha256 |
|---|---|---:|---:|---|
| `ckpt_d128k16rec.pkl` | `d128k16rec` | 16 | 512 | `df27781fe64b800a789a0c57e755cae53dfe13342be14e386487b4109c227bf8` |
| `ckpt_d512k16rec.pkl` | `d512k16rec` | 16 | 512 | `d514775ed064288a5dda6ac9095f4a59f425f1dc478b17cac3faa388c1a0aa00` |
| `ckpt_d2048k16rec.pkl` | `d2048k16rec` | 16 | 512 | `1d4965310b09aaba92582dd85a3072e01928fa94e905228e7ef282af359b859a` |
| `ckpt_d4608k16rec.pkl` | `d4608k16rec` | 16 | 512 | `7e5dec446298cb06b48ddf482e55df62f2e9206807d439fa4717ee71b81c064a` |
| `ckpt_d128k32rec.pkl` | `d128k32rec` | 32 | 512 | `f999eed1eebd564f33679cc3dcebd71913a54634b37fef92229d2dc09aa007e4` |
| `ckpt_d2048k32rec.pkl` | `d2048k32rec` | 32 | 512 | `fe8010bc7679b43160491266c5e010ba982240c1d044e808e6ce419608fa4ae5` |
| `ckpt_d128k16w.pkl` | `d128k16w` | 16 | 512 | `373bfbcaa86bfb0c6e4393b72bcf4448b396dfc34a8ba2c50d88ce9d9ebaa286` |
| `ckpt_d128k16t.pkl` | `d128k16t` | 16 | 512 | `8eb58f1634559b0fb4bcd12a26015f7431ea0ed766b32f3c1374e29701ce08ef` |
| `ckpt_d128k16z.pkl` | `d128k16z` | 16 | 512 | `f81c6e0a17cbd7b4bfb1afe996ec2b3ce00dd4a4c53af60f85cca255baecec08` |
| `ckpt_best_d2048k32w.pkl` | `best_d2048k32w` | 32 | 512 | `81fa0e3d065132c6282a926195700080187b9e0d8c32c4be08581401e6d4f7cb` |
| `ckpt_joint_d2048k32r512.pkl` | `joint_d2048k32r512` | 32 | 512 | `2630caefd9d09504cab183f646d6fca9bba1e6541476a6030448e7d37579bbfc` |
| `ckpt_joint_d2048k32r1024.pkl` | `joint_d2048k32r1024` | 32 | 1024 | `49b7800d9383391ad360f0b3090df75828a8614b47150a1f1a80254f136f66ba` |
