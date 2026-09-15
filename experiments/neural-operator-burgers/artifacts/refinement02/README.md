# Full raw Burgers archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including full numerical arrays, immutable staged source, logs and manifests.

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

The whole-archive checksum is recorded in `archive.json`. Referenced calibration
inputs for a successor job are retained in the separately archived calibration job.
