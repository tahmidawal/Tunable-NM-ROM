# Full raw b-seeds archive

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

The extraction npz is deliberately NOT in these chunks; see
`EXCLUDED-FROM-ARCHIVE.txt` and `EXCLUDED-SHA256.txt`. `result.json` here is the
SEED LADDER result; the other result files live inside the archive under `output/`.
