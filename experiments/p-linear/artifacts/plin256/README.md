# Full raw p-linear archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including every numerical array (all output fields), the immutable staged source, logs
and manifests.

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads;
`audit.json` is the independent NumPy audit of it. The whole-archive checksum is in
`archive.json`.
