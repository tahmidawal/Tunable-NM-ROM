# Full raw ns2d archive

The ordered chunks preserve the checksum-verified original compressed job archive.

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads.
