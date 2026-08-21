#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pull_phase11_g2.sh"
root="$(mktemp -d)"; trap 'rm -rf -- "$root"' EXIT
job=123456; commit=0123456789abcdef0123456789abcdef01234567
mkdir -p "$root/out" "$root/logs"
printf 'x\n' > "$root/logs/$job.out"; : > "$root/logs/$job.err"
printf '{}\n' > "$root/out/PROGRESS.json"; printf 'x\n' > "$root/out/checkpoint.pkl"
printf 'x\n' > "$root/out/phase11_g2.json"; printf 'x\n' > "$root/out/phase11_g2.npz"; printf 'x\n' > "$root/out/work_checkpoint.pkl"
printf 'stage\n' > "$root/MANIFEST.sha256"; manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"
cat > "$root/out/AUDIT.json" <<EOF
{"status":"pass","negative_aware":true,"expected_commit":"$commit","expected_job":"$job","manifest_sha256":"$manifest","source_json_sha256":"$(sha256sum "$root/out/phase11_g2.json"|cut -d' ' -f1)","source_npz_sha256":"$(sha256sum "$root/out/phase11_g2.npz"|cut -d' ' -f1)","checkpoint_sha256":"$(sha256sum "$root/out/checkpoint.pkl"|cut -d' ' -f1)","decision":{"retracted_capacity_used":false,"third_architecture_licensed":false}}
EOF
(cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
printf 'JobIDRaw|JobName|State|ExitCode|Elapsed|AllocTRES|NodeList\n%s|p11_g2_s11_r1|COMPLETED|0:0|00:01:00|gres/gpu=1|pax008\n' "$job" > "$root/SACCT.txt"
verify_pulled_bundle "$root" "$job" "$commit" "$manifest"
mv "$root/out/work_checkpoint.pkl" "$root/work_checkpoint.saved"
(cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
verify_pulled_bundle "$root" "$job" "$commit" "$manifest"
mv "$root/work_checkpoint.saved" "$root/out/work_checkpoint.pkl"
(cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
cp "$root/MANIFEST.sha256" "$root/MANIFEST.good"
printf 'wrong\n' > "$root/MANIFEST.sha256"; ! verify_pulled_bundle "$root" "$job" "$commit" "$manifest"; mv "$root/MANIFEST.good" "$root/MANIFEST.sha256"
! verify_pulled_bundle "$root" 999999 "$commit" "$manifest"
printf 'extra\n' > "$root/out/EXTRA"; (cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256); ! verify_pulled_bundle "$root" "$job" "$commit" "$manifest"; rm "$root/out/EXTRA"
(cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
sed -i 's/COMPLETED/FAILED/' "$root/SACCT.txt"; ! verify_pulled_bundle "$root" "$job" "$commit" "$manifest"
sed -i 's/FAILED/COMPLETED/' "$root/SACCT.txt"
jq '.decision.retracted_capacity_used=true' "$root/out/AUDIT.json" > "$root/out/AUDIT.tmp"; mv "$root/out/AUDIT.tmp" "$root/out/AUDIT.json"
(cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256); ! verify_pulled_bundle "$root" "$job" "$commit" "$manifest"
echo phase11_pull_contract=pass corruptions=4 optional_work_checkpoint=pass
