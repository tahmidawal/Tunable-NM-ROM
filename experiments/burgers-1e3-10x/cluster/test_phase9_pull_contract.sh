#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pull_phase9_train.sh
source "$HERE/pull_phase9_train.sh"
fixture="$(mktemp -d)"; trap 'rm -rf -- "$fixture"' EXIT
job=12345; commit=1111111111111111111111111111111111111111

make_fixture() {
  local root="$1"
  mkdir -p "$root/out" "$root/logs"
  printf 'manifest fixture\n' > "$root/MANIFEST.sha256"
  printf '{}\n' > "$root/out/phase9_train.json"
  printf 'npz\n' > "$root/out/phase9_train.npz"
  printf 'checkpoint\n' > "$root/out/checkpoint.pkl"
  printf '{}\n' > "$root/out/PROGRESS.json"
  printf 'work\n' > "$root/out/work_checkpoint.pkl"
  printf 'jax_backend=gpu\nALL-DONE\n' > "$root/logs/$job.out"
  : > "$root/logs/$job.err"
  local manifest json npz checkpoint
  manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"
  json="$(sha256sum "$root/out/phase9_train.json"|cut -d' ' -f1)"
  npz="$(sha256sum "$root/out/phase9_train.npz"|cut -d' ' -f1)"
  checkpoint="$(sha256sum "$root/out/checkpoint.pkl"|cut -d' ' -f1)"
  jq -n --arg commit "$commit" --arg job "$job" --arg manifest "$manifest" \
    --arg json "$json" --arg npz "$npz" --arg checkpoint "$checkpoint" \
    '{status:"pass",negative_aware:true,expected_commit:$commit,expected_job:$job,
      manifest_sha256:$manifest,source_json_sha256:$json,source_npz_sha256:$npz,
      checkpoint_sha256:$checkpoint}' > "$root/out/AUDIT.json"
  printf 'JobIDRaw|JobName|State|ExitCode|Elapsed|AllocTRES|NodeList|\n%s|p9_t1_s11_r1|COMPLETED|0:0|00:01:00|gres/gpu=1|pax001|\n' "$job" > "$root/SACCT.txt"
  (cd "$root" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
}

refresh_remote() {
  (cd "$1" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256)
}

good="$fixture/good"; make_fixture "$good"
manifest="$(sha256sum "$good/MANIFEST.sha256"|cut -d' ' -f1)"
verify_pulled_bundle "$good" "$job" "$commit" "$manifest"

if verify_pulled_bundle "$good" "$job" "$commit" "$(printf '0%.0s' {1..64})"; then exit 1; fi
if verify_pulled_bundle "$good" bad "$commit" "$manifest"; then exit 1; fi

for kind in audit_commit audit_manifest audit_source sacct extra; do
  one="$fixture/$kind"; cp -a "$good" "$one"
  case "$kind" in
    audit_commit) jq '.expected_commit="2222222222222222222222222222222222222222"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh_remote "$one" ;;
    audit_manifest) jq '.manifest_sha256="0000000000000000000000000000000000000000000000000000000000000000"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh_remote "$one" ;;
    audit_source) jq '.source_npz_sha256="0000000000000000000000000000000000000000000000000000000000000000"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh_remote "$one" ;;
    sacct) sed -i 's/COMPLETED|0:0/FAILED|1:0/' "$one/SACCT.txt" ;;
    extra) printf 'unexpected\n' > "$one/out/unexpected.bin"; refresh_remote "$one" ;;
  esac
  if verify_pulled_bundle "$one" "$job" "$commit" "$manifest" >/dev/null 2>&1; then
    echo "corruption accepted: $kind" >&2; exit 1
  fi
done
echo phase9_pull_contract=pass
