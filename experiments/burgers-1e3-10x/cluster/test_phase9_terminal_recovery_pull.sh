#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pull_phase9_terminal_recovery.sh
source "$HERE/pull_phase9_terminal_recovery.sh"
fixture="$(mktemp -d)"; trap 'rm -rf -- "$fixture"' EXIT
job=12345; commit=1111111111111111111111111111111111111111; root="$fixture/good"
mkdir -p "$root/out" "$root/logs"; printf 'fixture\n' > "$root/MANIFEST.sha256"
printf '{}\n' > "$root/out/phase9_terminal_recovery.json"; printf npz > "$root/out/phase9_terminal_recovery.npz"
printf checkpoint > "$root/out/terminal_checkpoint.pkl"; printf '{}\n' > "$root/out/PROGRESS.json"
printf 'jax_backend=gpu\nALL-DONE\n' > "$root/logs/$job.out"; : > "$root/logs/$job.err"
manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"; json="$(sha256sum "$root/out/phase9_terminal_recovery.json"|cut -d' ' -f1)"; npz="$(sha256sum "$root/out/phase9_terminal_recovery.npz"|cut -d' ' -f1)"; checkpoint="$(sha256sum "$root/out/terminal_checkpoint.pkl"|cut -d' ' -f1)"
jq -n --arg commit "$commit" --arg job "$job" --arg manifest "$manifest" --arg json "$json" --arg npz "$npz" --arg checkpoint "$checkpoint" \
  '{status:"pass",negative_aware:true,expected_commit:$commit,expected_job:$job,manifest_sha256:$manifest,
    source_json_sha256:$json,source_npz_sha256:$npz,checkpoint_sha256:$checkpoint,
    decision:{optimizer_updates:0,complete_original_phase9_result:false,selection_evaluated:false,capacity_license_complete:false,g2_licensed:false}}' > "$root/out/AUDIT.json"
printf 'JobIDRaw|JobName|State|ExitCode|Elapsed|AllocTRES|NodeList|\n%s|p9_terminal_recovery_s11_r1|COMPLETED|0:0|00:01:00|gres/gpu=1|pax001|\n' "$job" > "$root/SACCT.txt"
refresh() { (cd "$1" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256); }
refresh "$root"; verify_terminal_recovery_bundle "$root" "$job" "$commit" "$manifest"
for kind in manifest job extra sacct audit; do
  one="$fixture/$kind"; cp -a "$root" "$one"
  case "$kind" in
    manifest) expected="$(printf '0%.0s' {1..64})" ;;
    job) if verify_terminal_recovery_bundle "$one" bad "$commit" "$manifest" >/dev/null 2>&1; then exit 1; fi; continue ;;
    extra) printf bad > "$one/out/extra"; refresh "$one" ;;
    sacct) sed -i 's/COMPLETED|0:0/FAILED|1:0/' "$one/SACCT.txt" ;;
    audit) jq '.decision.g2_licensed=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one" ;;
  esac
  expected="${expected:-$manifest}"
  if verify_terminal_recovery_bundle "$one" "$job" "$commit" "$expected" >/dev/null 2>&1; then echo "corruption accepted: $kind" >&2; exit 1; fi
  unset expected
done
echo phase9_terminal_recovery_pull_contract=pass
