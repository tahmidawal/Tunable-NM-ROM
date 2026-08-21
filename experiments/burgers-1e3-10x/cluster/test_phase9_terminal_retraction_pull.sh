#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pull_phase9_terminal_retraction_audit.sh
source "$HERE/pull_phase9_terminal_retraction_audit.sh"
fixture="$(mktemp -d)"; trap 'rm -rf -- "$fixture"' EXIT
job=12345; commit=1111111111111111111111111111111111111111; root="$fixture/good"
mkdir -p "$root/out" "$root/logs"; printf 'fixture\n' > "$root/MANIFEST.sha256"
printf work > "$root/out/AUDIT-WORK.npz"; work="$(sha256sum "$root/out/AUDIT-WORK.npz"|cut -d' ' -f1)"
jq -n --arg commit "$commit" --arg job "$job" --arg manifest "$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)" --arg work "$work" \
  '{status:"pass",negative_aware:true,expected_commit:$commit,expected_job:$job,manifest_sha256:$manifest,
    source_json_sha256:"8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d",
    source_npz_sha256:"d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e",
    checkpoint_sha256:"90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9",
    prior_failed_audit_sha256:"a1c6c0ee23e16da4aab1e4320aedcab4bd7039ec650429908feb15a471f99cec",
    audit_work_npz_sha256:$work,decision:{optimizer_updates:0,terminal_full_field_accepted:true,
    capacity_retracted:true,capacity_reproducible:false,capacity_accepted:false,
    capacity_license_complete:false,g2_licensed:false,complete_original_phase9_result:false,selection_evaluated:false}}' > "$root/out/AUDIT.json"
printf 'jax_backend=gpu\nALL-DONE\n' > "$root/logs/$job.out"; : > "$root/logs/$job.err"
printf 'JobIDRaw|JobName|State|ExitCode|Elapsed|AllocTRES|NodeList|\n%s|p9_terminal_recovery_audit_r1|COMPLETED|0:0|00:01:00|gres/gpu=1|pax001|\n' "$job" > "$root/SACCT.txt"
refresh() { (cd "$1" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256); }
refresh "$root"; manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"; verify_terminal_retraction_audit_bundle "$root" "$job" "$commit" "$manifest"
for kind in manifest extra sacct work acceptance license; do
  one="$fixture/$kind"; cp -a "$root" "$one"; expected="$manifest"
  case "$kind" in
    manifest) expected="$(printf '0%.0s' {1..64})" ;;
    extra) printf bad > "$one/out/extra"; refresh "$one" ;;
    sacct) sed -i 's/COMPLETED|0:0/FAILED|1:0/' "$one/SACCT.txt" ;;
    work) printf corrupt >> "$one/out/AUDIT-WORK.npz"; refresh "$one" ;;
    acceptance) jq '.decision.capacity_accepted=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one" ;;
    license) jq '.decision.g2_licensed=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one" ;;
  esac
  if verify_terminal_retraction_audit_bundle "$one" "$job" "$commit" "$expected" >/dev/null 2>&1; then echo "corruption accepted: $kind" >&2; exit 1; fi
done
echo phase9_terminal_retraction_pull_contract=pass corruptions=6
