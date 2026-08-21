#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/pull_phase10_d_portability_audit.sh"
fixture="$(mktemp -d)"; trap 'rm -rf -- "$fixture"' EXIT; root="$fixture/good"; job=12345; commit=1111111111111111111111111111111111111111
mkdir -p "$root/out" "$root/logs"; printf fixture > "$root/MANIFEST.sha256"; manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"
jq -n --arg commit "$commit" --arg job "$job" --arg manifest "$manifest" \
  '{status:"pass",audit_only:true,negative_aware:true,expected_audit_commit:$commit,expected_audit_job:$job,manifest_sha256:$manifest,
    source_commit:"25bb3502b851a4eb56c51a25af3fc844e3583219",source_job:"2739690",
    source_json_sha256:"f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523",
    source_npz_sha256:"35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664",
    initial_control_sha256:"03bcbd51fc375b87bb876953b910d8c6a11dcffc893048ed7d9a6f2c46f4290c",
    work_checkpoint_sha256:"5fd879c32f2205d6427b272042c095624d824a8278e1eaf074c5295bcb574d68",
    checks:{accepted_initial_control:true,repeated_full_field:true,audit_r1_failure_binding:true,recovered_normalization_binding:{pass:true},trust_trace:{pass:true,termination_portable:true},negative_self_test:{pass:true}},
    decision:{optimizer_updates:0,architecture_increase_licensed:false,g2_licensed:false,selection_evaluated:false},capacity:{accepted:false,used:false}}' > "$root/out/AUDIT.json"
printf 'jax_backend=gpu\nALL-DONE\n' > "$root/logs/$job.out"; : > "$root/logs/$job.err"
printf 'JobIDRaw|JobName|State|ExitCode|\n%s|p10_d_r2_audit_r2|COMPLETED|0:0|\n' "$job" > "$root/SACCT.txt"
refresh(){ (cd "$1" && find out logs -type f -exec sha256sum {} \; | LC_ALL=C sort > REMOTE.sha256); }; refresh "$root"
verify_phase10_portability_audit_bundle "$root" "$job" "$commit" "$manifest"
for kind in extra sacct manifest source trust license initial work; do one="$fixture/$kind"; cp -a "$root" "$one"; expected="$manifest"
  case "$kind" in
    extra) printf bad > "$one/out/extra"; refresh "$one";;
    sacct) sed -i 's/COMPLETED|0:0/FAILED|1:0/' "$one/SACCT.txt";;
    manifest) expected="$(printf '0%.0s' {1..64})";;
    source) jq '.source_job="bad"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    trust) jq '.checks.trust_trace.termination_portable=false' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    license) jq '.decision.g2_licensed=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    initial) jq '.initial_control_sha256="bad"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    work) jq '.work_checkpoint_sha256="bad"' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
  esac
  if verify_phase10_portability_audit_bundle "$one" "$job" "$commit" "$expected" >/dev/null 2>&1; then echo "corruption accepted: $kind" >&2; exit 1; fi
done
echo phase10_portability_pull_contract=pass corruptions=8
