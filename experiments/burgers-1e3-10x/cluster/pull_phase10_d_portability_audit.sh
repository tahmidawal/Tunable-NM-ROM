#!/usr/bin/env bash
set -euo pipefail

verify_phase10_portability_audit_bundle() {
  local root="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ -d "$root/out" && -d "$root/logs" && "$job" =~ ^[0-9]+$ && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || return 20
  [[ "$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)" == "$expected_manifest" ]] || return 21
  (cd "$root" && sha256sum -c REMOTE.sha256) || return 22
  local expected actual; expected="$(mktemp)"; actual="$(mktemp)"
  printf '%s\n' "logs/$job.err" "logs/$job.out" out/AUDIT.json | LC_ALL=C sort > "$expected"
  (cd "$root" && find out logs -type f -print | LC_ALL=C sort) > "$actual"
  cmp -s "$expected" "$actual" || { rm -f "$expected" "$actual"; return 23; }
  awk '{print $2}' "$root/REMOTE.sha256" | LC_ALL=C sort > "$actual.remote"
  cmp -s "$expected" "$actual.remote" || { rm -f "$expected" "$actual" "$actual.remote"; return 24; }
  rm -f "$expected" "$actual" "$actual.remote"
  awk -F'|' -v job="$job" '$1==job && $2=="p10_d_r2_audit_r1" && $3=="COMPLETED" && $4=="0:0" {ok=1} END {exit !ok}' "$root/SACCT.txt" || return 25
  jq -e --arg commit "$commit" --arg job "$job" --arg manifest "$expected_manifest" \
    '.status=="pass" and .audit_only==true and .negative_aware==true
     and .expected_audit_commit==$commit and .expected_audit_job==$job and .manifest_sha256==$manifest
     and .source_commit=="25bb3502b851a4eb56c51a25af3fc844e3583219" and .source_job=="2739690"
     and .checks.accepted_initial_control==true and .checks.repeated_full_field==true
     and .checks.trust_trace.pass==true and .checks.trust_trace.termination_portable==true
     and .checks.negative_self_test.pass==true and .decision.optimizer_updates==0
     and .decision.architecture_increase_licensed==false and .decision.g2_licensed==false
     and .decision.selection_evaluated==false and .capacity.accepted==false and .capacity.used==false' "$root/out/AUDIT.json" >/dev/null || return 26
  [[ "$(jq -r .source_json_sha256 "$root/out/AUDIT.json")" == f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523 ]] || return 27
  [[ "$(jq -r .source_npz_sha256 "$root/out/AUDIT.json")" == 35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664 ]] || return 28
  [[ "$(jq -r .initial_control_sha256 "$root/out/AUDIT.json")" == 03bcbd51fc375b87bb876953b910d8c6a11dcffc893048ed7d9a6f2c46f4290c ]] || return 29
  [[ "$(jq -r .work_checkpoint_sha256 "$root/out/AUDIT.json")" == 5fd879c32f2205d6427b272042c095624d824a8278e1eaf074c5295bcb574d68 ]] || return 30
}

verify_original_phase10_r2_remote() {
  local remote="$1"
  ssh tufts-login "cd '$remote' \
    && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = 60cd3e3f2d4946c8671f3cb43c3a852fd72a12b86735f3314bf7e78379edb45d \
    && ./verify_manifest_file_set.sh . MANIFEST.sha256 ./logs/2739690.out ./logs/2739690.err ./out/PROGRESS.json ./out/initial_control.npz ./out/phase10_d.json ./out/phase10_d.npz ./out/work_checkpoint.pkl ./out/AUDIT.json ./REMOTE.sha256 ./SACCT.txt \
    && sha256sum -c MANIFEST.sha256 && sha256sum -c REMOTE.sha256 \
    && test \"\$(sha256sum out/phase10_d.json|cut -d' ' -f1)\" = f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523 \
    && test \"\$(sha256sum out/phase10_d.npz|cut -d' ' -f1)\" = 35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664 \
    && awk -F'|' '\$1==2739690 && \$2==\"p10_d_r2\" && \$3==\"FAILED\" && \$4==\"1:0\" {ok=1} END {exit !ok}' SACCT.txt"
}

main() {
  [[ $# -eq 4 ]] || exit 2
  local cell="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ "$cell" == p10_d_r2_audit_r1 ]] || exit 2
  local here exp remote source_remote local_dir transfer state
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; exp="$(dirname "$here")"
  remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
  source_remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p10_d_r2"
  local_dir="$exp/runs/$cell"; [[ ! -e "$local_dir" ]] || exit 3
  state="$(ssh tufts-login "sacct -j '$job' -X -n -P -o JobIDRaw,State,ExitCode" | awk -F'|' -v job="$job" '$1==job {print $2"|"$3;exit}')"; [[ "$state" == COMPLETED\|0:0 ]] || exit 4
  ssh tufts-login "cd '$remote' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$expected_manifest' \
    && sacct -j '$job' -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList -P > SACCT.txt \
    && find out logs -type f -exec sha256sum {} \; | LC_ALL=C sort > REMOTE.sha256"
  transfer="$(mktemp -d "$exp/runs/.${cell}.pull.XXXXXX")"; trap '[[ -z "${transfer:-}" ]] || rm -rf -- "$transfer"' EXIT
  scp -qr "tufts-login:$remote/out" "tufts-login:$remote/logs" "tufts-login:$remote/MANIFEST.sha256" "tufts-login:$remote/REMOTE.sha256" "tufts-login:$remote/SACCT.txt" "$transfer/"
  verify_phase10_portability_audit_bundle "$transfer" "$job" "$commit" "$expected_manifest"
  (cd "$transfer" && find out logs MANIFEST.sha256 REMOTE.sha256 SACCT.txt -type f -exec sha256sum {} \; | LC_ALL=C sort > LOCAL.sha256 && sha256sum -c LOCAL.sha256)
  mv "$transfer" "$local_dir"; transfer=""; (cd "$local_dir" && sha256sum -c LOCAL.sha256)
  verify_original_phase10_r2_remote "$source_remote"
  ssh tufts-login "test '$remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p10_d_r2_audit_r1' \
    && test '$source_remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p10_d_r2' \
    && rm -rf -- '$remote' '$source_remote' && test ! -e '$remote' && test ! -e '$source_remote'"
  echo "pulled=$local_dir job=$job commit=$commit manifest=$expected_manifest audit_remote_removed=true source_remote_removed=true"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
