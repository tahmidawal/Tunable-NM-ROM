#!/usr/bin/env bash
# pull_phase9_terminal_retraction_audit.sh p9_terminal_recovery_audit_r1 JOB COMMIT MANIFEST_SHA256
set -euo pipefail

verify_terminal_retraction_audit_bundle() {
  local root="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ -d "$root/out" && -d "$root/logs" && "$job" =~ ^[0-9]+$ \
     && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || return 20
  [[ "$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)" == "$expected_manifest" ]] || return 21
  (cd "$root" && sha256sum -c REMOTE.sha256) || return 22
  local expected actual
  expected="$(mktemp)"; actual="$(mktemp)"
  printf '%s\n' "logs/$job.err" "logs/$job.out" out/AUDIT-WORK.npz out/AUDIT.json | LC_ALL=C sort > "$expected"
  (cd "$root" && find out logs -type f -print | LC_ALL=C sort) > "$actual"
  cmp -s "$expected" "$actual" || { rm -f "$expected" "$actual"; return 23; }
  awk '{print $2}' "$root/REMOTE.sha256" | LC_ALL=C sort > "$actual.remote"
  cmp -s "$expected" "$actual.remote" || { rm -f "$expected" "$actual" "$actual.remote"; return 24; }
  rm -f "$expected" "$actual" "$actual.remote"
  awk -F'|' -v job="$job" '$1==job && $2=="p9_terminal_recovery_audit_r1" && $3=="COMPLETED" && $4=="0:0" {ok=1} END {exit !ok}' "$root/SACCT.txt" || return 25
  jq -e --arg commit "$commit" --arg job "$job" --arg manifest "$expected_manifest" \
    '.status=="pass" and .negative_aware==true and .expected_commit==$commit and .expected_job==$job
     and .manifest_sha256==$manifest and .decision.optimizer_updates==0
     and .decision.terminal_full_field_accepted==true and .decision.capacity_retracted==true
     and .decision.capacity_reproducible==false and .decision.capacity_accepted==false
     and .decision.capacity_license_complete==false and .decision.g2_licensed==false
     and .decision.complete_original_phase9_result==false and .decision.selection_evaluated==false' "$root/out/AUDIT.json" >/dev/null || return 26
  [[ "$(jq -r .audit_work_npz_sha256 "$root/out/AUDIT.json")" == "$(sha256sum "$root/out/AUDIT-WORK.npz"|cut -d' ' -f1)" ]] || return 27
  [[ "$(jq -r .source_json_sha256 "$root/out/AUDIT.json")" == 8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d ]] || return 28
  [[ "$(jq -r .source_npz_sha256 "$root/out/AUDIT.json")" == d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e ]] || return 29
  [[ "$(jq -r .checkpoint_sha256 "$root/out/AUDIT.json")" == 90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9 ]] || return 30
  [[ "$(jq -r .prior_failed_audit_sha256 "$root/out/AUDIT.json")" == a1c6c0ee23e16da4aab1e4320aedcab4bd7039ec650429908feb15a471f99cec ]] || return 31
}

verify_original_recovery_remote() {
  local recovery_remote="$1" expected_manifest=3b001ee45df4d9889dec99052dee9489623da2ca158b9db9d9006e35739279dd
  ssh tufts-login "cd '$recovery_remote' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$expected_manifest' \
    && ./verify_manifest_file_set.sh . MANIFEST.sha256 ./logs/2735251.out ./logs/2735251.err ./out/PROGRESS.json \
       ./out/phase9_terminal_recovery.json ./out/phase9_terminal_recovery.npz ./out/terminal_checkpoint.pkl ./out/AUDIT.json ./REMOTE.sha256 ./SACCT.txt \
    && sha256sum -c MANIFEST.sha256 && sha256sum -c REMOTE.sha256"
}

main() {
  [[ $# -eq 4 ]] || exit 2
  local cell="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ "$cell" == p9_terminal_recovery_audit_r1 && "$job" =~ ^[0-9]+$ && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
  local here exp remote recovery_remote local_dir transfer state
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; exp="$(dirname "$here")"
  remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
  recovery_remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p9_terminal_recovery_s11_r1"
  local_dir="$exp/runs/$cell"; [[ ! -e "$local_dir" ]] || exit 3
  state="$(ssh tufts-login "sacct -j '$job' -X -n -P -o JobIDRaw,State,ExitCode" | awk -F'|' -v job="$job" '$1==job {print $2"|"$3; exit}')"
  [[ "$state" == "COMPLETED|0:0" ]] || exit 4
  ssh tufts-login "cd '$remote' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$expected_manifest' \
    && sacct -j '$job' -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList -P > SACCT.txt \
    && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256"
  transfer="$(mktemp -d "$exp/runs/.${cell}.pull.XXXXXX")"; trap '[[ -z "${transfer:-}" ]] || rm -rf -- "$transfer"' EXIT
  scp -qr "tufts-login:$remote/out" "tufts-login:$remote/logs" "tufts-login:$remote/MANIFEST.sha256" \
    "tufts-login:$remote/REMOTE.sha256" "tufts-login:$remote/SACCT.txt" "$transfer/"
  verify_terminal_retraction_audit_bundle "$transfer" "$job" "$commit" "$expected_manifest"
  (cd "$transfer" && find out logs MANIFEST.sha256 REMOTE.sha256 SACCT.txt -type f ! -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256 && sha256sum -c LOCAL.sha256)
  mv "$transfer" "$local_dir"; transfer=""
  (cd "$local_dir" && sha256sum -c LOCAL.sha256)
  verify_original_recovery_remote "$recovery_remote"
  ssh tufts-login "test '$remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell' \
    && test '$recovery_remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p9_terminal_recovery_s11_r1' \
    && rm -rf -- '$remote' '$recovery_remote' && test ! -e '$remote' && test ! -e '$recovery_remote'"
  echo "pulled=$local_dir commit=$commit job=$job manifest=$expected_manifest audit_remote_removed=true original_recovery_remote_removed=true"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
