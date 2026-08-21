#!/usr/bin/env bash
set -euo pipefail
verify_phase10_bundle() {
  local root="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ "$job" =~ ^[0-9]+$ && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || return 20
  [[ "$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)" == "$expected_manifest" ]] || return 21
  (cd "$root" && sha256sum -c REMOTE.sha256) || return 22
  local expected actual; expected="$(mktemp)"; actual="$(mktemp)"
  printf '%s\n' "logs/$job.err" "logs/$job.out" out/AUDIT.json out/PROGRESS.json out/initial_control.npz out/phase10_d.json out/phase10_d.npz out/work_checkpoint.pkl | sort > "$expected"
  (cd "$root" && find out logs -type f -print | sort) > "$actual"; cmp -s "$expected" "$actual" || { rm -f "$expected" "$actual"; return 23; }
  awk '{print $2}' "$root/REMOTE.sha256" | sort > "$actual.remote"; cmp -s "$expected" "$actual.remote" || { rm -f "$expected" "$actual" "$actual.remote"; return 24; }
  rm -f "$expected" "$actual" "$actual.remote"
  awk -F'|' -v job="$job" '$1==job && $2=="p10_d_r2" && $3=="COMPLETED" && $4=="0:0" {ok=1} END {exit !ok}' "$root/SACCT.txt" || return 25
  jq -e --arg commit "$commit" --arg job "$job" --arg manifest "$expected_manifest" \
    '.status=="pass" and .negative_aware==true and .expected_commit==$commit and .expected_job==$job and .manifest_sha256==$manifest
     and .decision.architecture_increase_licensed==false and .decision.g2_licensed==false
     and .decision.selection_evaluated==false and .decision.optimizer_updates==0
     and .capacity.accepted==false and .capacity.used==false' "$root/out/AUDIT.json" >/dev/null || return 26
  [[ "$(jq -r .source_json_sha256 "$root/out/AUDIT.json")" == "$(sha256sum "$root/out/phase10_d.json"|cut -d' ' -f1)" ]] || return 27
  [[ "$(jq -r .source_npz_sha256 "$root/out/AUDIT.json")" == "$(sha256sum "$root/out/phase10_d.npz"|cut -d' ' -f1)" ]] || return 28
  [[ "$(jq -r .work_checkpoint_sha256 "$root/out/AUDIT.json")" == "$(sha256sum "$root/out/work_checkpoint.pkl"|cut -d' ' -f1)" ]] || return 29
  [[ "$(jq -r .initial_control_sha256 "$root/out/AUDIT.json")" == "$(sha256sum "$root/out/initial_control.npz"|cut -d' ' -f1)" ]] || return 30
}
main() {
  [[ $# -eq 4 ]] || exit 2; local cell="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ "$cell" == p10_d_r2 ]] || exit 2
  local here exp remote local_dir transfer state; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; exp="$(dirname "$here")"
  remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"; local_dir="$exp/runs/$cell"; [[ ! -e "$local_dir" ]] || exit 3
  state="$(ssh tufts-login "sacct -j '$job' -X -n -P -o JobIDRaw,State,ExitCode" | awk -F'|' -v job="$job" '$1==job {print $2"|"$3;exit}')"; [[ "$state" == COMPLETED\|0:0 ]] || exit 4
  ssh tufts-login "cd '$remote' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$expected_manifest' \
    && sacct -j '$job' -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList -P > SACCT.txt \
    && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256"
  transfer="$(mktemp -d "$exp/runs/.${cell}.pull.XXXXXX")"; trap '[[ -z "${transfer:-}" ]] || rm -rf -- "$transfer"' EXIT
  scp -qr "tufts-login:$remote/out" "tufts-login:$remote/logs" "tufts-login:$remote/MANIFEST.sha256" "tufts-login:$remote/REMOTE.sha256" "tufts-login:$remote/SACCT.txt" "$transfer/"
  verify_phase10_bundle "$transfer" "$job" "$commit" "$expected_manifest"
  (cd "$transfer" && find out logs MANIFEST.sha256 REMOTE.sha256 SACCT.txt -type f -exec sha256sum {} \; | sort > LOCAL.sha256 && sha256sum -c LOCAL.sha256)
  mv "$transfer" "$local_dir"; transfer=""; (cd "$local_dir" && sha256sum -c LOCAL.sha256)
  ssh tufts-login "test '$remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/p10_d_r2' && rm -rf -- '$remote' && test ! -e '$remote'"
  echo "pulled=$local_dir job=$job commit=$commit manifest=$expected_manifest remote_removed=true"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
