#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/pull_phase10_d.sh"
fixture="$(mktemp -d)"; trap 'rm -rf -- "$fixture"' EXIT; root="$fixture/good"; job=12345; commit=1111111111111111111111111111111111111111
mkdir -p "$root/out" "$root/logs"; printf fixture > "$root/MANIFEST.sha256"
for name in PROGRESS.json phase10_d.json; do printf '{}\n' > "$root/out/$name"; done; printf npz > "$root/out/phase10_d.npz"; printf control > "$root/out/initial_control.npz"; printf work > "$root/out/work_checkpoint.pkl"
json="$(sha256sum "$root/out/phase10_d.json"|cut -d' ' -f1)"; npz="$(sha256sum "$root/out/phase10_d.npz"|cut -d' ' -f1)"; control="$(sha256sum "$root/out/initial_control.npz"|cut -d' ' -f1)"; work="$(sha256sum "$root/out/work_checkpoint.pkl"|cut -d' ' -f1)"; manifest="$(sha256sum "$root/MANIFEST.sha256"|cut -d' ' -f1)"
jq -n --arg commit "$commit" --arg job "$job" --arg manifest "$manifest" --arg json "$json" --arg npz "$npz" --arg control "$control" --arg work "$work" \
  '{status:"pass",negative_aware:true,expected_commit:$commit,expected_job:$job,manifest_sha256:$manifest,source_json_sha256:$json,source_npz_sha256:$npz,initial_control_sha256:$control,work_checkpoint_sha256:$work,
    decision:{architecture_increase_licensed:false,g2_licensed:false,selection_evaluated:false,optimizer_updates:0},capacity:{accepted:false,used:false}}' > "$root/out/AUDIT.json"
printf 'jax_backend=gpu\n' > "$root/logs/$job.out"; : > "$root/logs/$job.err"; printf 'JobIDRaw|JobName|State|ExitCode|\n%s|p10_d_r2|COMPLETED|0:0|\n' "$job" > "$root/SACCT.txt"
refresh(){ (cd "$1" && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256); }; refresh "$root"; verify_phase10_bundle "$root" "$job" "$commit" "$manifest"
for kind in extra sacct manifest architecture capacity work control; do one="$fixture/$kind"; cp -a "$root" "$one"; expected="$manifest"
  case "$kind" in extra) printf bad > "$one/out/extra"; refresh "$one";; sacct) sed -i 's/COMPLETED|0:0/FAILED|1:0/' "$one/SACCT.txt";; manifest) expected="$(printf '0%.0s' {1..64})";;
    architecture) jq '.decision.architecture_increase_licensed=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    capacity) jq '.capacity.accepted=true' "$one/out/AUDIT.json" > "$one/a"; mv "$one/a" "$one/out/AUDIT.json"; refresh "$one";;
    work) printf bad >> "$one/out/work_checkpoint.pkl"; refresh "$one";;
    control) printf bad >> "$one/out/initial_control.npz"; refresh "$one";; esac
  if verify_phase10_bundle "$one" "$job" "$commit" "$expected" >/dev/null 2>&1; then echo "corruption accepted: $kind" >&2; exit 1; fi
done
echo phase10_pull_contract=pass corruptions=7
