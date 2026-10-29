"""Use the reviewed exact warm reader/cold fallback for one fine-prep packet."""
from pathlib import Path
import argparse, json, os, shutil, sys
from prepare2d import preparation_digest, verify_release
from fe_adapter import CAPS, sha_file
from capacity_context2d import validate
from stage_materialization_packet import owned_inputs

ROOT = Path(__file__).resolve().parent


def stage(packet, release_path, expected_release_sha256):
    packet = Path(packet).resolve()
    release_path = Path(release_path).resolve()
    if sha_file(release_path) != expected_release_sha256:
        raise ValueError('exact coordinator release required')
    release = json.loads(release_path.read_text())
    if sha_file(ROOT/'stage_materialization_packet.py') != 'be6aca98791331c90bc33d9001d25eebc69b510795469d1b22b56e43defc1a37':
        raise ValueError('reviewed inherited scratch owner changed')
    inventory_path = packet/'INPUT-INVENTORY.json'
    inventory = json.loads(inventory_path.read_text())
    if (release.get('input_inventory_sha256') != sha_file(inventory_path) or
        release.get('staging_source_sha256') != sha_file(__file__) or
        inventory.get('staging_source_sha256') != sha_file(__file__) or
        release.get('single_execution') is not True or release.get('allocation_seconds') != 1350 or
        release.get('outer_timeout_seconds') != 1335 or release.get('outer_kill_grace_seconds') != 10 or
        release.get('verify_warm_archives') is not True or
        release.get('conservative_recovery_reservations_unchanged') is not True):
        raise ValueError('reviewed stager, exact packet and unchanged cold reserves required')
    for key, name in [('source_inputs_sha256','source_inputs.py'), ('archive_recovery_sha256','git_archive_recovery.py')]:
        if sha_file(ROOT/'reference_inputs'/name) != inventory[key] or release.get(key) != inventory[key]:
            raise ValueError('reviewed reader identity differs')
    _, admitted = verify_release(ROOT/inventory['training_manifest'], release_path)
    validate(release_path, release, admitted, preparation_digest(), CAPS)
    other = release.get('local_other_reserved_bytes')
    if type(other) is not int or other < 0:
        raise ValueError('explicit concurrent reservation required')
    required = sum(r['bytes'] for r in inventory['files'])+inventory['all_unique_cold_recovery_bytes']+10*2**30+64*2**20+other
    free = shutil.disk_usage(ROOT).free
    if free < required:
        raise MemoryError('complete staging/cold recovery/concurrent reserve required')
    out = packet/'payload'
    out.mkdir(exist_ok=False)
    record = {'staging_complete':False, 'release_sha256':expected_release_sha256,
              'required_free_bytes':required, 'initial_free_bytes':free}
    try:
        if 'source_inputs' in sys.modules or 'git_archive_recovery' in sys.modules:
            raise RuntimeError('fresh reviewed helper namespace required')
        sys.path.insert(0, str(ROOT/'reference_inputs'))
        import source_inputs
        os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='gc.auto', GIT_CONFIG_VALUE_0='0')
        binding = {'producer_prefix':'experiments/ns-obstacle-rom', 'producer_git_commit':release['source_commit']}
        with owned_inputs(source_inputs, ROOT.parents[1], binding) as sources:
            sources.materialize(inventory['files'], out)
            record['verified_warm_inputs'] = sources.verified_warm_inputs
            record['cold_archives_materialized'] = list(sources.archives)
        # Separately issued capacity review is hash-bound by the exact release.
        descriptor = release['capacity_review']
        relative = Path(descriptor['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('portable confined capacity review path required')
        review = release_path.parent/relative
        if sha_file(review) != descriptor['sha256']:
            raise ValueError('capacity review changed')
        (out/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(review, out/relative)
        shutil.copyfile(release_path, out/'PREPARATION-RELEASE.json')
        shutil.copyfile(inventory_path, out/'INPUT-INVENTORY.json')
        (out/'SOURCE_COMMIT.txt').write_text(release['source_commit']+'\n')
        (out/'SOURCE-SHA256SUMS').write_text(''.join(sha_file(p)+'  '+str(p.relative_to(out))+'\n'
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SOURCE-SHA256SUMS'))
        record['staging_complete'] = True
    except BaseException as error:
        record['exception'] = repr(error)
        raise
    finally:
        record['actual_staged_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        record['final_free_bytes'] = shutil.disk_usage(ROOT).free
        (packet/'STAGING.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--expected-release-sha256', required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.packet, a.release, a.expected_release_sha256), indent=2))
