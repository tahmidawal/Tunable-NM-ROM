"""Check shared-data exposure, trained artifacts and both capacity diagnoses."""
import json
from pathlib import Path
import pickle
import numpy as np
from audit import audit,digest
from audit_stationarity import inspect


def parameter_count(tree):
    if isinstance(tree,dict):return sum(parameter_count(v) for v in tree.values())
    if isinstance(tree,(tuple,list)):return sum(parameter_count(v) for v in tree)
    return int(np.asarray(tree).size)


def run(archive):
    archive=Path(archive);results={};trainings={};initials={}
    for rank in [128,256]:
        folder=archive/'out'/f'r{rank}'
        training=json.loads((folder/'training.json').read_text());trainings[rank]=training
        assert training['complete'] and training['rank']==rank and training['k']==16
        assert training['validation_access'] is False
        assert len(training['training_case_ids'])==128
        assert training['config']['bank_hidden']==training['config']['head_hidden']==256
        assert len(training['phases'])==len(training['config']['phases'])==4
        for phase,spec in zip(training['phases'],training['config']['phases']):
            assert phase['finite'] and phase['updates']==spec['steps']
            assert phase['sampled_source_exposures']==spec['steps']*64
            if phase['phase']!='head':assert phase['sampled_field_entry_exposures']==spec['steps']*64*2048
            assert sum(block['last_update']-block['first_update']+1 for block in phase['optimizer_blocks'])==phase['updates']
        for checkpoint in training['checkpoints']:
            assert digest(folder/checkpoint['path'])==checkpoint['sha256']
        assert digest(folder/'basis.npz')==training['basis_sha256']
        with np.load(folder/'basis.npz') as a:
            R=a['R'];C=a['coefficient_directions']
            assert C.shape==(rank,32)
            assert np.linalg.norm(C.T@R.T@R@C-np.eye(32))<1e-9
        initials[rank]=pickle.loads((folder/'initial.pkl').read_bytes())
        summary=audit(archive/'out',f'diagnosis_r{rank}',archive/'out/data')
        stationarity=inspect(archive,f'diagnosis_r{rank}','data/validation')
        results[rank]=dict(summary=summary,stationarity=stationarity,
            parameter_count=parameter_count(initials[rank]['params']),final_checkpoint_bytes=(folder/'final.pkl').stat().st_size,
            phase_optimizer_seconds={p['tag']:p['optimizer_seconds'] for p in training['phases']},
            training_metrics=training['training_metrics'])
    np.testing.assert_array_equal(initials[128]['Z_tr'],initials[256]['Z_tr'])
    np.testing.assert_array_equal(initials[128]['params']['B'],initials[256]['params']['B'])
    for key in ['g','h']:
        for pair128,pair256 in zip(initials[128]['params'][key][:-1],initials[256]['params'][key][:-1]):
            for a,b in zip(pair128,pair256):np.testing.assert_array_equal(a,b)
    assert trainings[128]['dataset_index_sha256']==trainings[256]['dataset_index_sha256']
    assert trainings[128]['config']==trainings[256]['config']
    return dict(complete=True,matched_data_and_exposure_audit_pass=True,results=results,
        limits='One initialization seed and fixed update budgets; no deployment FNO comparison or final-cohort validation. Both use common hidden width256, not the historical architecture.')
