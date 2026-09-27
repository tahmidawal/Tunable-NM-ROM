"""Immutable 32-case final NS3D evaluation; all training occurs elsewhere."""
import argparse,datetime,json,os,subprocess
from pathlib import Path
import jax
from final_freeze import verify
from frozen_panel import verify_replay,evaluate
from pilot import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    freeze=verify(cfg);source=Path(cfg['frozen_source_directory'])
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
    replay=verify_replay(cfg,source/'assets',source);write(replay,out/'development_replay.json');assert replay['passed'],'Frozen development replay failed; final cases remain unopened'
    record=dict(freeze=freeze,development_replay=replay,verified_before_final_parameter_generation=True,
        final_access_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip(),
        backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'])
    write(record,out/'freeze_verification.json')
    evaluate(cfg,source/'assets',out,cfg['final_seed_unopened'],cfg['timed_cases'],record)


if __name__=='__main__':main()
