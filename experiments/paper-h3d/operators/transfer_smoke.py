"""Check DeepONet refactor parity and native-sensor continuous-trunk transfer."""
from pathlib import Path
import json,subprocess,types
import numpy as np
import jax
import jax.numpy as jnp
from . import models3d as M,extra_models3d as E,heat_adapter as H
old=types.ModuleType('operators.old_extra');old.__package__='operators'
source=subprocess.check_output(['git','show','5933b3706c119e8ffbcfdf98db3940464af7b4fc:experiments/paper-h3d/operators/extra_models3d.py'],text=True)
exec(compile(source,'frozen_extra_models3d.py','exec'),old.__dict__)
spec=dict(kind='deeponet3d',width=2,levels=2,pool_bins=2,rank=4,trunk_width=8,periodic=False,coordinate_channels=[-3,-2,-1])
p=M.init_model(jax.random.PRNGKey(57),spec,4,5);prior=old.init_extra(jax.random.PRNGKey(57),spec,4,5)
for a,b in zip(jax.tree_util.tree_leaves(p),jax.tree_util.tree_leaves(prior)):np.testing.assert_array_equal(a,b)
n=8;fine=16;scale=.3;rng=np.random.default_rng(91);u=jnp.asarray(rng.normal(size=(fine-1,)*3));coarse=u[1::2,1::2,1::2]
x=jnp.concatenate((coarse[...,None]/scale,H.coordinates(n)),axis=-1)[None]
a=np.asarray(E.apply_extra(p,x,spec));b=np.asarray(old.apply_extra(p,x,spec));np.testing.assert_array_equal(a,b)
native=np.asarray(H.native_sensor_deeponet_engine(p,spec,scale,n,n)(coarse));standard=np.asarray(H.engine(p,spec,scale,n)(coarse))
np.testing.assert_allclose(native,standard,rtol=2e-13,atol=2e-13)
transferred=np.asarray(H.native_sensor_deeponet_engine(p,spec,scale,n,fine)(u));np.testing.assert_allclose(transferred[:,1::2,1::2,1::2],native,rtol=2e-13,atol=2e-13)
assert transferred.shape==(6,15,15,15)
fno=dict(kind='fno3d',width=2,modes=[2,2,2],depth=1,padding=3);fp=M.init_model(jax.random.PRNGKey(81),fno,4,5)
padding=round(fine/n*(n-1+fno['padding']))-(fine-1);assert padding==5 and (n-1+3)/n==(fine-1+padding)/fine
output=H.engine(fp,{**fno,'padding':padding},scale,fine)(u);assert output.shape==(6,15,15,15) and np.isfinite(output).all()
out=Path(__file__).resolve().parents[1]/'smokes/transfer';out.mkdir(exist_ok=True)
(out/'audit.json').write_text(json.dumps(dict(passed=True,old_deeponet_source='5933b3706c119e8ffbcfdf98db3940464af7b4fc',initialization_identical=True,native_application_bitwise_identical=True,native_adapter_parity=True,continuous_trunk_native_node_parity=True,physical_padding_shape_and_domain_passed=True),indent=2)+'\n')
print('TRANSFER_SMOKE_PASS',flush=True)
