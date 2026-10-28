"""Dense versus blocked all-point algebra only; no physical fit admission."""
from pathlib import Path
import json
import tempfile
import numpy as np
from smoke import polynomial_rule
from quadrature import output_moments, fit_positive
from quadrature_blocked import build_blocks, fit_positive_blocks

rule,points=polynomial_rule()
coefficients=np.array([[.1,.2],[.3,-.1],[-.2,.4]],dtype=np.float64)
dense=output_moments(rule,coefficients,points)
reference=fit_positive(dense,rule.weights,32)
records=[]
with tempfile.TemporaryDirectory() as root:
 for block_points in (7,16,64):
  blocks=build_blocks(rule,coefficients,points,Path(root)/str(block_points),block_points)
  restored=np.concatenate([np.array(a) for _,_,a in blocks.iter_blocks()],axis=1)
  assert np.array_equal(restored,dense)
  result=fit_positive_blocks(blocks,rule.weights,32)
  assert result['passed']==reference['passed']
  assert result['selected']==reference['selected']
  np.testing.assert_allclose(result['row_scaling'],reference['row_scaling'],rtol=5e-15,atol=1e-15)
  ids=result['support'];weights=np.array(result['weights'])
  error=np.linalg.norm((dense[:,ids]@weights-dense@rule.weights)/np.array(reference['row_scaling']))
  assert error<1e-7
  records.append({'block_points':block_points,'identical_raw_moments':True,
    'identical_support':True,'scaled_full_moment_error':float(error),
    'max_weight_difference':float(np.max(np.abs(np.array(result['selected_weights'])-reference['selected_weights'])))})
print(json.dumps({'scope':'synthetic all-point algebra','physical_admission':False,'cases':records},indent=2))
