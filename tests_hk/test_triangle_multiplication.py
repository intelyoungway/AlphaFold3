import sys
sys.path.append('/home/intel/AlphaFold3')
from ref_from_deepmind.triangle_multiplication import TriangleMultiplication
from tools.hk_io import get_pure_fn, randomized_params
import jax
import numpy as np

from alphafold3_cpu.basics import Configuration

general_seed = 123
np.random.seed(general_seed)
print('##### load configurations')
f_json = '/home/intel/AlphaFold3/ref_from_deepmind/model_config.json'
root_config = Configuration(f_json)
gc = root_config.global_config
c = root_config.evoformer.pairformer.triangle_multiplication_outgoing

seq_z = 32
seq_len = 100
bs = 1

act = np.random.randn(seq_len, seq_len, seq_z)
mask = np.random.randn(seq_len, seq_len)
print(np.asarray(act)[0,0])

init, apply = get_pure_fn(
  TriangleMultiplication, c, gc, 
  name='TriangleMultiplication',
  is_jit=False)
# apply = get_fn_with_sample(TriangleMultiplication, c, gc)

rng = jax.random.PRNGKey(general_seed)
params = init(rng, act, mask)
params = randomized_params(params) # ensure to make some change into result

res = apply(params, rng, act, mask)
jax.tree_map(lambda x:x.block_until_ready(), res)
res = jax.tree_map(lambda x:np.asarray(x), res)
print(res[0,0]) # act: seq_len x seq_len x seq_z