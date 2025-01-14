import sys
sys.path.append('/home/intel/AlphaFold3')
from ref_from_deepmind.triangle_multiplication import TriangleMultiplication
from tools.hk_io import get_pure_fn, get_fn_with_sample
import jax
from jax import numpy as jnp
import numpy as np

from alphafold3_cpu.basics import Configuration

print('##### load configurations')
f_json = '/home/intel/AlphaFold3/ref_from_deepmind/model_config.json'
root_config = Configuration(f_json)
gc = root_config.global_config
c = root_config.evoformer.pairformer.triangle_multiplication_outgoing

seq_z = 32
seq_len = 100
bs = 1

act = np.ones((seq_len, seq_len, seq_z))
mask = np.ones((seq_len, seq_len))

init, apply = get_pure_fn(TriangleMultiplication, c, gc, name='TriangleMultiplication')
# apply = get_fn_with_sample(TriangleMultiplication, c, gc)

rng = jax.random.PRNGKey(0)
params = init(rng, act, mask)
res = apply(params, rng, act, mask)
jax.tree_map(lambda x:x.block_until_ready(), res)
res = jax.tree_map(lambda x:np.asarray(x), res)
print(res.shape)