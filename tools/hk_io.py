import jax
import haiku as hk
import functools
import numpy as np


def get_pure_fn(model, c, gc, name):
  """
  get pure function
  return init,apply
  """
  def _forward(*args):
    mod = model(config=c,global_config=gc, name=name)
    return mod(*args)

  init = jax.jit(hk.transform(_forward).init)
  apply = jax.jit(hk.transform(_forward).apply)
  return init,apply


def randomized_params(params):
  new_params = {}
  if not isinstance(params, dict):
    shapes = v.shape
    return np.random.randn(*shapes)
  for k, v in params.items():
    if isinstance(v, dict):
      new_params[k] = randomized_params(v)
    else:
      shapes = v.shape
      new_params[k] = np.random.randn(*shapes)
  return new_params