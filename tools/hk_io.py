import jax
import haiku as hk
import functools


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


def get_fn_with_sample(model, c, gc):
  @hk.transform
  def fwd_fn(feed_dict):
    return model(c,gc)(**feed_dict)
  
  return functools.partial(
    jax.jit(fwd_fn.apply),
    None
  )