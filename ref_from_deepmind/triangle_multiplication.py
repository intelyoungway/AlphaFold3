from ref_from_deepmind import base_config
from typing import Literal
import haiku as hk
from ref_from_deepmind.jax.gated_linear_unit import gated_linear_unit
from ref_from_deepmind import model_config
from ref_from_deepmind.components import haiku_modules as hm
import jax
import jax.numpy as jnp


class TriangleMultiplication(hk.Module):
  """Triangle Multiplication."""

  class Config(base_config.BaseConfig):
    equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
    use_glu_kernel: bool = True

  def __init__(
      self, config: Config, global_config: model_config.GlobalConfig, *, name
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask):
    """Applies Module.

    Args:
      act: The activation.
      mask: The mask.

    Returns:
      Outputs, should have same shape/type as output_act
    """
    mask = mask[None, ...]
    num_channels = act.shape[-1] # pt.triangularmulti.c_z, 32
    
    equation = {
        'ikc,jkc->ijc': 'cik,cjk->cij', # outgoing
        'kjc,kic->ijc': 'ckj,cki->cij', # incoming
    }[self.config.equation]

    act = hm.LayerNorm(name='left_norm_input')(act)
    input_act = act

    if self.config.use_glu_kernel: # true
      weights_projection, _ = hm.haiku_linear_get_params(
          act, num_output=num_channels * 2, name='projection'
      )
      weights_gate, _ = hm.haiku_linear_get_params(
          act,
          num_output=num_channels * 2,
          initializer=self.global_config.final_init,
          name='gate',
      )
      weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)

      projection = gated_linear_unit.gated_linear_unit(
          x=act,
          weight=weights_glu,
          activation=jax.nn.sigmoid,
          implementation=None,
      )
      projection = jnp.transpose(projection, (2, 0, 1))
      projection *= mask
    else:
      projection = hm.Linear(num_channels * 2, name='projection')(act)
      projection = jnp.transpose(projection, (2, 0, 1))
      projection *= mask

      gate = hm.Linear(
          num_channels * 2,
          name='gate',
          bias_init=1.0,
          initializer=self.global_config.final_init,
      )(act)
      gate = jnp.transpose(gate, (2, 0, 1))
      projection *= jax.nn.sigmoid(gate)

    projection = projection.reshape(num_channels, 2, *projection.shape[1:])
    a, b = jnp.split(projection, 2, axis=1)
    a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
    act = jnp.einsum(equation, a, b)
    act = hm.LayerNorm(name='center_norm', axis=0, param_axis=0)(act)

    act = jnp.transpose(act, (1, 2, 0))
    act = hm.Linear(
        num_channels,
        initializer=self.global_config.final_init,
        name='output_projection',
    )(act)

    gate_out = hm.Linear(
        num_channels,
        name='gating_linear',
        bias_init=1.0,
        initializer=self.global_config.final_init,
    )(input_act)
    act *= jax.nn.sigmoid(gate_out)

    return act