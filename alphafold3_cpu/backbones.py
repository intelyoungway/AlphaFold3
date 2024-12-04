import torch
from torch.nn import Module


# TriangleMultiplication on CPU
class TriangleMultiplication(Module):
  def __init__(self, 
    c, # config for this class
    gc): # config in global env
    super(TriangleMultiplication, self).__init__()
    self.config = c
    self.global_config = gc
    # determine outgoing or incoming
    if c.equation == 'ikc,jkc->ijc':
      self._outgoing = True
    else:
      self._outgoing = False
    # build bricks for topo
    self.gate = Linear( # linear_g
      self.c_z, self.c_z, init="gating") # [TODO] change to use_glu_kernel
    self.output_projection = Linear( # linear_z
      self.c_hidden, self.c_z, init="final")

    self.layer_norm_in = LayerNorm(self.c_z)
    self.layer_norm_out = LayerNorm(self.c_hidden)

    self.sigmoid = nn.Sigmoid()