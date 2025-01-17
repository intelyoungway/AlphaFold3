import torch
from torch import Tensor
from torch.nn import Linear, LayerNorm, Module
from torch.nn import functional as F
from alphafold3_cpu.basics import Configuration
from alphafold3_cpu.miscs import permute_final_dims


# TriangleMultiplication on CPU
class TriangleMultiplication(Module):
  def __init__(self, 
    c:Configuration, 
    gc:Configuration, 
    c_hidden_mul=128
  ):
    super(TriangleMultiplication, self).__init__()
    self.config = c # use_glu_kernel: True
    self.global_config = gc
    self.c_z = 64
    self.c_hidden = c_hidden_mul
    # PairStack.c_hidden_tri_mul
    # <- PairformerStackBlock.c_hidden_mul
    # <- PairformerStack.c_hidden_mul
    # <- pairformer_stack.c_hidden_mul

    # use_glu_kernel: Not used in PyTorch, but TPP kernel instead


    # determine outgoing or incoming
    if c.equation == 'ikc,jkc->ijc': # outgoing
      self._outgoing = True
      self.equation = 'cik,cjk->cij'
    else: # incoming: c.equation == kjc,kic->ijc
      self._outgoing = False
      self.equation = 'ckj,cki->cij'
    # build bricks for topo
    self.gate = Linear( # linear_g
      self.c_z, self.c_hidden, init="gating")
      
    self.projection = Linear( # linear_p
      self.c_z, self.c_hidden)

    self.left_norm_input = LayerNorm(self.c_z) # layer_norm_in
    self.center_norm = LayerNorm(self.c_hidden) # layer_norm_out

    self.output_projection = Linear( # linear_z
      self.c_hidden, self.c_z, init="final")
    self.gating_linear = Linear(
      self.c_z, self.c_z, init='gating')
    

  def forward(self, act:Tensor, mask:Tensor=None):
    # [TODO] slicing/chunking strategy by Ligo-biosci maybe useful during training
    # if mask is None:
    #   mask = act.new_ones(act.shape[:-1])
    mask = mask.unsqueeze(-1)
    act = self.left_norm_input(act)
    input_act = act
    proj = self.projection(act)
    act = self.gate(act)
    proj = F.sigmoid(proj)
    proj *= mask # [TODO] does it exist in dm.af3
    p = permute_final_dims(proj, (2, 0, 1))
    return p
