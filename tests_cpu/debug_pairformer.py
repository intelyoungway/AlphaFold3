import sys
sys.path.append('/home/intel/AlphaFold3')
import torch
import unittest
from src.models.pairformer import PairformerStack


class TestPairformer:
    def __init__(self):
        self.batch_size = 1
        self.n_tokens = 55
        self.c_s = 64
        self.c_z = 32
        self.no_blocks = 1
        self.c_hidden_mul = 64
        self.c_hidden_pair_att = 16
        self.no_heads_pair = 4
        self.no_heads_single_att = 2
        self.transition_n = 1
        self.pair_dropout = 0.25
        self.fuse_projection_weights = False
        self.module = PairformerStack(self.c_s, self.c_z)
    
        self.module.to(device='cpu')
        self.mode = 'eager_infer'
        # self.dtype = torch.float 
        self.dtype = torch.bfloat16
        if self.mode == 'eager_infer': # [CPU] infer
            self.module.eval()
        else:
            self.module.train()


    def test_forward(self):
        s = torch.randn((self.batch_size, self.n_tokens, self.c_s))
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        single_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        if self.mode == 'eager_infer':
            with torch.no_grad():
              with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)
        else:
            with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)

    def test_forward_with_different_shapes(self):
        s = torch.randn((self.batch_size, self.n_tokens // 2, self.c_s))
        z = torch.randn((self.batch_size, self.n_tokens // 2, self.n_tokens // 2, self.c_z))
        single_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens // 2))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens // 2, self.n_tokens // 2))
        if self.mode == 'eager_infer':
            with torch.no_grad():
              with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)
        else:
            with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)

    def test_forward_with_zero_input(self):
        s = torch.zeros((self.batch_size, self.n_tokens, self.c_s))
        z = torch.zeros((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        single_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        if self.mode == 'eager_infer':
            with torch.no_grad():
              with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)
        else:
            with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)

    def test_forward_with_extreme_values(self):
        s = torch.full((self.batch_size, self.n_tokens, self.c_s), 1e6)
        z = torch.full((self.batch_size, self.n_tokens, self.n_tokens, self.c_z), -1e6)
        single_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        if self.mode == 'eager_infer':
            with torch.no_grad():
              with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)
        else:
            with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                s_out, z_out = self.module(s, z, single_mask, pair_mask)


if __name__ == "__main__":
    tester = TestPairformer()
    tester.test_forward()
