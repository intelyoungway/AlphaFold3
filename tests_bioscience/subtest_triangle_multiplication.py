import sys
sys.path.append('/home/intel/AlphaFold3')
import torch
import unittest
from src.models.components.triangular_multiplicative_update import FusedTriangleMultiplicationOutgoing


class TestTriangleMultiplication(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_tokens = 64
        self.c_z = 32
        self.c_hidden = 128
        self.module = FusedTriangleMultiplicationOutgoing(self.c_z, self.c_hidden)
    
        self.module.to(device='cpu')
        self.mode = 'eager_infer'
        # self.dtype = torch.float 
        self.dtype = torch.bfloat16
        if self.mode == 'eager_infer': # [CPU] infer
            self.module.eval()
        else:
            self.module.train()


    def test_forward(self):
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        if self.mode == 'eager_infer':
            with torch.inference_mode():
              with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                z_out = self.module(z, mask)
        else:
            with torch.autocast(device_type='cpu', dtype=self.dtype, enabled=True):
                z_out = self.module(z, mask)
        self.assertEqual(z_out.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_z))


if __name__ == "__main__":
    unittest.main()
