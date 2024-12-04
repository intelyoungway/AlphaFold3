import os
import sys
sys.path.append('/root/sources/AlphaFold3')
from alphafold3_cpu.basics import Configuration


f_json = '/root/sources/AlphaFold3/ref_from_deepmind/model_config.json'
root_config = Configuration(f_json)
# print('----------- root ----------')
# print(root_config)

evoformer_config = root_config.evoformer.pairformer.triangle_multiplication_incoming
# evoformer_config = root_config
# evoformer_config = root_config.sub_config('evoformer')
print(evoformer_config)
