import os
import sys
sys.path.append('/home/intel/AlphaFold3')
from alphafold3_cpu.basics import Configuration


f_json = '/home/intel/AlphaFold3/ref_from_deepmind/model_config.json'
root_config = Configuration(f_json)
print(root_config)
gc = root_config.global_config
# print(gc)

trimul_outgoing = root_config.evoformer.pairformer.triangle_multiplication_outgoing
print(trimul_outgoing)
# trimul_incoming = root_config.evoformer.pairformer.triangle_multiplication_incoming
# print(trimul_incoming)
# print(trimul_incoming.equation)
# atom_transformer_cfg = root_config.evoformer.per_atom_conditioning.atom_transformer
# print(atom_transformer_cfg)