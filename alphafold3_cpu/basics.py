import os
import json
import torch

class Configuration:
  def __init__(self, f_json=None):
    if f_json is not None:
      assert os.path.exists(f_json)
      with open(f_json,'rb') as h:
        df_config = json.load(h)
      self.all_params = dict(df_config)
      self.name = 'alphafold3'
  
  @classmethod
  def create(cls, param_dict:dict, name:str):
    instance = cls()
    instance.all_params = param_dict
    instance.name = name
    return instance

  def sub_config(self, k):
    sub_params = self.all_params[k]
    return Configuration.create(sub_params, self.name + '.' + k)
  
  def __getattr__(self, k): # use . to get subtree of params
    assert k in self.all_params.keys()
    if isinstance(self.all_params[k], dict):
      return self.sub_config(k)
    else:
      return self.all_params[k]

  def __str__(self): # show members and their types if in Dict
    res = f'----------- Configuration <{self.name}> -----------\n'
    for k, v in self.all_params.items():
      r = '\033[33m' + f'{k}' + '\033[0m' + ': '
      if isinstance(v, torch.Tensor):
        r += '\n  \033[34m' + 'Tensor' + '\033[0m' + f' [{v.shape}] '
      elif isinstance(v, list):
        r += '\033[34m' + 'List' + '\033[0m' + f' [{len(v)}]'
      elif isinstance(v, dict):
        itms = ''
        for k, val in v.items():
          itms += '\n    \033[35m' + k + '\033[0m'
          itms += '\033[34m' + ' -> ' + '\033[0m'
          if isinstance(val, dict): 
            itms += '\033[34m' + 'Dict' + '\033[0m'
          elif isinstance(val, list):
            itms += f'{len(val)}'
          elif isinstance(val, torch.Tensor):
            itms += f'Tensor[{val.shape}]'
          else:
            itms += f'{val}'
        r += '\n  \033[34m' + 'Dict' + '\033[0m' + f' {itms}'
      else:
        r += '\033[34m' + f'{v}' + '\033[0m'
      res += f'{r}\n'
    return res



