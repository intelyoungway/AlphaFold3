
from tests_gaudi.test_embedders import TestInputEmbedder, TestTemplateEmbedder


if __name__ == '__main__':
  test_modules = [
    TestInputEmbedder,
    #TestTemplateEmbedder
  ]

  for mod in test_modules:
    print(f'# [INFO] testing {mod.__name__}')
    curr_mod = mod()
    curr_mod.setUp()
    curr_mod.test_forward()