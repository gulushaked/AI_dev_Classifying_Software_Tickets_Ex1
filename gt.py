def correct_category(num: int, category: str) -> bool:

    GT = {1: 'security and access control',
          2: 'lacking feature',
          3: 'interface',
          4: 'security and access control',
          5: 'stability',
          6: 'logic defect',
          7: 'performance',
          8: 'data',
          9: 'interface',
          10: 'logic defect',
          11: 'logic defect',
          12: 'interface',
          13: 'logic defect',
          14: 'interface',
          15: 'configuration',
          16: 'security and access control',
          17: 'configuration',
          18: 'stability',
          19: 'data'}

    return GT[num] == category
