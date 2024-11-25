import utils

MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [(utils.sigmoid, 0.015, 450, 1.0),
               (utils.sigmoid, 0.35, 25, 1.0)]}