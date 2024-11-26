# Define the new sigmoid function with adjusted steepness
import numpy as np
from matplotlib import pyplot as plt

import utils

center = 25
k = 2.0

x = np.linspace(0, 50, 500)
# plt.figure(figsize=(10, 6))
plt.plot(x, utils.sigmoid(x, k, center), color='purple')
plt.axvline(center, color='gray', linestyle=':')
plt.grid()
plt.show()
