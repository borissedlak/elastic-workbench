import sys

import pandas as pd
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
# from pgmpy.factors.continuous import ContinuousGaussian
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter

try:
    model = XMLBIFReader("model.xml").get_model()
except FileNotFoundError as a:

    data = {
        'X': [2.3, 3.5, 4.2, 5.1, 3.9],
        'Y': [1.2, 2.1, 1.9, 2.8, 2.4],
        'Z': [0.7, 1.0, 0.6, 0.9, 0.8]
    }
    df = pd.DataFrame(data)

    model = LinearGaussianBayesianNetwork([('X', 'Y'), ('X', 'Z'), ('Y', 'Z')])
    model_name = 'model_Xavier_GPU.xml'
    XMLBIFWriter(model).write_xmlbif("model.xml")
    model.fit(df)

# Step 4: Access the learned CPDs (Continuous Gaussian distributions)
for cpd in model.get_cpds():
    print(cpd)

sys.exit()

# Define the new sigmoid function with adjusted steepness
import numpy as np
from matplotlib import pyplot as plt

import utils


center = 25
k = 0.35

x = np.linspace(0, 50, 500)
# plt.figure(figsize=(10, 6))
plt.plot(x, utils.sigmoid(x, k, center), color='purple')
plt.axvline(center, color='gray', linestyle=':')
plt.grid()
plt.show()
